"""
localdistill — REST API server for dashboard + curation.

Endpoints:
  GET  /health
  GET  /api/status         — training stats, curated count, recent convos
  GET  /api/conversations   — paginated list with scores
  POST /api/conversations/:id/rate — promote/demote/neutral
  GET  /api/conversations/:id       — full conversation detail
  GET  /api/search?q=...   — RAG search
  GET  /api/export?format=chatml — download training dataset
"""

import sys, os, json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "proxy"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from db import get_db, get_conversation, get_unreviewed_conversations
from quality import AUTO_PROMOTE_THRESHOLD, should_auto_promote, should_auto_exclude

from fastapi.responses import HTMLResponse

app = FastAPI(title="localdistill-api")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><title>localdistill</title><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<style>body{font:14px monospace;background:#111;color:#eee;padding:2em;max-width:700px;margin:auto}
h1{color:#0f0}h3{color:#aaa;margin-top:2em}table{width:100%;border-collapse:collapse}td,th{padding:8px 12px;text-align:left;border-bottom:1px solid #333}
.green{color:#0f0}.yellow{color:#cc0}.dim{color:#666}.bar{height:4px;background:#333;margin:4px 0}.bar div{height:100%;background:#0f0}</style></head>
<body><h1>localdistill</h1><div id="status">Loading...</div>
<script>
fetch('/api/status').then(r=>r.json()).then(d=>{
  let h='<h3>Proxy</h3><p class="green">'+d.proxy+'</p>';
  h+='<h3>Conversations</h3><p>Total: <b>'+d.total_conversations+'</b> | Curated: <b class="green">'+d.curated_count+'</b></p>';
  h+='<div class="bar"><div style="width:'+Math.min(100,d.curated_count*10)+'%"></div></div>';
  h+='<p class="dim">Quality threshold: '+d.auto_promote_threshold+' (auto-promote)</p>';
  h+='<h3>Training</h3><p>Runs: <b>'+d.training_runs+'</b> | Status: <b class="yellow">'+d.training_status+'</b></p>';
  h+='<p class="dim">Model: '+d.model+'</p>';
  h+='<p class="dim">Run: <code>./train.sh</code></p>';
  h+='<h3>Recent</h3><table>'+d.recent.map(r=>'<tr><td class="dim">#'+r.id+'</td><td>'+r.model+'</td><td class="'+(r.score>=0.7?'green':'yellow')+'">'+r.score.toFixed(2)+'</td><td class="dim">'+r.status+'</td></tr>').join('')+'</table>';
  document.getElementById('status').innerHTML=h;
});
</script></body></html>"""

@app.get("/")
async def root():
    return HTMLResponse(DASHBOARD_HTML)

DATA_DIR = Path("/data")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    db = get_db()
    curated = db.execute("SELECT COUNT(*) as cnt FROM curated_training WHERE used_in_training = 0").fetchone()["cnt"]
    total = db.execute("SELECT COUNT(*) as cnt FROM conversations WHERE status = 'completed'").fetchone()["cnt"]
    runs = db.execute("SELECT COUNT(*) as cnt FROM training_runs").fetchone()["cnt"]
    last_run = db.execute("SELECT status FROM training_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    
    recent = db.execute("""SELECT substr(id,1,8) as id, '–' as model, 
        COALESCE(quality_score, 0.5) as score, status
        FROM conversations ORDER BY created_at DESC LIMIT 5""").fetchall()
    db.close()
    
    import os
    model = os.environ.get("LOCALDISTILL_MODEL", "not set")
    proxy_ok = os.system("curl -sf http://proxy:8787/health >/dev/null 2>&1") == 0
    
    return {
        "proxy": "healthy" if proxy_ok else "down",
        "total_conversations": total,
        "curated_count": curated,
        "auto_promote_threshold": AUTO_PROMOTE_THRESHOLD,
        "training_runs": runs,
        "training_status": last_run["status"] if last_run else "not started (run ./train.sh)",
        "model": model,
        "recent": [dict(r) for r in recent],
    }


@app.get("/api/conversations")
async def list_conversations(page: int = Query(1, ge=1), per_page: int = Query(20, le=100)):
    db = get_db()
    offset = (page - 1) * per_page
    rows = db.execute(
        """SELECT id, title, turn_count, quality_score, status, created_at
           FROM conversations
           WHERE status = 'completed'
           ORDER BY created_at DESC
           LIMIT ? OFFSET ?""",
        (per_page, offset)
    ).fetchall()
    
    total = db.execute("SELECT COUNT(*) as cnt FROM conversations WHERE status = 'completed'").fetchone()
    db.close()
    
    return {
        "page": page,
        "per_page": per_page,
        "total": total["cnt"],
        "items": [dict(r) for r in rows],
    }


@app.get("/api/conversations/{conv_id}")
async def get_conv_detail(conv_id: str):
    data = get_conversation(conv_id)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return data


@app.post("/api/conversations/{conv_id}/rate")
async def rate_conversation(conv_id: str, rating: str = Query(..., regex="^(good|bad|neutral)$")):
    db = get_db()
    conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not conv:
        db.close()
        raise HTTPException(status_code=404)
    
    if rating == "good":
        new_score = max(conv["quality_score"] or 0.5, AUTO_PROMOTE_THRESHOLD + 0.01)
        db.execute("UPDATE conversations SET quality_score = ? WHERE id = ?", (new_score, conv_id))
        db.execute("""INSERT OR REPLACE INTO curated_training 
            (conversation_id, promoted_by, quality_score, promoted_at)
            VALUES (?, 'manual', ?, ?)""",
            (conv_id, new_score, datetime.now(timezone.utc).isoformat()))
    elif rating == "bad":
        db.execute("UPDATE conversations SET quality_score = 0.10 WHERE id = ?", (conv_id,))
        db.execute("DELETE FROM curated_training WHERE conversation_id = ?", (conv_id,))
    elif rating == "neutral":
        db.execute("UPDATE conversations SET quality_score = NULL WHERE id = ?", (conv_id,))
        db.execute("DELETE FROM curated_training WHERE conversation_id = ?", (conv_id,))
    
    db.commit()
    db.close()
    return {"conversation_id": conv_id, "rating": rating, "ok": True}


@app.get("/api/search")
async def search(q: str = Query(..., min_length=2), limit: int = Query(5, le=20)):
    db = get_db()
    rows = db.execute(
        """SELECT DISTINCT c.id, c.title, c.quality_score,
                  substr(i.content, 1, 200) as preview
           FROM interactions i
           JOIN conversations c ON i.conversation_id = c.id
           WHERE i.role = 'assistant' AND i.content LIKE ?
           ORDER BY c.quality_score DESC LIMIT ?""",
        (f"%{q}%", limit)
    ).fetchall()
    db.close()
    return {"query": q, "results": [dict(r) for r in rows]}


@app.get("/api/export")
async def export_dataset(fmt: str = Query("chatml", regex="^(chatml|sharegpt|alpaca)$")):
    sys.path.insert(0, str(Path(__file__).parent.parent / "trainer"))
    from dataset_exporter import export_dataset
    
    tmpfile = f"/tmp/localdistill_export_{fmt}.jsonl"
    count = export_dataset(output_path=tmpfile, fmt=fmt, exclude_used=False)
    if count == 0:
        raise HTTPException(status_code=404, detail="No curated conversations")
    
    with open(tmpfile) as f:
        content = f.read()
    
    return PlainTextResponse(content=content, media_type="application/x-ndjson",
                             headers={"Content-Disposition": f"attachment; filename=train.{fmt}.jsonl"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)