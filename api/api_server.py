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

app = FastAPI(title="localdistill-api")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"app": "localdistill", "endpoints": {"health": "/health", "status": "/api/status", "conversations": "/api/conversations", "export": "/api/export?format=chatml"}}

DATA_DIR = Path("/data")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    db = get_db()
    curated_new = db.execute("SELECT COUNT(*) as cnt FROM curated_training WHERE used_in_training = 0").fetchone()
    curated_total = db.execute("SELECT COUNT(*) as cnt FROM curated_training").fetchone()
    completed = db.execute("SELECT COUNT(*) as cnt FROM conversations WHERE status = 'completed'").fetchone()
    active = db.execute("SELECT COUNT(*) as cnt FROM conversations WHERE status = 'active'").fetchone()
    
    last_run = db.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    db.close()
    
    return {
        "curated_new": curated_new["cnt"],
        "curated_total": curated_total["cnt"],
        "completed_conversations": completed["cnt"],
        "active_conversations": active["cnt"],
        "last_training_run": {
            "id": last_run["id"][:8],
            "status": last_run["status"],
            "examples": last_run["num_examples"],
            "base_model": last_run["base_model"],
            "completed_at": last_run["completed_at"],
        } if last_run else None,
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