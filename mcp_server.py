"""
localdistill — MCP Curation Server.

Provides tools for inline quality curation during conversations:
  - signal_quality: mark a conversation as good/bad
  - get_last_conversation: check recent conversation score
  - search_knowledge: RAG search (placeholder for Phase 3)
  - get_training_status: training pipeline overview
  - add_to_rag: manually add a conversation to RAG index

Start with:  python mcp_server.py
Configure in your MCP client (Hermes, Claude Desktop, etc.).

Hermes example:
  hermes mcp add localdistill --command "python /root/localdistill/mcp_server.py"
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from db import get_db, get_conversation, get_unreviewed_conversations
from quality import should_auto_promote, should_auto_exclude, AUTO_PROMOTE_THRESHOLD

mcp = FastMCP("localdistill-curator")


# ═══════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
def signal_quality(conversation_id: str = "", rating: str = "good") -> str:
    """
    Mark a conversation's quality rating.
    
    Use this when you disagree with the auto-scored quality of a conversation.
    
    Args:
        conversation_id: The conversation ID (use 'last' for most recent, or leave empty for 'last')
        rating: One of 'good' (promote to training), 'bad' (exclude), or 'neutral' (let auto-scoring decide)
    
    Returns a summary of the action taken.
    """
    db = get_db()
    
    # Resolve conversation_id
    cid = conversation_id.strip()
    if not cid or cid.lower() == "last":
        row = db.execute(
            "SELECT id, title, quality_score FROM conversations ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            db.close()
            return "No conversations found in the database."
        cid = row["id"]
        current_score = row["quality_score"]
        title = row["title"]
    else:
        row = db.execute(
            "SELECT id, title, quality_score FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
        if not row:
            db.close()
            return f"Conversation {cid} not found."
        current_score = row["quality_score"]
        title = row["title"]
    
    # Apply rating
    rating = rating.lower().strip()
    
    if rating in ("good", "promote"):
        new_score = max(current_score or 0.5, AUTO_PROMOTE_THRESHOLD + 0.01)
        db.execute(
            "UPDATE conversations SET quality_score = ? WHERE id = ?", (new_score, cid)
        )
        # Also promote to curated_training
        db.execute(
            """INSERT OR REPLACE INTO curated_training 
               (conversation_id, promoted_by, quality_score, promoted_at)
               VALUES (?, 'manual', ?, ?)""",
            (cid, new_score, datetime.now(timezone.utc).isoformat())
        )
        db.commit()
        db.close()
        return (
            f"✓ Conversation '{title or cid[:12]}...' promoted to training.\n"
            f"  Score set to {new_score} (manual override).\n"
            f"  This conversation will be included in the next training run."
        )
    
    elif rating in ("bad", "exclude", "demote"):
        new_score = 0.10
        db.execute(
            "UPDATE conversations SET quality_score = ? WHERE id = ?", (new_score, cid)
        )
        # Remove from curated if it was there
        db.execute(
            "DELETE FROM curated_training WHERE conversation_id = ?", (cid,)
        )
        db.commit()
        db.close()
        return (
            f"✗ Conversation '{title or cid[:12]}...' excluded from training.\n"
            f"  Score set to {new_score} (manual override).\n"
            f"  This conversation will NOT be used for training."
        )
    
    elif rating in ("neutral", "reset"):
        db.execute(
            "UPDATE conversations SET quality_score = NULL WHERE id = ?", (cid,)
        )
        db.execute(
            "DELETE FROM curated_training WHERE conversation_id = ?", (cid,)
        )
        db.commit()
        db.close()
        return (
            f"Conversation '{title or cid[:12]}...' reset to neutral.\n"
            f"  Auto-scoring will re-evaluate on next finalization."
        )
    
    else:
        db.close()
        return f"Unknown rating '{rating}'. Use 'good', 'bad', or 'neutral'."


@mcp.tool()
def get_last_conversation() -> str:
    """
    Show the most recent conversation's quality score and details.
    Call this after a conversation ends to see if it was auto-promoted.
    
    Returns a formatted summary including quality score, signals detected, and auto-promotion status.
    """
    db = get_db()
    
    # Get most recent conversation
    conv_row = db.execute(
        "SELECT * FROM conversations ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    
    if not conv_row:
        db.close()
        return "No conversations in the database yet."
    
    cid = conv_row["id"]
    
    # Get signals
    signals = db.execute(
        "SELECT signal_type, matched_text FROM conversation_signals WHERE conversation_id = ?",
        (cid,)
    ).fetchall()
    
    # Get turn count and tool usage
    tool_count = db.execute(
        "SELECT COUNT(*) as cnt FROM interactions WHERE conversation_id = ? AND tool_calls IS NOT NULL",
        (cid,)
    ).fetchone()
    
    # Check curation status
    curated = db.execute(
        "SELECT * FROM curated_training WHERE conversation_id = ?", (cid,)
    ).fetchone()
    
    db.close()
    
    score = conv_row["quality_score"]
    title = conv_row["title"] or "(untitled)"
    turns = conv_row["turn_count"]
    status = conv_row["status"]
    
    # Build summary
    lines = [
        f"📋 Conversation: {title}",
        f"   ID: {cid[:12]}...",
        f"   Turns: {turns}  |  Tool calls: {tool_count['cnt'] if tool_count else 0}",
        f"   Status: {status}",
        f"   Quality score: {score if score is not None else 'not yet scored'}",
    ]
    
    if signals:
        signal_types = {}
        for s in signals:
            st = s["signal_type"]
            signal_types[st] = signal_types.get(st, 0) + 1
        sig_summary = ", ".join(f"{k} (x{v})" for k, v in signal_types.items())
        lines.append(f"   Signals: {sig_summary}")
    
    if curated:
        lines.append(f"   ✓ In training set (promoted by: {curated['promoted_by']})")
    elif score and should_auto_promote(score):
        lines.append(f"   ✓ Auto-promoted to training (score >= {AUTO_PROMOTE_THRESHOLD})")
    elif score and should_auto_exclude(score):
        lines.append(f"   ✗ Auto-excluded from training (score too low)")
    elif score is not None:
        lines.append(f"   ⚠ Needs review (score in gray zone)")
    
    lines.append(f"\n   Commands: /good | /bad | /neutral")
    
    return "\n".join(lines)


@mcp.tool()
def search_knowledge(query: str, limit: int = 5) -> str:
    """
    Search the RAG knowledge base.
    
    Currently uses basic text matching. Full vector search coming in Phase 3.
    
    Args:
        query: The search query
        limit: Maximum number of results (default 5)
    
    Returns matching conversation excerpts.
    """
    db = get_db()
    
    # Basic text search (will be replaced with vector search in Phase 3)
    rows = db.execute(
        """SELECT DISTINCT c.id, c.title, c.quality_score, 
                  substr(i.content, 1, 200) as preview
           FROM interactions i
           JOIN conversations c ON i.conversation_id = c.id
           WHERE i.role = 'assistant' 
             AND i.content LIKE ?
           ORDER BY c.quality_score DESC
           LIMIT ?""",
        (f"%{query}%", limit)
    ).fetchall()
    
    db.close()
    
    if not rows:
        return f"No knowledge found for '{query}'."
    
    lines = [f"🔍 Knowledge results for '{query}':"]
    for i, row in enumerate(rows, 1):
        score_str = f"⭐{row['quality_score']}" if row['quality_score'] else "unscored"
        lines.append(f"\n  {i}. {row['title'] or '(untitled)'} [{score_str}]")
        lines.append(f"     {row['preview'][:150]}...")
    
    return "\n".join(lines)


@mcp.tool()
def get_training_status() -> str:
    """
    Show the current state of the training pipeline.
    
    Returns: training runs history, curated dataset size, next scheduled run.
    """
    db = get_db()
    
    # Training runs
    runs = db.execute(
        "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    
    # Curated dataset
    curated_count = db.execute(
        "SELECT COUNT(*) as cnt FROM curated_training WHERE used_in_training = 0"
    ).fetchone()
    
    total_curated = db.execute(
        "SELECT COUNT(*) as cnt FROM curated_training"
    ).fetchone()
    
    # Total conversations
    total_convs = db.execute(
        "SELECT COUNT(*) as cnt FROM conversations WHERE status = 'completed'"
    ).fetchone()
    
    db.close()
    
    lines = [
        "📊 LOCALDISTILL TRAINING STATUS",
        f"   Curated for training: {curated_count['cnt']} new / {total_curated['cnt']} total",
        f"   Completed conversations: {total_convs['cnt']}",
    ]
    
    if runs:
        lines.append(f"\n   Recent training runs:")
        for run in runs:
            status_icon = {"completed": "✓", "failed": "✗", "running": "⏳", "pending": "○"}.get(run["status"], "?")
            lines.append(
                f"   {status_icon} {run['id'][:8]}... | {run['base_model']} | "
                f"{run['num_examples']} examples | score: {run.get('eval_score_before', '?')}→{run.get('eval_score_after', '?')} | "
                f"{run['status']}"
            )
    else:
        lines.append(f"\n   No training runs yet. Phase 3 not started.")
    
    lines.append(f"\n   Phase 1 (capture): ✓  |  Phase 2 (curation): ✓  |  Phase 3 (training): pending")
    
    return "\n".join(lines)


@mcp.tool()
def add_to_rag(conversation_id: str = "", chunk_type: str = "turn") -> str:
    """
    Manually add a conversation to the RAG index.
    
    Args:
        conversation_id: Conversation ID (use 'last' for most recent)
        chunk_type: 'turn' (individual turns) or 'summary' (whole conversation)
    
    Returns confirmation of what was indexed.
    """
    db = get_db()
    
    # Resolve ID
    cid = conversation_id.strip()
    if not cid or cid.lower() == "last":
        row = db.execute(
            "SELECT id FROM conversations ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            db.close()
            return "No conversations found."
        cid = row["id"]
    
    # Get interactions
    interactions = db.execute(
        "SELECT * FROM interactions WHERE conversation_id = ? ORDER BY turn_number",
        (cid,)
    ).fetchall()
    
    if not interactions:
        db.close()
        return f"No interactions found for conversation {cid[:12]}..."
    
    added = 0
    if chunk_type == "turn":
        for inter in interactions:
            if inter["role"] in ("user", "assistant"):
                db.execute(
                    """INSERT INTO rag_embeddings (conversation_id, chunk_text, chunk_type, metadata)
                       VALUES (?, ?, ?, ?)""",
                    (cid, inter["content"][:2000], "turn",
                     json.dumps({"role": inter["role"], "turn": inter["turn_number"]}))
                )
                added += 1
    else:
        # Full conversation as one chunk
        full_text = "\n\n".join(
            f"[{i['role']}] {i['content'][:500]}" for i in interactions
        )
        db.execute(
            """INSERT INTO rag_embeddings (conversation_id, chunk_text, chunk_type, metadata)
               VALUES (?, ?, ?, ?)""",
            (cid, full_text[:4000], "summary",
             json.dumps({"turns": len(interactions)}))
        )
        added = 1
    
    db.commit()
    db.close()
    
    conv_title = interactions[0]["content"][:80] if interactions else "unknown"
    return f"✓ Added {added} chunks from '{conv_title}...' to RAG index."


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[localdistill] MCP curation server starting...")
    print("[localdistill] Tools: signal_quality, get_last_conversation, search_knowledge, get_training_status, add_to_rag")
    mcp.run()