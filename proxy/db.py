"""
localdistill — SQLite database schema and queries.

Tables:
  interactions      — raw API request/response logs
  conversation_signals — detected quality signals per turn
  curated_training  — promoted interactions for fine-tuning
  rag_embeddings    — vector store for RAG retrieval
  training_runs     — history of LoRA training runs
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("~/localdistill/localdistill.db").expanduser()


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db():
    """Create all tables and indexes. Idempotent."""
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS interactions (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            turn_number     INTEGER NOT NULL DEFAULT 1,
            role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
            content         TEXT NOT NULL,
            tool_calls      TEXT,       -- JSON array of tool calls (assistant only)
            tool_call_id    TEXT,       -- tool call ID (tool role only)
            tool_name       TEXT,       -- tool name (tool role only)
            model           TEXT,       -- model used for this response
            provider        TEXT,       -- API provider
            tokens_prompt   INTEGER,
            tokens_completion INTEGER,
            latency_ms      INTEGER,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id              TEXT PRIMARY KEY,
            title           TEXT,       -- auto-generated from first user message
            status          TEXT NOT NULL DEFAULT 'active'
                            CHECK(status IN ('active', 'completed', 'abandoned')),
            turn_count      INTEGER NOT NULL DEFAULT 0,
            quality_score   REAL,       -- overall score, computed
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS conversation_signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            interaction_id  TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            signal_subtype  TEXT,
            matched_text    TEXT,       -- the text that triggered the signal
            confidence      REAL NOT NULL DEFAULT 1.0,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            
            FOREIGN KEY (conversation_id) REFERENCES conversations(id),
            FOREIGN KEY (interaction_id) REFERENCES interactions(id)
        );

        CREATE TABLE IF NOT EXISTS curated_training (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL UNIQUE,
            promoted_by     TEXT NOT NULL DEFAULT 'auto'
                            CHECK(promoted_by IN ('auto', 'manual')),
            quality_score   REAL NOT NULL,
            format           TEXT NOT NULL DEFAULT 'chatml'
                            CHECK(format IN ('chatml', 'sharegpt', 'alpaca')),
            used_in_training BOOLEAN NOT NULL DEFAULT 0,
            training_run_id TEXT,
            promoted_at     TEXT NOT NULL DEFAULT (datetime('now')),
            
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE TABLE IF NOT EXISTS rag_embeddings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            chunk_text      TEXT NOT NULL,
            chunk_type      TEXT NOT NULL DEFAULT 'turn'
                            CHECK(chunk_type IN ('turn', 'summary', 'document')),
            metadata        TEXT,       -- JSON: topic, tags, etc.
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
        -- Vector embeddings stored via sqlite-vec extension
        -- CREATE VIRTUAL TABLE rag_vectors USING vec0(embedding float[768]);

        CREATE TABLE IF NOT EXISTS training_runs (
            id              TEXT PRIMARY KEY,
            base_model      TEXT NOT NULL,
            adapter_path    TEXT,
            num_examples    INTEGER NOT NULL DEFAULT 0,
            lora_rank       INTEGER NOT NULL DEFAULT 16,
            lora_alpha      INTEGER NOT NULL DEFAULT 32,
            learning_rate   REAL NOT NULL DEFAULT 2e-4,
            num_epochs      INTEGER NOT NULL DEFAULT 3,
            eval_score_before REAL,
            eval_score_after  REAL,
            eval_benchmark  TEXT,
            status          TEXT NOT NULL DEFAULT 'pending'
                            CHECK(status IN ('pending', 'running', 'completed', 'failed', 'rolled_back')),
            error_log       TEXT,
            started_at      TEXT,
            completed_at    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_interactions_conv 
            ON interactions(conversation_id, turn_number);
        CREATE INDEX IF NOT EXISTS idx_interactions_created 
            ON interactions(created_at);
        CREATE INDEX IF NOT EXISTS idx_conversations_status 
            ON conversations(status, quality_score);
        CREATE INDEX IF NOT EXISTS idx_conversations_created 
            ON conversations(created_at);
        CREATE INDEX IF NOT EXISTS idx_signals_conv 
            ON conversation_signals(conversation_id, signal_type);
        CREATE INDEX IF NOT EXISTS idx_curated_used 
            ON curated_training(used_in_training, promoted_at);
        CREATE INDEX IF NOT EXISTS idx_rag_conv 
            ON rag_embeddings(conversation_id);
    """)
    db.commit()
    db.close()


# ── Query helpers ────────────────────────────────────────────

def create_conversation(conversation_id: str = None, first_message: str = None) -> str:
    """Create a new conversation. Returns the conversation_id."""
    conv_id = conversation_id or str(uuid.uuid4())
    title = None
    if first_message:
        title = first_message[:100].replace("\n", " ")
    db = get_db()
    db.execute(
        "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
        (conv_id, title, datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    db.close()
    return conv_id


def log_interaction(
    conversation_id: str,
    turn_number: int,
    role: str,
    content: str,
    tool_calls: list = None,
    tool_call_id: str = None,
    tool_name: str = None,
    model: str = None,
    provider: str = None,
    tokens_prompt: int = None,
    tokens_completion: int = None,
    latency_ms: int = None,
) -> str:
    """Log a single interaction turn. Returns the interaction_id."""
    interaction_id = str(uuid.uuid4())
    db = get_db()
    db.execute(
        """INSERT INTO interactions 
           (id, conversation_id, turn_number, role, content, tool_calls, 
            tool_call_id, tool_name, model, provider, 
            tokens_prompt, tokens_completion, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            interaction_id, conversation_id, turn_number, role, content,
            json.dumps(tool_calls) if tool_calls else None,
            tool_call_id, tool_name, model, provider,
            tokens_prompt, tokens_completion, latency_ms,
        )
    )
    db.execute(
        "UPDATE conversations SET turn_count = ? WHERE id = ?",
        (turn_number, conversation_id)
    )
    db.commit()
    db.close()
    return interaction_id


def complete_conversation(conversation_id: str, status: str = "completed"):
    """Mark a conversation as completed or abandoned."""
    db = get_db()
    db.execute(
        "UPDATE conversations SET status = ?, completed_at = ? WHERE id = ?",
        (status, datetime.now(timezone.utc).isoformat(), conversation_id)
    )
    db.commit()
    db.close()


def set_quality_score(conversation_id: str, score: float):
    """Set the overall quality score for a conversation."""
    db = get_db()
    db.execute(
        "UPDATE conversations SET quality_score = ? WHERE id = ?",
        (score, conversation_id)
    )
    db.commit()
    db.close()


def get_unreviewed_conversations(limit: int = 50):
    """Get conversations that need curation review."""
    db = get_db()
    rows = db.execute(
        """SELECT c.* FROM conversations c
           LEFT JOIN curated_training ct ON c.id = ct.conversation_id
           WHERE c.status = 'completed'
             AND ct.conversation_id IS NULL
           ORDER BY c.quality_score DESC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str):
    """Get full conversation with all turns and signals."""
    db = get_db()
    conv = db.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    if not conv:
        db.close()
        return None
    
    turns = db.execute(
        "SELECT * FROM interactions WHERE conversation_id = ? ORDER BY turn_number",
        (conversation_id,)
    ).fetchall()
    
    signals = db.execute(
        "SELECT * FROM conversation_signals WHERE conversation_id = ? ORDER BY created_at",
        (conversation_id,)
    ).fetchall()
    
    db.close()
    return {
        "conversation": dict(conv),
        "turns": [dict(t) for t in turns],
        "signals": [dict(s) for s in signals],
    }