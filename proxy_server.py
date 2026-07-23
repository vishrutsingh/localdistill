"""
localdistill — Standalone Proxy Server.

A lightweight OpenAI-compatible proxy that:
  1. Accepts /v1/chat/completions requests
  2. Routes them to the real API via litellm
  3. Logs every request/response to SQLite
  4. Detects quality signals in user messages
  5. Scores conversations on completion

Start with:  python proxy_server.py --port 8787
"""

import os
import sys
import json
import time
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

# Add localdistill to path
sys.path.insert(0, str(Path(__file__).parent))

import litellm
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from db import (
    init_db, create_conversation, log_interaction,
    complete_conversation, set_quality_score,
)
from quality import analyze_message, score_conversation, score_conversation_from_db

app = FastAPI(title="localdistill-proxy")
CONVERSATION_TIMEOUT = 300  # 5 min idle = new conversation

# In-memory conversation state
_conversations: dict = {}  # conversation_id → {"last_activity": timestamp, "turn": int}


@app.on_event("startup")
async def startup():
    init_db()
    print("[localdistill] Database initialized")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    body = await request.json()
    
    messages = body.get("messages", [])
    model = body.get("model", "gpt-4o-mini")
    conv_id = request.headers.get("x-conversation-id")
    
    # ── Conversation tracking ──
    conv_id = _resolve_conversation(conv_id, messages)
    
    # ── Log user message ──
    user_msg = _extract_last_user_message(messages)
    turn = _conversations[conv_id]["turn"] + 1
    _conversations[conv_id]["turn"] = turn
    _conversations[conv_id]["last_activity"] = time.time()
    
    user_interaction_id = None
    if user_msg:
        # Detect quality signals
        signals = analyze_message(user_msg)
        
        user_interaction_id = log_interaction(
            conversation_id=conv_id,
            turn_number=turn,
            role="user",
            content=user_msg,
            model=model,
        )
        
        # Store signals in DB
        from db import get_db
        db = get_db()
        for sig in signals:
            db.execute(
                """INSERT INTO conversation_signals 
                   (conversation_id, interaction_id, signal_type, matched_text, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (conv_id, user_interaction_id, sig["type"], sig["matched_text"], sig["confidence"])
            )
        db.commit()
        db.close()
    
    # ── Route to real API ──
    start_time = time.time()
    try:
        # Use litellm for multi-provider routing
        litellm_params = {
            "model": model,
            "messages": messages,
        }
        
        # Forward optional params
        for key in ("temperature", "max_tokens", "top_p", "stream", "tools", "tool_choice"):
            if key in body:
                litellm_params[key] = body[key]
        
        response = await litellm.acompletion(**litellm_params)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # ── Extract response ──
        choice = response.choices[0]
        assistant_content = choice.message.content or ""
        
        # Extract tool calls if any
        tool_calls = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in choice.message.tool_calls
            ]
        
        # ── Log assistant response ──
        usage = response.usage
        log_interaction(
            conversation_id=conv_id,
            turn_number=turn + 0.5,
            role="assistant",
            content=assistant_content,
            tool_calls=tool_calls,
            model=model,
            provider=model.split("/")[0] if "/" in model else "unknown",
            tokens_prompt=usage.prompt_tokens if usage else None,
            tokens_completion=usage.completion_tokens if usage else None,
            latency_ms=latency_ms,
        )
        
        # ── Return OpenAI-formatted response ──
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}
                        }
                        for tc in (tool_calls or [])
                    ] if tool_calls else None,
                },
                "finish_reason": choice.finish_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": (usage.total_tokens if usage else 0),
            }
        }
    
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        error_msg = str(e)
        
        # Log error
        log_interaction(
            conversation_id=conv_id,
            turn_number=turn + 0.5,
            role="assistant",
            content=f"[ERROR] {error_msg}",
            model=model,
            latency_ms=latency_ms,
        )
        
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/v1/conversations/{conv_id}/finalize")
async def finalize_conversation(conv_id: str):
    """Explicitly finalize a conversation — computes quality score."""
    score = _finalize_conv(conv_id)
    return {"conversation_id": conv_id, "quality_score": score}


@app.get("/health")
async def health():
    return {"status": "ok", "active_conversations": len(_conversations)}


# ── Helpers ──

def _resolve_conversation(conv_id: Optional[str], messages: list) -> str:
    """Get or create a conversation."""
    global _conversations
    
    # If explicitly provided and exists, use it
    if conv_id and conv_id in _conversations:
        _conversations[conv_id]["last_activity"] = time.time()
        return conv_id
    
    # Check for idle timeout on existing conversations
    now = time.time()
    for cid, state in list(_conversations.items()):
        if now - state["last_activity"] > CONVERSATION_TIMEOUT:
            _finalize_conv(cid)
            del _conversations[cid]
    
    # Create new conversation
    first_msg = _extract_last_user_message(messages)
    conv_id = create_conversation(first_message=first_msg)
    _conversations[conv_id] = {"last_activity": now, "turn": 0}
    
    return conv_id


def _extract_last_user_message(messages: list) -> Optional[str]:
    """Extract last user message from the messages array."""
    for msg in reversed(messages):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role == "user":
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(text_parts)
            return content
    return None


def _finalize_conv(conv_id: str) -> float:
    """Compute final quality score for a conversation."""
    from db import get_db
    try:
        db = get_db()
        
        # Get all signals
        signals = db.execute(
            "SELECT signal_type, matched_text, confidence FROM conversation_signals WHERE conversation_id = ?",
            (conv_id,)
        ).fetchall()
        
        # Get turn count
        conv = db.execute(
            "SELECT turn_count FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        turn_count = conv["turn_count"] if conv else 0
        
        # Check for tool calls
        tool_rows = db.execute(
            "SELECT COUNT(*) as cnt FROM interactions WHERE conversation_id = ? AND role = 'assistant' AND tool_calls IS NOT NULL",
            (conv_id,)
        ).fetchone()
        has_tools = tool_rows["cnt"] > 0 if tool_rows else False
        
        db.close()
        
        signal_list = [dict(s) for s in signals]
        score, breakdown = score_conversation_from_db(
            signals=signal_list,
            turn_count=turn_count,
            has_tool_calls=has_tools,
        )
        
        set_quality_score(conv_id, score)
        complete_conversation(conv_id, "completed")
        
        print(f"[localdistill] Conv {conv_id[:8]}... scored {score} "
              f"({len(signal_list)} signals, {turn_count} turns): {breakdown}")
        
        return score
    except Exception as e:
        print(f"[localdistill] Finalization error: {e}", file=sys.stderr)
        return 0.0


# ── Background task: auto-finalize idle conversations ──

async def auto_finalize_loop():
    """Periodically finalize idle conversations."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        now = time.time()
        to_finalize = []
        for cid, state in list(_conversations.items()):
            if now - state["last_activity"] > CONVERSATION_TIMEOUT:
                to_finalize.append(cid)
        
        for cid in to_finalize:
            _finalize_conv(cid)
            del _conversations[cid]


@app.on_event("startup")
async def startup_with_cleanup():
    init_db()
    asyncio.create_task(auto_finalize_loop())
    print("[localdistill] Database initialized, auto-finalizer started")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="localdistill proxy server")
    parser.add_argument("--port", type=int, default=8787, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)