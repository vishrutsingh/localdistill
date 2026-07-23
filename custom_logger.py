"""
localdistill — LiteLLM Custom Callback for SQLite Logging.

Intercepts every API call through the LiteLLM proxy and logs:
  - Full request (user messages + system prompt)
  - Full response (assistant message + tool calls)
  - Tool call results (if any)
  - Token counts, latency, model info

Conversation tracking:
  - A new conversation is started when a request arrives without a 
    conversation_id header, or after a configurable idle timeout.
  - Conversation ID is persisted via x-conversation-id HTTP header.
  - Quality signals are detected on user messages and scored on completion.
"""

import sys
import os
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Ensure localdistill is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import (
    create_conversation, log_interaction, complete_conversation,
    set_quality_score, get_db,
)
from quality import analyze_message, score_conversation

import litellm
from litellm.integrations.custom_logger import CustomLogger


CONVERSATION_TIMEOUT_SECONDS = 300  # 5 min idle = new conversation
_conversation_state: Dict[str, Any] = {}  # litellm_call_id → conversation tracking


class LoggingCallback(CustomLogger):
    """Custom LiteLLM callback that logs to localdistill SQLite DB."""

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called on successful LLM completion."""
        self._log_event(kwargs, response_obj, start_time, end_time, success=True)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called on failed LLM completion."""
        self._log_event(kwargs, response_obj, start_time, end_time, success=False)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Sync fallback."""
        self._log_event(kwargs, response_obj, start_time, end_time, success=True)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Sync fallback."""
        self._log_event(kwargs, response_obj, start_time, end_time, success=False)

    def _log_event(self, kwargs, response_obj, start_time, end_time, success: bool):
        """Core logging logic — writes to SQLite."""
        try:
            litellm_call_id = kwargs.get("litellm_call_id", "unknown")
            messages = kwargs.get("messages", [])
            model = kwargs.get("model", "unknown")
            
            # ── Conversation tracking ──
            conv_id = self._get_or_create_conversation(kwargs, litellm_call_id)
            
            # ── Determine turn number ──
            state = _conversation_state.get(litellm_call_id, {})
            turn_number = state.get("turn", 0) + 1
            _conversation_state[litellm_call_id] = {
                "conversation_id": conv_id,
                "turn": turn_number,
                "last_activity": time.time(),
            }
            
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # ── Log USER message (the last user message in the messages array) ──
            user_message = self._extract_last_user_message(messages)
            if user_message:
                # Detect quality signals in the user message
                signals = analyze_message(user_message)
                for sig in signals:
                    self._log_signal(conv_id, litellm_call_id, sig)
                
                user_interaction_id = log_interaction(
                    conversation_id=conv_id,
                    turn_number=turn_number,
                    role="user",
                    content=user_message,
                    model=model,
                )
            
            # ── Log ASSISTANT response ──
            if success and response_obj:
                assistant_content, tool_calls = self._extract_response(response_obj)
                
                usage = response_obj.get("usage", {}) if isinstance(response_obj, dict) else getattr(response_obj, "usage", {})
                if isinstance(usage, dict):
                    tokens_prompt = usage.get("prompt_tokens")
                    tokens_completion = usage.get("completion_tokens")
                else:
                    tokens_prompt = getattr(usage, "prompt_tokens", None)
                    tokens_completion = getattr(usage, "completion_tokens", None)
                
                log_interaction(
                    conversation_id=conv_id,
                    turn_number=turn_number + 0.5,  # interleave between user and next user
                    role="assistant",
                    content=assistant_content or "",
                    tool_calls=tool_calls,
                    model=model,
                    provider=kwargs.get("custom_llm_provider", "unknown"),
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    latency_ms=latency_ms,
                )
            
            elif not success:
                # Log the error
                error_msg = str(response_obj) if response_obj else "Unknown error"
                log_interaction(
                    conversation_id=conv_id,
                    turn_number=turn_number + 0.5,
                    role="assistant",
                    content=f"[ERROR] {error_msg}",
                    model=model,
                )

        except Exception as e:
            # Never let logging failures break the proxy
            print(f"[localdistill] Logging error (non-fatal): {e}", file=sys.stderr)

    def _get_or_create_conversation(self, kwargs, litellm_call_id: str) -> str:
        """Get existing conversation or create a new one."""
        # Check for explicit conversation_id in request headers/metadata
        metadata = kwargs.get("metadata", {}) or {}
        conv_id = metadata.get("conversation_id")
        
        if not conv_id:
            # Check for x-conversation-id header
            headers = kwargs.get("headers", {}) or {}
            conv_id = headers.get("x-conversation-id")
        
        if not conv_id:
            # Check if we have an active conversation from recent calls
            state = _conversation_state.get(litellm_call_id, {})
            last_activity = state.get("last_activity", 0)
            if time.time() - last_activity < CONVERSATION_TIMEOUT_SECONDS:
                conv_id = state.get("conversation_id")
        
        if not conv_id:
            # Start a new conversation
            first_msg = self._extract_last_user_message(kwargs.get("messages", []))
            conv_id = create_conversation(first_message=first_msg)
        
        # Mark conversation active
        _conversation_state[litellm_call_id] = {
            "conversation_id": conv_id,
            "turn": _conversation_state.get(litellm_call_id, {}).get("turn", 0),
            "last_activity": time.time(),
        }
        
        return conv_id

    def _extract_last_user_message(self, messages: list) -> Optional[str]:
        """Extract the last user message from the messages array."""
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Vision format: [{type: "text", text: "..."}, {type: "image_url", ...}]
                        text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                        return " ".join(text_parts)
                    return content
            elif hasattr(msg, "role") and msg.role == "user":
                return getattr(msg, "content", "")
        return None

    def _extract_response(self, response_obj) -> tuple:
        """Extract content and tool calls from the response."""
        content = ""
        tool_calls = None
        
        if isinstance(response_obj, dict):
            choices = response_obj.get("choices", [])
        else:
            choices = getattr(response_obj, "choices", [])
        
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message", {})
            else:
                message = getattr(choice, "message", None)
            
            if message:
                if isinstance(message, dict):
                    content = message.get("content", "") or ""
                    tc = message.get("tool_calls")
                else:
                    content = getattr(message, "content", "") or ""
                    tc = getattr(message, "tool_calls", None)
                
                if tc:
                    tool_calls = []
                    for t in tc:
                        if isinstance(t, dict):
                            func = t.get("function", {})
                            tool_calls.append({
                                "id": t.get("id"),
                                "name": func.get("name"),
                                "arguments": func.get("arguments"),
                            })
                        else:
                            tool_calls.append({
                                "id": getattr(t, "id", None),
                                "name": getattr(t.function, "name", None) if hasattr(t, "function") else None,
                                "arguments": getattr(t.function, "arguments", None) if hasattr(t, "function") else None,
                            })
        
        return content, tool_calls

    def _log_signal(self, conv_id: str, interaction_id: str, signal: dict):
        """Log a detected quality signal."""
        try:
            db = get_db()
            db.execute(
                """INSERT INTO conversation_signals 
                   (conversation_id, interaction_id, signal_type, signal_subtype, 
                    matched_text, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    conv_id,
                    interaction_id or "",
                    signal["type"],
                    signal.get("description", ""),
                    signal.get("matched_text", ""),
                    signal.get("confidence", 1.0),
                )
            )
            db.commit()
            db.close()
        except Exception as e:
            print(f"[localdistill] Signal logging error: {e}", file=sys.stderr)


# ── Completion hook: score conversation when it ends ──

def _finalize_conversation(conversation_id: str):
    """Called when a conversation is detected as complete.
    Computes final quality score and decides auto-promotion."""
    try:
        db = get_db()
        
        # Get all signals for this conversation
        signals = db.execute(
            "SELECT signal_type, matched_text, confidence FROM conversation_signals WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchall()
        
        # Get turn count and whether tools were used
        conv = db.execute(
            "SELECT turn_count FROM conversations WHERE id = ?",
            (conversation_id,)
        ).fetchone()
        
        if not conv:
            db.close()
            return
        
        turn_count = conv["turn_count"]
        
        # Check for tool calls
        tool_rows = db.execute(
            "SELECT COUNT(*) as cnt FROM interactions WHERE conversation_id = ? AND role = 'assistant' AND tool_calls IS NOT NULL",
            (conversation_id,)
        ).fetchone()
        has_tool_calls = tool_rows["cnt"] > 0 if tool_rows else False
        
        db.close()
        
        # Score
        signal_list = [dict(s) for s in signals]
        score, breakdown = score_conversation(
            signals=signal_list,
            turn_count=turn_count,
            has_tool_calls=has_tool_calls,
            was_abandoned=False,
        )
        
        # Save score
        set_quality_score(conversation_id, score)
        
        # Mark completed
        complete_conversation(conversation_id, "completed")
        
        print(f"[localdistill] Conversation {conversation_id[:8]}... scored {score} "
              f"({len(signal_list)} signals, {turn_count} turns)", file=sys.stderr)
        
    except Exception as e:
        print(f"[localdistill] Finalization error: {e}", file=sys.stderr)


# LiteLLM expects the callback instance at module level
custom_logger = LoggingCallback()


# Also export a utility to finalize conversations (called by MCP server or watchdog)
def finalize_idle_conversations():
    """Check for idle conversations and finalize them."""
    now = time.time()
    to_finalize = []
    
    for call_id, state in list(_conversation_state.items()):
        last_activity = state.get("last_activity", 0)
        conv_id = state.get("conversation_id")
        if now - last_activity > CONVERSATION_TIMEOUT_SECONDS and conv_id:
            to_finalize.append(conv_id)
            del _conversation_state[call_id]
    
    for conv_id in to_finalize:
        _finalize_conversation(conv_id)
    
    return len(to_finalize)