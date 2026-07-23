"""
localdistill — Dataset Exporter.

Queries the curated_training table and exports conversations in
instruction-tuning formats ready for Unsloth / Axolotl / HuggingFace.

Supported formats:
  - chatml: ChatML format (used by Mistral, Qwen, etc.)
  - sharegpt: ShareGPT format (used by Llama 3, Vicuna)
  - alpaca: Alpaca format (instruction/input/output)

Usage:
  python dataset_exporter.py --format chatml --output train.jsonl
  python dataset_exporter.py --format sharegpt --output train.json --max-examples 100
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from db import get_db


def export_chatml(conversation_id: str, turns: List[Dict]) -> Dict:
    """
    ChatML format:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
    """
    messages = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role in ("user", "assistant", "system"):
            messages.append({"role": role, "content": content})
    
    # Ensure starts with user and alternates properly
    cleaned = []
    for msg in messages:
        if msg["role"] == "system":
            cleaned.append(msg)
        elif cleaned and msg["role"] == cleaned[-1]["role"]:
            # Merge consecutive same-role messages
            cleaned[-1]["content"] += "\n" + msg["content"]
        else:
            cleaned.append(msg)
    
    # Remove leading system if no user follows
    if cleaned and cleaned[0]["role"] == "system" and len(cleaned) > 1:
        pass  # Keep system+user pattern
    
    return {"messages": cleaned}


def export_sharegpt(conversation_id: str, turns: List[Dict]) -> Dict:
    """
    ShareGPT format:
    {"conversations": [{"from": "human", "value": "..."},
                        {"from": "gpt", "value": "..."}]}
    """
    conversations = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            conversations.append({"from": "human", "value": content})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": content})
        elif role == "system":
            conversations.append({"from": "system", "value": content})
    
    return {"conversations": conversations}


def export_alpaca(conversation_id: str, turns: List[Dict]) -> Dict:
    """
    Alpaca format (first user msg → instruction, assistant → output):
    {"instruction": "...", "input": "", "output": "..."}
    For multi-turn, concatenates with newlines.
    """
    instructions = []
    outputs = []
    current_role = None
    
    for turn in turns:
        if turn["role"] == "user":
            instructions.append(turn["content"])
        elif turn["role"] == "assistant":
            outputs.append(turn["content"])
    
    instruction = instructions[0] if instructions else ""
    # Subsequent user messages become "input"
    user_input = "\n".join(instructions[1:]) if len(instructions) > 1 else ""
    output = "\n".join(outputs)
    
    return {"instruction": instruction, "input": user_input, "output": output}


FORMATTERS = {
    "chatml": export_chatml,
    "sharegpt": export_sharegpt,
    "alpaca": export_alpaca,
}


def export_dataset(
    output_path: str,
    fmt: str = "chatml",
    max_examples: Optional[int] = None,
    min_score: float = 0.0,
    exclude_used: bool = True,
) -> int:
    """
    Export curated conversations to a training dataset file.
    
    Returns the number of examples exported.
    """
    db = get_db()
    
    query = """
        SELECT ct.conversation_id, ct.quality_score
        FROM curated_training ct
        WHERE ct.quality_score >= ?
    """
    params = [min_score]
    
    if exclude_used:
        query += " AND ct.used_in_training = 0"
    
    query += " ORDER BY ct.quality_score DESC"
    
    if max_examples:
        query += " LIMIT ?"
        params.append(max_examples)
    
    curated = db.execute(query, params).fetchall()
    
    if not curated:
        print(f"No curated conversations found (min_score={min_score}, exclude_used={exclude_used})")
        db.close()
        return 0
    
    formatter = FORMATTERS[fmt]
    examples = []
    
    for row in curated:
        cid = row["conversation_id"]
        interactions = db.execute(
            """SELECT role, content, turn_number 
               FROM interactions 
               WHERE conversation_id = ? 
               ORDER BY turn_number""",
            (cid,)
        ).fetchall()
        
        if not interactions:
            continue
        
        formatted = formatter(cid, [dict(i) for i in interactions])
        examples.append(formatted)
    
    db.close()
    
    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if fmt == "sharegpt" or output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
    else:
        # JSONL format
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"Exported {len(examples)} examples to {output_path}")
    print(f"  Format: {fmt}")
    print(f"  Example preview (first turn):")
    if examples:
        ex = examples[0]
        msgs = ex.get("messages", ex.get("conversations", []))
        if msgs:
            first = msgs[0]
            content = first.get("content", first.get("value", str(first)))
            print(f"    {first.get('role', first.get('from', '?'))}: {content[:100]}...")
    
    return len(examples)


def mark_as_used(conversation_ids: List[str], training_run_id: str):
    """Mark exported conversations as used in training."""
    db = get_db()
    for cid in conversation_ids:
        db.execute(
            "UPDATE curated_training SET used_in_training = 1, training_run_id = ? WHERE conversation_id = ?",
            (training_run_id, cid)
        )
    db.commit()
    db.close()
    print(f"Marked {len(conversation_ids)} conversations as used (run: {training_run_id[:8]}...)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="localdistill dataset exporter")
    parser.add_argument("--format", choices=["chatml", "sharegpt", "alpaca"],
                        default="chatml", help="Output format (default: chatml)")
    parser.add_argument("--output", default="train.jsonl", help="Output file path")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum examples to export")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum quality score (default: 0.0 = all)")
    parser.add_argument("--include-used", action="store_true",
                        help="Include already-used examples")
    
    args = parser.parse_args()
    
    count = export_dataset(
        output_path=args.output,
        fmt=args.format,
        max_examples=args.max_examples,
        min_score=args.min_score,
        exclude_used=not args.include_used,
    )
    
    if count == 0:
        sys.exit(1)