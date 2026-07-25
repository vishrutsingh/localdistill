#!/usr/bin/env python3
"""Curate ShareGPT52K → clean ChatML JSONL for distillation training."""
import json, re, sys
from pathlib import Path

SRC = Path.home() / ".cache/huggingface/hub/datasets--RyokoAI--ShareGPT52K/snapshots/6f9b78cc1dd15dbb51d3c51ccc219c558962fd77/old/sg_52k.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "localdistill/curated_train.jsonl"

HTML_RE = re.compile(r"<[^>]+>")
CODE_FENCE_RE = re.compile(r"```")


def strip_html(text: str) -> str:
    return HTML_RE.sub("", text)


def is_code_heavy(text: str) -> bool:
    """True if the response is >80% code blocks (by line count)."""
    lines = text.split("\n")
    inside = False
    code_lines = 0
    for line in lines:
        if line.strip().startswith("```"):
            inside = not inside
            continue
        if inside:
            code_lines += 1
    total = len([l for l in lines if l.strip()])
    return total > 0 and code_lines / total > 0.8


def is_english(text: str) -> bool:
    """Rough: >90% ASCII chars."""
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > 0.9 if text else True


def curate():
    with open(SRC) as f:
        raw = json.load(f)

    kept = 0
    skipped_html = 0
    skipped_single = 0
    skipped_lang = 0
    skipped_code = 0

    with open(OUT, "w") as out:
        for item in raw:
            convs = item.get("conversations", [])
            if len(convs) < 2:
                skipped_single += 1
                continue

            messages = []
            drop = False
            for turn in convs:
                role = "user" if turn["from"] == "human" else "assistant"
                content = turn["value"]

                if role == "assistant":
                    content = strip_html(content)
                    if not is_english(content):
                        drop = True
                        skipped_lang += 1
                        break
                    if is_code_heavy(content):
                        drop = True
                        skipped_code += 1
                        break

                messages.append({"role": role, "content": content})

            if drop or len(messages) < 2:
                continue

            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept:    {kept}")
    print(f"Dropped: single-turn={skipped_single}  non-english={skipped_lang}  code-only={skipped_code}")
    print(f"→ {OUT}")


def insert_into_db(jsonl_path=None):
    """Insert curated JSONL into localdistill DB. One-shot, no dep."""
    import sqlite3, uuid
    from pathlib import Path
    db_path = Path.home() / "localdistill/data/localdistill.db"
    src = Path(jsonl_path) if jsonl_path else OUT
    db = sqlite3.connect(str(db_path))
    db.execute("DELETE FROM curated_training WHERE promoted_by='auto'")
    db.execute("DELETE FROM interactions WHERE conversation_id IN (SELECT id FROM conversations WHERE id GLOB '*-*-*-*-*')")
    db.execute("DELETE FROM conversations WHERE id GLOB '*-*-*-*-*'")
    MAX_INSERT = 5000  # ponytail: 42K inserts is slow, 5K is plenty for training
    ci = ii = 0
    conv_rows = []
    int_rows = []
    cur_rows = []
    with open(src) as f:
        for i, line in enumerate(f):
            if i >= MAX_INSERT:
                break
            ex = json.loads(line)
            msgs = ex["messages"]
            cid = str(uuid.uuid4())
            title = msgs[0]["content"][:80].replace("\n", " ")
            conv_rows.append((cid, title, "completed", len(msgs)))
            for tn, m in enumerate(msgs, 1):
                int_rows.append((str(uuid.uuid4()), cid, tn, m["role"], m["content"]))
                ii += 1
            cur_rows.append((cid, "auto", 0.85, "chatml"))
            ci += 1
    db.executemany("INSERT INTO conversations (id,title,status,turn_count,created_at) VALUES (?,?,?,?,datetime('now'))", conv_rows)
    db.executemany("INSERT INTO interactions (id,conversation_id,turn_number,role,content,created_at) VALUES (?,?,?,?,?,datetime('now'))", int_rows)
    db.executemany("INSERT INTO curated_training (conversation_id,promoted_by,quality_score,format) VALUES (?,?,?,?)", cur_rows)
    db.commit(); db.close()
    print(f"DB: {ci} conversations, {ii} interactions → curated_training")


if __name__ == "__main__":
    curate()
    insert_into_db()
