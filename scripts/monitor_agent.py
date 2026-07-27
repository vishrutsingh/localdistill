#!/usr/bin/env python3
"""
LocalDistill Training Monitor Agent

Polls an active training run and auto-reports results to Second Brain (Obsidian vault).
Run in background: nohup python3 scripts/monitor_agent.py &

Usage:
    python3 scripts/monitor_agent.py [--run-dir DIR] [--vault mind] [--poll 60]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

VAULT = os.environ.get("SECOND_BRAIN_VAULT", "mind")
AGENT_LOG = Path("logs/agent_watch.log")
NOTE_TITLE = "agent-training-watch"


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENT_LOG, "a") as f:
        f.write(line + "\n")


def shell(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return 1, str(e)


def obsidian_cmd(subcmd: str, *args) -> str:
    """Run obsidian CLI, return stdout or empty string on error."""
    cmd = ["obsidian", f"vault={VAULT}", subcmd] + list(args)
    rc, out = shell(cmd)
    if rc != 0:
        log(f"obsidian error ({rc}): {out[:200]}")
        return ""
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

def find_active_run():
    """Find the most recent run dir with status.json that hasn't completed."""
    runs_dir = Path("logs/runs")
    if not runs_dir.exists():
        return None

    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for rd in run_dirs:
        status_file = rd / "status.json"
        if not status_file.exists():
            continue
        try:
            with open(status_file) as f:
                status = json.load(f)
            # Not completed and not errored = active
            if not status.get("completed_at") and not status.get("error"):
                return rd, status
        except Exception:
            continue
    return None


def get_gpu_info():
    """Get concise GPU status."""
    rc, out = shell(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"])
    if rc != 0:
        return "no-gpu"
    parts = [p.strip() for p in out.split(",")]
    if len(parts) >= 4:
        return f"T:{parts[0]}°C U:{parts[1]}% M:{parts[2]}MiB P:{parts[3]}W"
    return out[:60]


def check_training_process(pid: int) -> bool:
    rc, _ = shell(["kill", "-0", str(pid)])
    return rc == 0


def poll_status(run_dir: Path):
    status_file = run_dir / "status.json"
    if not status_file.exists():
        return None
    with open(status_file) as f:
        return json.load(f)


def tail_log(run_dir: Path, n: int = 5) -> str:
    for name in ("run.log", "train.log"):
        logf = run_dir / name
        if logf.exists():
            rc, out = shell(["tail", "-n", str(n), str(logf)])
            if rc == 0:
                return out
    return "(no log)"


# ═══════════════════════════════════════════════════════════════════════════════
# Second Brain Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def write_inbox_note(content: str):
    """Write a note to Inbox with timestamp."""
    slug = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{NOTE_TITLE}"
    path = f"Inbox/{slug}.md"
    obsidian_cmd("create", f"path={path}", "silent")
    obsidian_cmd("append", f"path={path}", f"content={content}")
    log(f"Wrote inbox note: {path}")


def ensure_agent_rules():
    """Ensure there's a rule about monitoring agents."""
    # Just a placeholder — we won't create a rule for this automatically
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

def monitor_loop(run_dir: Path, poll_interval: int = 60):
    status_file = run_dir / "status.json"
    run_name = run_dir.name

    last_step = -1
    last_stage = ""
    last_status = {}
    stalled_count = 0

    log(f"👁️ Agent watching run: {run_name}")
    log(f"   Status file: {status_file}")
    log(f"   Poll interval: {poll_interval}s")
    log(f"   GPU: {get_gpu_info()}")

    while True:
        status = poll_status(run_dir)
        if status is None:
            log("⚠️ Status file disappeared — run may have ended")
            break

        stage = status.get("stage", "?")
        step = status.get("metrics", {}).get("step", 0)
        loss = status.get("metrics", {}).get("loss", 0)
        lr = status.get("metrics", {}).get("lr", 0)
        error = status.get("error")
        completed = status.get("completed_at")

        # Detect completed
        if completed:
            log(f"✅ Run completed! Stage: {stage}, step: {step}")
            _report_completion(run_dir, status)
            break

        # Detect error
        if error:
            log(f"❌ Run FAILED: {error}")
            _report_failure(run_dir, status)
            break

        # Detect progress
        if step != last_step or stage != last_stage:
            log(f"📈 {stage} | step={step} loss={loss:.4f} lr={lr:.2e} | GPU: {get_gpu_info()}")
            stalled_count = 0
            last_step = step
            last_stage = stage
        else:
            stalled_count += 1
            if stalled_count % 10 == 0:
                log(f"⏳ Still {stage} step={step} (unchanged for {stalled_count * poll_interval // 60} min)")

        # Detect stalled (no update for 30+ minutes)
        if stalled_count > 30:
            log(f"🚨 STALLED: No progress for {stalled_count * poll_interval // 60} minutes")
            write_inbox_note(f"""---
tags:
  - type/bug
  - project: localdistill
---

# 🚨 Training Stall Detected

**Run:** {run_name}
**Stage:** {stage}
**Last step:** {step}
**Stalled for:** {stalled_count * poll_interval // 60} minutes
**GPU:** {get_gpu_info()}
**Last log tail:**
```
{tail_log(run_dir, 10)}
```

## Check
- Is the process alive? `ps -p {status.get('pid', 'N/A')}`
- Is GPU still active? `nvidia-smi`
- Check for OOM or error in logs

## Action Needed
Investigate if training is stuck or just slow.
""")
            break  # Stop monitoring after reporting stall

        time.sleep(poll_interval)

    log("👁️ Agent exiting.")


def _report_completion(run_dir: Path, status: dict):
    run_name = run_dir.name
    stage = status.get("stage", "?")
    metrics = status.get("metrics", {})
    step = metrics.get("step", 0)
    loss = metrics.get("loss", 0)
    lr = metrics.get("lr", 0)
    adapter_path = status.get("adapter_path", "N/A")

    content = f"""---
tags:
  - type/learning
  - project: localdistill
---

# ✅ Training Completed — {run_name}

**Stage:** {stage}
**Final step:** {step}
**Final loss:** {loss:.4f}
**Final lr:** {lr:.2e}
**Adapter:** {adapter_path}
**GPU at end:** {get_gpu_info()}

## Run Log (last 20 lines)
```
{tail_log(run_dir, 20)}
```

## Next Steps
1. Check evaluation results: `cat {run_dir}/eval.log`
2. If preference run, check win rate in status
3. Consider running `./distill run --mode preference --steps evaluate` if not auto-evaluated

## Action Needed
Check evaluation results and save to [[Projects/localdistill/evaluation-results]].
""")
    write_inbox_note(content)


def _report_failure(run_dir: Path, status: dict):
    run_name = run_dir.name
    error = status.get("error", "Unknown")
    content = f"""---
tags:
  - type/bug
  - project: localdistill
---

# ❌ Training Failed — {run_name}

**Error:** {error}
**GPU at failure:** {get_gpu_info()}

## Log Tail (last 30 lines)
```
{tail_log(run_dir, 30)}
```

## Action Needed
1. Check if OOM → reduce batch_size or max_seq_length
2. Check if API error → verify .env keys
3. Save to [[Projects/localdistill/issues]]
""")
    write_inbox_note(content)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Monitor a LocalDistill training run")
    parser.add_argument("--run-dir", type=str, help="Specific run dir (auto-detect if omitted)")
    parser.add_argument("--vault", type=str, default=VAULT, help="Obsidian vault name")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--pid", type=int, default=0, help="Process ID to watch")
    args = parser.parse_args()

    global VAULT
    VAULT = args.vault

    cd = Path(__file__).resolve().parent.parent
    os.chdir(cd)
    log(f"Working dir: {cd}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        result = find_active_run()
        if result is None:
            log("No active run found. Is training running?")
            sys.exit(1)
        run_dir, status = result

    try:
        monitor_loop(run_dir, poll_interval=args.poll)
    except KeyboardInterrupt:
        log("👁️ Agent stopped by user.")


if __name__ == "__main__":
    main()
