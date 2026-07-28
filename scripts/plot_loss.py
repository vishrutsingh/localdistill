#!/usr/bin/env python3
"""
Plot training loss curve from events.jsonl

Usage:
    python scripts/plot_loss.py                    # Plot latest run
    python scripts/plot_loss.py logs/runs/YYYY-MM-DD_RUNID  # Specific run
    python scripts/plot_loss.py --output loss.png  # Save to file
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def find_latest_run() -> Path:
    runs_dir = Path("logs/runs")
    if not runs_dir.exists():
        raise FileNotFoundError("No logs/runs directory")
    runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "events.jsonl").exists():
            return r
    raise FileNotFoundError("No events.jsonl found")


def parse_events(events_path: Path):
    steps, losses, lrs, phases = [], [], [], []
    by_phase = defaultdict(lambda: {"steps": [], "losses": []})

    with open(events_path) as f:
        for line in f:
            try:
                event = json.loads(line)
                data = event.get("data", {})
                if "loss" in data and "step" in data:
                    step = data["step"]
                    loss = data["loss"]
                    lr = data.get("lr", 0)
                    phase = data.get("phase", event.get("stage", "train"))

                    steps.append(step)
                    losses.append(loss)
                    lrs.append(lr)
                    phases.append(phase)

                    by_phase[phase]["steps"].append(step)
                    by_phase[phase]["losses"].append(loss)
            except Exception:
                continue

    return steps, losses, lrs, phases, dict(by_phase)


def plot_matplotlib(by_phase: dict, output: Path = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    colors = {"sft": "#2196f3", "reopd": "#ff9800", "train": "#2196f3", "on_policy": "#ff9800"}

    for phase, data in by_phase.items():
        if not data["steps"]:
            continue
        color = colors.get(phase, "#999")
        ax1.plot(data["steps"], data["losses"], color=color, alpha=0.7,
                label=phase, linewidth=1)

    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Convergence annotation
    for phase, data in by_phase.items():
        if len(data["losses"]) >= 20:
            last_20 = data["losses"][-20:]
            final = last_20[-1]
            avg_last = sum(last_20) / len(last_20)
            ax1.axhline(y=avg_last, color=colors.get(phase, "#999"),
                       linestyle="--", alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Phase")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_ascii(by_phase: dict):
    """ASCII fallback plot."""
    for phase, data in by_phase.items():
        if not data["steps"]:
            continue
        losses = data["losses"]
        print(f"\n=== {phase} ===")
        print(f"  Steps: {len(losses)}")
        print(f"  Start loss: {losses[0]:.4f}")
        print(f"  End loss:   {losses[-1]:.4f}")
        print(f"  Min loss:   {min(losses):.4f}")
        print(f"  Max loss:   {max(losses):.4f}")
        if len(losses) >= 2:
            improvement = losses[0] - losses[-1]
            print(f"  Δ:          {improvement:+.4f}")


def main():
    ap = argparse.ArgumentParser(description="Plot training loss")
    ap.add_argument("run_dir", nargs="?", help="Run directory (auto-detect if omitted)")
    ap.add_argument("--output", "-o", help="Output file (png)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    events_path = run_dir / "events.jsonl"

    if not events_path.exists():
        print(f"No events.jsonl in {run_dir}")
        return 1

    steps, losses, lrs, phases, by_phase = parse_events(events_path)

    if not steps:
        print("No training steps found in events")
        return 1

    print(f"Run: {run_dir.name}")
    print(f"Total steps: {len(steps)}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Min loss: {min(losses):.4f}")
    if len(losses) >= 10:
        last10 = sum(losses[-10:]) / 10
        print(f"Last 10 avg: {last10:.4f}")

    plot_ascii(by_phase)

    try:
        plot_matplotlib(by_phase, Path(args.output) if args.output else None)
    except Exception as e:
        print(f"Plot failed: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
