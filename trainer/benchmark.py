#!/usr/bin/env python3
"""ponytail: eval a LoRA adapter. One call, no subprocess."""
import sys, argparse
from pathlib import Path
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

ADAPTER = Path.home() / "localdistill/adapters"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="latest")
    parser.add_argument("--tasks", default="gsm8k")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    adapter = str(sorted(ADAPTER.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[0]) if args.adapter == "latest" else args.adapter
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"Eval: {adapter} on {tasks}")
    model = HFLM(pretrained="unsloth/Llama-3.2-3B-Instruct", peft=adapter, batch_size="auto", trust_remote_code=True, device="cuda")
    results = simple_evaluate(model=model, tasks=tasks, limit=args.limit, batch_size="auto")

    for task, info in results.get("results", {}).items():
        score = info.get("exact_match,strict-match") or info.get("acc,none") or info.get("acc_norm,none")
        print(f"  {task}: {score:.4f}" if score else f"  {task}: {info}")

if __name__ == "__main__":
    main()