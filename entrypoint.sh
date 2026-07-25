#!/bin/bash
set -e
echo "[localdistill] Starting automated pipeline"
echo "[localdistill] Step 1/3: Curation"
python3 /app/trainer/curate.py
echo "[localdistill] Step 2/3: Training"
python3 /app/trainer/train.py \
  --base "${BASE_MODEL:-unsloth/Llama-3.2-3B-Instruct}" \
  --max-examples "${MAX_EXAMPLES:-300}" \
  --teacher-model "${TEACHER_MODEL:-}" \
  --on-policy "${ON_POLICY:-0}"
echo "[localdistill] Step 3/3: Benchmark"
python3 /app/trainer/benchmark.py \
  --adapter latest \
  --tasks "${BENCH_TASKS:-gsm8k}" \
  --limit "${BENCH_LIMIT:-10}"
echo "[localdistill] Pipeline complete"
