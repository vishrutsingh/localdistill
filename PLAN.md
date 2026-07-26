# LocalDistill - Proof of Concept Plan

## Goal

Prove the distillation pipeline works: train a smaller local model on human-preferred responses, measure if it improves.

## Target User (Hypothetical)

Business running ~10k LLM queries/day. At $0.002/query, that's ~$600/month. If a local model handles 50% of queries, saves $300/month.

## Phase 1: Dataset Validation

### Dataset Choice

**Using:** `HuggingFaceH4/ultrafeedback_binarized`

**Why this one:**
- GPT-4 scored response quality (not just safety labels)
- Already in message format (no parsing needed)
- 60k+ examples available
- Quality preference, not refusal preference

**Rejected alternatives:**
- `lmsys/chatbot_arena_conversations` - gated, requires auth
- `Anthropic/hh-rlhf` - about safety/refusals, not quality
- `ShareGPT` - no preference labels, mixed quality

### Data Split

- 90% train
- 10% holdout (for evaluation)

## Phase 2: Training

### Model

- **Student:** Llama 3.2 3B (via Unsloth)
- **Why:** Small enough to run locally, big enough to learn something

### Method

- SFT on "chosen" responses only
- Standard LoRA fine-tuning (rank 16, alpha 32)
- 1 epoch (can increase if needed)
- Checkpoints every 50 steps

### What we're testing

Does training on human-preferred responses make the model produce better responses?

## Phase 3: Evaluation

### Method

1. Take holdout prompts
2. Generate student response
3. Compare to "chosen" response from dataset
4. Use judge (heuristic or LLM) to pick winner

### Success Criteria

**Student wins or ties >60% of the time** compared to base (untrained) model.

If this fails, the pipeline doesn't work and we stop here.

## Phase 4: Real Usage (Future)

If Phase 1-3 succeed:

1. Set up proxy to capture my own LLM usage
2. Add thumbs up/down feedback UI
3. On thumbs down: fallback to teacher, log the correction
4. Periodically retrain on accumulated corrections
5. Measure: does fallback rate decrease over time?

## What Needs to Be Built

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset loader (preference format) | DONE | `lib/dataset.py` |
| Train/holdout split | DONE | 90/10 split |
| SFT training | DONE | `distill.py` |
| Checkpointing | DONE | Every 50 steps |
| Evaluation harness | DONE | `lib/evaluate.py` |
| LLM judge | TODO | Using heuristic for now |
| Feedback loop | FUTURE | After PoC succeeds |

## Commands

```bash
# Full pipeline
./distill run --mode preference

# Step by step
./distill run --mode preference --steps curate
./distill run --mode preference --steps train  
./distill run --mode preference --steps evaluate

# With more data
./distill run --mode preference --max-examples 5000
```

## Open Questions

1. Is simple heuristic judge good enough, or do we need LLM judge?
2. How many training examples are enough? Starting with 1k.
3. Should we compare against base model too, not just chosen?

## Non-Goals (For Now)

- Production-ready proxy
- Multi-round on-policy training
- Deployment to Ollama
- Benchmarks (GSM8K, MMLU) - not representative of real usage
