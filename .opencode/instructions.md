# LocalDistill Project Instructions

## What this project does

Distills knowledge from expensive cloud LLMs into smaller local models. User runs LLM through proxy, we train a local model on their usage patterns to reduce cost.

## Current Status

**Proof of Concept phase** - validating the pipeline works before building full user-facing proxy.

### Implemented
- Dataset loader for preference format (UltraFeedback)
- Train/holdout split (90/10)
- SFT training on "chosen" responses
- Evaluation harness (student vs chosen comparison)
- Checkpointing support

### Not yet implemented
- LLM judge for evaluation (using simple heuristic for now)
- User feedback loop (thumbs up/down in proxy)
- Multi-round on-policy training

## Key Files

- `distill.py` - Main pipeline orchestrator
- `lib/dataset.py` - Dataset loading, preference pairs, splits
- `lib/evaluate.py` - Evaluation harness
- `lib/config.py` - Configuration dataclasses
- `config.yaml` - User configuration
- `PLAN.md` - Detailed proof of concept plan

## Running

```bash
# Full pipeline with preference data
./distill run --mode preference

# Step by step
./distill run --mode preference --steps curate
./distill run --mode preference --steps train
./distill run --mode preference --steps evaluate
```

## Success Criteria

Student model wins or ties >60% against chosen responses on holdout set.

## Architecture Decisions

1. **Dataset**: Using HuggingFaceH4/ultrafeedback_binarized (GPT-4 scored quality preferences)
2. **Training**: SFT on chosen responses only (not DPO for simplicity)
3. **Evaluation**: Simple heuristic judge (no API cost), can upgrade to LLM judge later
4. **Target user**: Business with ~10k queries/day (hypothetical for now)
