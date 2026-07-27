# Shipping Strategy Grilling Session

Date: 2026-07-27

## Context

Discussion about how to ship LocalDistill and monetize it.

## Key Points Discussed

### Initial Pitch Evolution

1. **Started with:** Self-hosted, privacy-first CLI tool (compete with OpenPipe on privacy)
2. **Pivoted to:** Hosted API proxy + training service
3. **Then to:** Open-core model (free CLI, paid chat UI + API)

### The Open-Core Model

```
FREE: CLI pipeline (train locally, export GGUF, run on Ollama)
PAID: Hosted chat UI + managed training via API
```

**Problems identified:**
- Free tier is the whole product — why pay for UI when Ollama/Open WebUI exist?
- API for training = one-time cost, no recurring revenue
- API for inference = GPU costs for you
- Competing with well-funded players (OpenPipe, Together, Anyscale)

### Economics Question (Unanswered)

- Customer pays $X/month to OpenAI currently
- They pay you $Y to train + $Z/month to run inference
- Y + Z < X?

Need napkin math: model size, training cost, inference cost, break-even point.

### Immediate Revenue Options

If money is needed now:

1. **Freelance/contract** — ship features for others
2. **Fine-tuning as a service** — manually do what LocalDistill automates, charge $500-2k per model
3. **Consulting** — "I'll help you reduce your LLM costs"

**Best option:** Find one business spending $1k+/month on OpenAI. Offer: "I'll train a model on your data that cuts your bill in half. $500 flat fee. Refund if it doesn't work."

### Post-PoC Launch Options

| Option | Effort | What you learn |
|--------|--------|----------------|
| Open source + HN | 1 day | Do devs care? |
| Landing page | 2 hrs | Nothing (vanity metrics) |
| 3 manual beta users | 1-2 weeks | Do businesses care? What do they actually need? |
| Paid offer post | 1 hr | Will anyone pay? |

**Recommendation:** Landing pages are procrastination. Manual beta users teach the most.

## Blockers

- PoC not done yet. Last run: 37.5% win rate (target: >60%)
- Need to run: `./distill run --mode preference`

## Next Steps

1. Run PoC training (5k examples, 3 epochs, DeepSeek judge)
2. Hit >60% win rate
3. Decide launch strategy (open source vs manual beta users vs paid offer)

## Open Questions

1. What's the actual cost savings math?
2. Who's the real customer? (Devs won't pay, businesses compare to funded alternatives)
3. What's the smallest thing that proves people want this?
