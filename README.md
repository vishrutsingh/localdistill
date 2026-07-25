# localdistill

Capture → curate → train your own local model from API teacher traces.

## Contents

- [How it works](#how-it-works)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Curation](#curation)
- [Training](#training)
- [On-Policy Distillation](#on-policy-distillation)
- [Benchmarking](#benchmarking)
- [Deploy (Ollama)](#deploy-ollama)
- [Architecture](#architecture)
- [Research](#research)

## How it works

```
Your tools → Proxy (:8787) → Real API
                │
          SQLite (capture)
                │
        ┌───────┴────────┐
        ▼                ▼
   Quality score    MCP curation
   (regex signals)  (/good /bad)
        │                │
        └───────┬────────┘
                ▼
       curated_training table
                │
        ┌───────┴────────┐
        ▼                ▼
   SFT (Unsloth)    On-policy (teacher API)
        │                │
        └───────┬────────┘
                ▼
           LoRA adapter
                │
                ▼
           Ollama deploy
```

## Quick Start

```bash
cd ~/localdistill
cp .env.example .env          # add OPENROUTER_API_KEY
docker compose up -d           # proxy :8787, dashboard :8000
```

## Demo (verified on RTX 2060 5GB)

```bash
python3 trainer/curate.py                                    # 1. Curate ShareGPT → 42K clean pairs
python3 trainer/train.py --base unsloth/Llama-3.2-3B-Instruct \
  --max-examples 100                                          # 2. SFT — 8min, loss 2.24→0.73
python3 trainer/benchmark.py --adapter latest --tasks gsm8k \
  --limit 10                                                  # 3. Benchmark (CPU fallback)
python3 trainer/train.py --base unsloth/Llama-3.2-3B-Instruct \
  --max-examples 100 --on-policy 1 \
  --teacher-model openrouter/qwen/qwen3-8b                   # 4. On-policy (+5min)
python3 trainer/benchmark.py --adapter latest --limit 10     # 5. Compare delta vs SFT
```

Result: `status: completed`, adapter at `~/localdistill/adapters/<id>/`. GGUF export skipped (upgrade unsloth).

## Pipeline

```bash
python3 trainer/curate.py                                    # 1. Curation
python3 trainer/train.py --max-examples 300                  # 2. SFT
python3 trainer/benchmark.py --adapter latest --limit 10     # 3. Benchmark SFT
python3 trainer/train.py --max-examples 300 --on-policy 1 \
  --teacher-model openrouter/qwen/qwen3-8b                  # 4. On-policy
python3 trainer/benchmark.py --adapter latest --limit 10     # 5. Compare
# GGUF → ollama create localdistill                          # 6. Deploy
```

## Curation

ShareGPT52K → strip HTML → drop non-English → drop single-turn → 42K clean ChatML pairs.

Research: LIMA (1K good > 50K noisy), AlpaGasus (drop bottom 50%), Deita (3-way filter).

```bash
python3 trainer/curate.py
```

## Training

Unsloth LoRA on curated pairs. 300 examples, 3 epochs, ~8min on RTX 2060 5GB.

```bash
python3 trainer/train.py --base unsloth/Llama-3.2-3B-Instruct --max-examples 300
```

## On-Policy Distillation

Teacher model scores student's actual outputs → corrections injected as new training pairs. Tinker blog: SFT alone 65% AIME'24 → on-policy 76.7%.

```bash
python3 trainer/train.py --max-examples 300 \
  --teacher-model openrouter/qwen/qwen3-8b --on-policy 1
```

## Benchmarking

lm-eval runs GSM8K/MMLU on LoRA adapter. CPU fallback for 5GB VRAM.

```bash
python3 trainer/benchmark.py --adapter latest --tasks gsm8k --limit 10
```

## Deploy (Ollama)

```bash
ollama create localdistill -f ~/localdistill/adapters/<id>/gguf/Modelfile
ollama run localdistill "your prompt"
```

## Architecture

| Component | File | Purpose |
|-----------|------|---------|
| Proxy | `proxy/proxy_server.py` | Capture API calls → SQLite |
| Quality | `proxy/quality.py` | 10 regex signals → score |
| MCP | `mcp/mcp_server.py` | Inline curation /good /bad |
| Dashboard | `api/api_server.py` | Tailwind UI at :8000 |
| Curation | `trainer/curate.py` | Filter ShareGPT → ChatML |
| Dataset | `trainer/dataset_exporter.py` | curated_training → JSONL |
| Training | `trainer/train.py` | Unsloth LoRA + on-policy |
| Benchmark | `trainer/benchmark.py` | lm-eval on adapter |

## Research

**Data curation matters.** LIMA (2305.11206): 1K filtered > 50K noisy. Deita (2312.15685): 3-way filter (quality+diversity+difficulty) beats random. AlpaGasus (2307.08701): drop bottom 50%, same score with half data.

**On-policy beats off-policy.** Tinker cookbook: SFT 65% → on-policy 76.7% AIME'24. Student generates, teacher corrects via KL divergence. Batched API calls, not per-sample.

**Small teacher works.** Qwen3-8B teaches Llama 3.2 3B just as well as bigger models. Don't pay DeepSeek for distillation.

**RL needs VRAM.** GRPO on 3B model needs 12GB+. Skipped on RTX 2060 5GB. Use Tinker cloud API for RL recipes.

**Curation before training always.** HTML wrapper divs in ShareGPT burn tokens on markup. 8,163 HTML-div responses found in uncured dataset. Strip first, train second.
