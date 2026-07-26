# LocalDistill

Distill cloud LLM knowledge into local models you own.

```
Cloud LLM (GPT-4, DeepSeek)    Your Data
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────────┐
         │    LocalDistill      │
         │                      │
         │  curate → train →    │
         │  evaluate → deploy   │
         └──────────────────────┘
                    │
                    ▼
         Local Model (Ollama)
         - Runs on your hardware
         - No API costs
         - Private
```

## Quick Start

```bash
# Setup (checks GPU, installs deps)
./distill setup

# Run training
./distill run --mode demo      # 5 min test
./distill run --mode full      # production run
```

## Requirements

- Python 3.10+
- NVIDIA GPU, 4GB+ VRAM
- CUDA drivers

## CLI

```
./distill <command>

Commands:
  setup       Check environment, install dependencies
  run         Run training pipeline
  monitor     Start dashboard at :8080
  stop        Stop dashboard
  status      Show recent runs
  logs        Tail latest logs

Run options:
  --mode <mode>       demo | full | preference | custom
  --steps <steps>     curate,train,evaluate,benchmark,deploy
  --max-examples <n>  Training examples
  --epochs <n>        Training epochs
  --student <model>   Student model (default: Llama-3.2-3B)
  --teacher <model>   Teacher/judge model (default: DeepSeek)
  --on-policy         Enable on-policy distillation
  --dry-run           Show plan only
```

**Examples:**

```bash
# Preference learning (UltraFeedback dataset)
./distill run --mode preference

# Step by step
./distill run --mode preference --steps curate
./distill run --mode preference --steps train
./distill run --mode preference --steps evaluate

# Low VRAM (<6GB)
./distill run --student unsloth/Llama-3.2-1B-Instruct

# Custom
./distill run --max-examples 2000 --epochs 2
```

## Configuration

Edit `config.yaml` or use CLI flags.

**Models:**

| Student | VRAM | Notes |
|---------|------|-------|
| `unsloth/Llama-3.2-1B-Instruct` | ~3GB | Low VRAM |
| `unsloth/Llama-3.2-3B-Instruct` | ~5GB | Default |
| `unsloth/Llama-3.1-8B-Instruct` | ~10GB | Higher quality |

**API key** (for teacher/judge):

```bash
echo "OPENROUTER_API_KEY=sk-or-v1-xxx" >> .env
```

## How It Works

**Preference mode** (current focus):

1. Load preference dataset (UltraFeedback - GPT-4 scored pairs)
2. Split 90% train / 10% holdout
3. SFT on "chosen" responses
4. Evaluate: student vs chosen, judged by teacher LLM
5. Success = student wins/ties >60%

**On-policy mode** (advanced):

1. Collect teacher responses on training prompts
2. Train with step-decay weighting (earlier turns weighted higher)
3. Benchmark against base model

## Troubleshooting

**GPU not detected:**
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory:**
```bash
./distill run --student unsloth/Llama-3.2-1B-Instruct
```

Or in config.yaml:
```yaml
training:
  hyperparams:
    batch_size: 1
    max_seq_length: 1024
```

**View logs:**
```bash
./distill logs
```

## License

MIT
