# LocalDistill

**Distill knowledge from powerful cloud LLMs into smaller local models you own.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Cloud LLMs (GPT-4, Claude, DeepSeek)                                     │
│                         │                                                   │
│                         ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    LOCALDISTILL PIPELINE                            │  │
│   │                                                                     │  │
│   │   [Curate] ──▶ [Train] ──▶ [On-Policy] ──▶ [Benchmark] ──▶ [Deploy] │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│                         ▼                                                   │
│   Your Local Model (Ollama / llama.cpp)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Table of Contents

- [Quick Start](#quick-start)
- [Local Run](#local-run)
- [Configuration](#configuration)
- [Docker](#docker)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Setup (checks GPU, installs dependencies)
./distill setup

# 2. Start dashboard (http://localhost:8080)
./distill monitor

# 3. Run demo training (~5 min)
./distill run --mode demo
```

That's it. For full training with on-policy distillation:

```bash
# Set API key for teacher model
echo "OPENROUTER_API_KEY=sk-or-v1-xxxx" >> .env

# Run full training with on-policy
./distill run --mode full --on-policy
```

---

## Local Run

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with 4GB+ VRAM (6GB+ recommended)
- **CUDA drivers** installed

### Setup

```bash
# Check environment and install dependencies
./distill setup
```

This will:
- Verify Python and GPU
- Install PyTorch, Unsloth, datasets, FastAPI
- Report your detected configuration

### Running Training

**Demo mode** (50 examples, 1 epoch, ~5 min):
```bash
./distill run --mode demo
```

**Full mode** (5000 examples, 3 epochs):
```bash
./distill run --mode full
```

**With on-policy distillation** (requires API key):
```bash
./distill run --mode full --on-policy
```

**Low VRAM** (<6GB free):
```bash
./distill run --mode demo --student unsloth/Llama-3.2-1B-Instruct
```

### Monitor Dashboard

```bash
# Start dashboard (background)
./distill monitor

# View at http://localhost:8080

# Stop dashboard
./distill stop
```

Dashboard shows:
- Real-time training progress
- Loss curves
- Run history
- Logs and metrics

---

## Configuration

All settings are in `config.yaml`. You can also override via CLI flags.

### API Keys

Create `.env` file for API keys (needed for on-policy distillation):

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# OpenRouter (recommended - one key for all models)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx

# Or use direct API keys
OPENAI_API_KEY=sk-xxxx
ANTHROPIC_API_KEY=sk-ant-xxxx
```

### Models

```yaml
models:
  # Student: local model to train
  student: unsloth/Llama-3.2-3B-Instruct
  
  # Teacher: API model for on-policy distillation
  teacher: openrouter/deepseek/deepseek-chat
```

**Available student models:**

| Model | VRAM | Notes |
|-------|------|-------|
| `unsloth/Llama-3.2-1B-Instruct` | ~3GB | Low VRAM option |
| `unsloth/Llama-3.2-3B-Instruct` | ~5GB | Default, good balance |
| `unsloth/Qwen2.5-3B-Instruct` | ~5GB | Alternative 3B |
| `unsloth/Llama-3.1-8B-Instruct` | ~10GB | Higher quality |

**Available teacher models:**

| Model | Cost | Quality |
|-------|------|---------|
| `openrouter/deepseek/deepseek-chat` | $ | Good |
| `openrouter/anthropic/claude-3-haiku` | $$ | Great |
| `openrouter/openai/gpt-4o-mini` | $$ | Great |
| `openrouter/anthropic/claude-3.5-sonnet` | $$$ | Best |

### Dataset

```yaml
dataset:
  # Local file (default: pre-curated 42K ShareGPT conversations)
  source: file
  path: ./curated_train.jsonl
  
  # Or load from HuggingFace
  source: huggingface
  huggingface:
    dataset_id: RyokoAI/ShareGPT52K
    split: train
```

### Training

```yaml
training:
  lora:
    rank: 16              # LoRA rank (higher = more capacity)
    alpha: 32             # LoRA alpha (usually 2x rank)
    dropout: 0            # Dropout (0 for small datasets)
    
  hyperparams:
    learning_rate: 2.0e-4
    epochs: 3
    batch_size: 2
    gradient_accumulation_steps: 4
    max_seq_length: 2048  # Reduce to 1024 for low VRAM
    warmup_steps: 5
    weight_decay: 0.01
    lr_scheduler: linear
    seed: 42
    
  quantization: 4bit      # 4bit | 8bit | none
```

### Curation

```yaml
curation:
  max_examples: 5000      # null = use all
  min_quality_score: 0.0
  
  filters:
    min_turns: 2          # Minimum conversation turns
    max_turns: 50         # Maximum conversation turns
    min_chars: 100        # Minimum total characters
    max_chars: 50000      # Maximum total characters
    exclude_code_heavy: false
```

### On-Policy Distillation (ReOPD)

```yaml
on_policy:
  enabled: true
  teacher_query_interval: 10    # Controls decay: kappa = 1 - 1/interval
  decay_function: exponential   # exponential | linear
  exponential_lambda: 0.9       # Direct kappa value
```

Two-phase offline distillation (based on ReOPD paper):
1. **Phase 1**: Collect teacher trajectories once (cached as `teacher_pool.jsonl`)
2. **Phase 2**: Train with step-decay weighting — earlier turns weighted higher (w_t = κ^t)

Token usage is logged during Phase 1. Phase 2 uses zero API calls.

### Benchmark

```yaml
benchmark:
  enabled: true
  tasks: [gsm8k]          # gsm8k, mmlu, hellaswag, truthfulqa
  limit: 50               # Examples per task
  compare_base: true      # Compare with base model
```

### Deploy

```yaml
deploy:
  gguf:
    enabled: true
    quantization: q4_k_m  # q4_k_m, q5_k_m, q8_0, f16
    
  ollama:
    enabled: false
    model_name: localdistill
    auto_register: true
```

### Presets

Built-in presets override settings:

| Mode | Examples | Epochs | Use Case |
|------|----------|--------|----------|
| `demo` | 50 | 1 | Quick test (~5 min) |
| `full` | 5000 | 3 | Production training |
| `custom` | Your config | Your config | Full control |

### Full config.yaml Example

```yaml
run_mode: demo

dataset:
  source: file
  path: ./curated_train.jsonl

models:
  student: unsloth/Llama-3.2-3B-Instruct
  teacher: openrouter/deepseek/deepseek-chat

curation:
  max_examples: 5000
  filters:
    min_turns: 2
    max_turns: 50

training:
  lora:
    rank: 16
    alpha: 32
  hyperparams:
    learning_rate: 2.0e-4
    epochs: 3
    batch_size: 2
    max_seq_length: 2048
  quantization: 4bit

on_policy:
  enabled: false
  teacher_query_interval: 10

benchmark:
  enabled: true
  tasks: [gsm8k]
  limit: 50

deploy:
  gguf:
    enabled: true
    quantization: q4_k_m
  ollama:
    enabled: false
    model_name: localdistill

logging:
  level: INFO
  dir: ./logs

presets:
  demo:
    curation:
      max_examples: 50
    training:
      hyperparams:
        epochs: 1
  full:
    curation:
      max_examples: 5000
    training:
      hyperparams:
        epochs: 3
```

---

## Docker

Docker is optional. Use it if you want containerized runs with NVIDIA Container Toolkit.

### Prerequisites

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Running with Docker

```bash
# Build images
docker compose build

# Start monitor dashboard
docker compose up -d monitor

# Run training
docker compose run --rm trainer python distill.py run --mode demo

# Run full training with on-policy
docker compose run --rm trainer python distill.py run --mode full --on-policy

# Stop everything
docker compose down
```

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `monitor` | 8080 | Dashboard |
| `trainer` | - | GPU training container |
| `proxy` | 8787 | API capture (optional) |

---

## CLI Reference

```
./distill <command> [options]

COMMANDS:
  setup               Check environment, install dependencies
  run [options]       Run training pipeline
  monitor [--fg]      Start dashboard at :8080
  stop                Stop dashboard
  status              Show recent runs
  logs                Tail latest logs

RUN OPTIONS:
  --mode <mode>       demo | full | custom
  --student <model>   Override student model
  --teacher <model>   Override teacher model  
  --on-policy         Enable on-policy distillation
  --max-examples <n>  Override max training examples
  --epochs <n>        Override epochs
  --steps <steps>     Run specific steps: curate,train,benchmark,deploy
  --dry-run           Show plan without running
```

### Examples

```bash
# Quick demo
./distill run --mode demo

# Full training
./distill run --mode full

# Full with on-policy
./distill run --mode full --on-policy

# Low VRAM
./distill run --mode demo --student unsloth/Llama-3.2-1B-Instruct

# Custom settings
./distill run --max-examples 1000 --epochs 2

# Only train (skip benchmark/deploy)
./distill run --steps curate,train

# Dry run
./distill run --dry-run
```

---

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch sees GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory

Reduce VRAM usage in `config.yaml`:
```yaml
training:
  hyperparams:
    batch_size: 1
    max_seq_length: 1024
```

Or use smaller model:
```bash
./distill run --student unsloth/Llama-3.2-1B-Instruct
```

### Training too slow

Use fewer examples:
```bash
./distill run --max-examples 100 --epochs 1
```

### View logs

```bash
# Latest run logs
./distill logs

# All runs
ls logs/runs/

# Specific run
cat logs/runs/2024-01-15_abc123/run.log
```

### Dashboard not showing data

Check paths:
```bash
# Logs should be in
ls logs/runs/

# Dashboard reads from these
curl http://localhost:8080/api/runs
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
