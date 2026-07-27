<p align="center">
  <h1 align="center">LocalDistill</h1>
  <p align="center">Distill cloud LLM knowledge into local models you own.</p>
</p>

<p align="center">
  <a href="#features"><strong>Features</strong></a> ·
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#usage"><strong>Usage</strong></a> ·
  <a href="#configuration"><strong>Configuration</strong></a>
</p>

---

**LocalDistill** turns expensive cloud LLMs into small, fast models that run on your hardware. You curate data, train a local model with LoRA, evaluate it against the original, and deploy — no API bills, no data leaving your machine.

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

## Why

Cloud APIs cost money and send your data off-device. Running local models fixes both, but a raw open-source model lags behind GPT-4 on most tasks. LocalDistill bridges the gap: feed the teacher's best outputs into a smaller model, and the smaller model learns to match the teacher on the tasks you care about.

## Features

- **Curate** — Filter and score training data from HuggingFace or local files
- **Train** — SFT, DPO, or on-policy ReOPD with Unsloth and LoRA
- **Evaluate** — Head-to-head judging by the teacher LLM; success = local model wins >60%
- **Deploy** — Export to OllamaGGUF adapters, or serve directly
- **Monitor** — Built-in dashboard at `:8080` for run tracking
- **Docker** — Containerized training pipeline included

## Quick Start

```bash
# Setup (checks GPU, installs deps)
./distill setup

# Run training
./distill run --mode demo      # 5 min test
./distill run --mode full      # production run
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU, 4GB+ VRAM
- CUDA drivers

## Usage

### Commands

```
./distill <command>

  setup     Check environment, install dependencies
  run       Run training pipeline
  monitor   Start dashboard at :8080
  stop      Stop dashboard
  status    Show recent runs
  logs      Tail latest logs
```

### Run Options

| Flag              | Description                                |
|-------------------|--------------------------------------------|
| `--mode`          | `demo` \| `full` \| `preference` \| `custom`    |
| `--steps`         | `curate,train,evaluate,benchmark,deploy`      |
| `--max-examples`  | Training examples                          |
| `--epochs`        | Training epochs                            |
| `--student`       | Student model (default: Llama-3.2-3B)      |
| `--teacher`       | Teacher/judge model (default: DeepSeek)    |
| `--on-policy`     | Enable on-policy distillation              |
| `--dry-run`       | Show plan only                             |

### Examples

```bash
# Preference learning (UltraFeedback)
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

All settings live in `config.yaml`. Override any with CLI flags.

### Student Models

| Model | VRAM | Best For |
|-------|------|----------|
| `unsloth/Llama-3.2-1B-Instruct` | ~3GB | Laptops, entry GPU |
| `unsloth/Llama-3.2-3B-Instruct` | ~5GB | Most desktops (default) |
| `unsloth/Llama-3.1-8B-Instruct` | ~10GB | Higher quality output |

### API Keys

Add your OpenRouter key for the teacher:

```bash
echo "OPENROUTER_API_KEY=sk-or-v1-xxx" >> .env
```

## How It Works

### Preference Mode (default)

The simplest way to improve a local model. Uses publicly available preference datasets scored by GPT-4.

1. **Load** a preference dataset (e.g. UltraFeedback — ~60K GPT-4 scored pairs)
2. **Split** 90% train / 10% holdout by quality bracket
3. **Train** SFT on the "chosen" responses with LoRA adapters
4. **Evaluate** head-to-head: student vs chosen, judged by the teacher LLM
5. **Success** = student wins or ties on >60% of holdout examples

### On-Policy Mode (advanced)

Collect fresh teacher responses on your own prompts, then train with replay-weighted SFT.

1. **Sample** teacher responses on training prompts via the teacher LLM
2. **Weight** by turn position (earlier turns matter more; later turns matter less)
3. **Train** on the weighted teacher trajectories
4. **Benchmark** against the base model on the same prompts

## Troubleshooting

| Problem | Fix |
|---------|-----|
| GPU not detected | Run `nvidia-smi` and `python3 -c "import torch; print(torch.cuda.is_available())"` |
| Out of memory | `./distill run --student unsloth/Llama-3.2-1B-Instruct` or lower `batch_size` and `max_seq_length` in `config.yaml` |
| Slow training | Switch to `--mode demo` or lower `--epochs` |

View logs anytime:
```bash
./distill logs
```

---

<p align="center">
  <sub>Built with <a href="https://github.com/unslothai/unsloth">Unsloth</a> · MIT License</sub>
</p>
