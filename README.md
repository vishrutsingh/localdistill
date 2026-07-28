<p align="center">
  <h1 align="center">LocalDistill</h1>
  <p align="center">A local pipeline that turns cloud LLMs into small models you own.</p>
</p>

---

## Philosophy

Cloud APIs cost money and see your data. Small open models are free and private
but dumber. LocalDistill closes the gap **unattended**: you pick a student
model, a teacher model, and a goal before bed — the pipeline curates data,
queries the teacher, trains, evaluates, and packages the result while you're
offline. In the morning you get an adapter, a verdict, and a full report.

The unit of value is the **run**, not the model:

- **Fully resumable.** Every artifact is content-addressed (a hash of the
  config that produces it). Rerun the same command after a crash, a reboot, or
  a Ctrl-C and it picks up exactly where it stopped — paid teacher API calls
  are never repeated, interrupted training resumes from checkpoints.
- **Methods are comparable.** Five training methods on one selector. Same
  data, same budget, different method → different adapter directory →
  compare win rates side by side. The pipeline is an experiment harness,
  not a one-shot script.
- **Nothing hidden, nothing deleted.** One method selector, no magic flags,
  no silent cleanup. `--force` is the only way to redo work.

```
student (Llama-3.2-3B)          teacher (DeepSeek, GPT-4, ...)
        │                                │
        └──────────┬─────────────────────┘
                   ▼
   curate → [teacher stages] → train → evaluate → benchmark → deploy
                   │
        everything cached under data/<hash> and adapters/<method>_<hash>
                   │
                   ▼
        local model (GGUF / Ollama) + win-rate verdict
```

## Quick Start

```bash
./distill setup                                # check GPU, install deps
echo "OPENROUTER_API_KEY=sk-or-v1-xxx" >> .env # teacher + judge key

./distill run --method sft --mode demo         # 5-min smoke test
./distill run --method sft --mode preference   # real run (UltraFeedback)
```

## Methods

`--method` (or `training.method` in `config.yaml`) selects the full stage chain:

| Method | Chain | Trains on | Use when |
|--------|-------|-----------|----------|
| `sft`  | curate → [generate] → train → evaluate → benchmark → deploy | curated chats or fresh teacher completions | default baseline |
| `reopd` | curate → collect → weight → train → evaluate → benchmark → deploy | decay-weighted multi-turn teacher trajectories | multi-turn data, style transfer |
| `dpo`  | curate → train → evaluate → benchmark → deploy | chosen/rejected pairs | you have preference data (canonical baseline) |
| `orpo` | same as dpo | chosen/rejected pairs | preference data + small GPU (no reference model) |
| `kto`  | same as dpo | pairs as binary good/bad labels | noisy or unpaired preferences |

Details, citations, and tuning knobs: `config.yaml` comments.

## Useful Commands

```bash
# Run a method (resume is automatic — safe to Ctrl-C and rerun)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./distill run --method reopd --mode multiturn

# Scale a run (teacher collection is parallel + resumable)
./distill run --method reopd --mode multiturn --max-examples 1000

# Warm-start from a previous adapter (e.g. ReOPD after SFT)
./distill run --method reopd --from-adapter latest

# Run only some stages / redo everything / preview the plan
./distill run --method reopd --steps curate,collect
./distill run --method sft --force
./distill run --method orpo --dry-run

# Inspect
./distill status        # recent runs + progress
./distill logs          # tail the latest run
./distill adapters      # trained adapters
./distill clean --keep 5

# Dashboard
./distill monitor       # http://localhost:8080
```

Presets (`--mode`): `demo` (10 examples, smoke test), `preference`
(UltraFeedback 5k, teacher completions), `full` (5k + benchmark),
`multiturn` (ShareGPT 200 multi-turn — the right input for ReOPD).

## How runs work

- **Stages are cached.** Each stage writes to `data/…<hash>` — a hash of the
  dataset config, teacher, and method settings that produce it. Same config →
  same path → skipped. Change any input → new path → old artifacts untouched.
- **Teacher stages are parallel and resumable.** `models.teacher_concurrency`
  (default 8) controls concurrent API calls; progress lines show count, rate,
  and ETA; interrupted collections resume from `.partial` files.
- **Training resumes too.** HF checkpoints land in the keyed adapter dir;
  an interrupted run resumes automatically on rerun.
- **Every run ends with a report.** `logs/runs/<id>/training_summary.json`:
  loss, steps, eval win rate, adapter path. The `evaluate` gate (preference
  datasets): student must win/tie >60% of holdout vs chosen, judged by the
  teacher LLM.

## Requirements

- Python 3.10+, NVIDIA GPU (6GB VRAM runs a 3B student in 4bit; 1B for less)
- An OpenRouter (or compatible) API key for the teacher/judge
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on ≤8GB cards

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA OOM | `--student unsloth/Llama-3.2-1B-Instruct`, or lower `max_seq_length`; use the env var above |
| Teacher rate limits | lower `models.teacher_concurrency` in `config.yaml` |
| Stale/wrong cache | artifacts are hashed — only `--force` regenerates; nothing goes stale silently |
| Where's my run? | `./distill status`, `./distill logs`, `logs/runs/<id>/` |

---

<p align="center">
  <sub>Built with <a href="https://github.com/unslothai/unsloth">Unsloth</a> · MIT License</sub>
</p>
