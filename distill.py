#!/usr/bin/env python3
"""
LocalDistill - Main Orchestrator

One method selector, explicit stage chains, content-addressed artifacts.

Methods (config training.method or --method):
  sft    - curate → [generate] → train → evaluate → benchmark → deploy
  reopd  - curate → collect → weight → train → evaluate → benchmark → deploy
  dpo    - curate → train → evaluate → benchmark → deploy (preference pairs)
  orpo   - same chain, reference-free (best on small GPUs)
  kto    - same chain, binary labels instead of pairs

Resume model: every artifact path is a hash of the config that produces it
(see lib/methods.py). Rerunning the same command reuses completed stages
automatically; interrupted teacher generation resumes from .partial files;
interrupted training resumes from HF checkpoints in the keyed adapter dir.
Nothing is deleted as "cleanup" — use --force to redo a run's stages.

Usage:
    python distill.py run --method sft --mode preference
    python distill.py run --method reopd --mode multiturn --from-adapter latest
    python distill.py run --method orpo --mode preference
    python distill.py run --method reopd --steps curate,collect
    python distill.py run --method sft --force        # redo all stages
    python distill.py status
"""

import os
import sys
import types
import uuid
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# ── Setup ─────────────────────────────────────────────────────────────────────

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Python 3.14 compatibility — datasets' custom Pickler._batch_setitems overrides
# a 3-arg stdlib method with a 2-arg one. Monkey-patch directly (unsloth pins
# datasets <4.4, so upgrading is not an option).
if sys.version_info >= (3, 14):
    try:
        import datasets.utils._dill as _dill_mod
        _parent_batch = __import__('pickle')._Pickler._batch_setitems

        def _batch_setitems_compat(self, items, obj=None):
            if getattr(self, '_legacy_no_dict_keys_sorting', False):
                return _parent_batch(self, items, obj)
            try:
                items = sorted(items)
            except Exception:
                from datasets.fingerprint import Hasher
                items = sorted(items, key=lambda x: Hasher.hash(x[0]))
            return _parent_batch(self, items, obj)

        _dill_mod.Pickler._batch_setitems = _batch_setitems_compat
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).parent))

from lib.config import load_config, save_config, print_config, validate_config, Config, METHODS, PREFERENCE_METHODS
from lib.logger import DistillLogger, create_logger, PipelineStage as Stage
from lib.dataset import DatasetLoader, split_dataset
from lib.deploy import (
    export_gguf, register_ollama_model, list_adapters,
    check_ollama_installed, cleanup_old_adapters,
)
from lib.methods import compute_artifacts, latest_adapter, ADAPTERS_DIR


# ── Trainer helpers ───────────────────────────────────────────────────────────

def _make_log_callback(logger, method: str, total_steps: int):
    """TrainerCallback that streams loss into the distill logger."""
    from transformers import TrainerCallback

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                logger.log_training_step(
                    state.global_step, logs["loss"], logs.get("learning_rate"),
                    total_steps=total_steps, phase=method,
                )

    return LogCallback()


def _import_dpo_trainer():
    """Import DPOTrainer, working around TRL 0.24's broken optional-dep chain.

    trl.trainer.callbacks hard-imports mergekit / llm_blender / weave; llm_blender
    is incompatible with modern transformers. DPOTrainer only needs
    SyncRefModelCallback from that module (used with sync_ref_model=True, which
    we never set), so stub the module if the real import fails.
    """
    try:
        from trl import DPOTrainer, DPOConfig
        return DPOTrainer, DPOConfig
    except RuntimeError:
        import importlib.machinery
        from transformers import TrainerCallback

        class SyncRefModelCallback(TrainerCallback):
            pass

        stub = types.ModuleType("trl.trainer.callbacks")
        stub.__spec__ = importlib.machinery.ModuleSpec("trl.trainer.callbacks", loader=None)
        stub.SyncRefModelCallback = SyncRefModelCallback
        sys.modules["trl.trainer.callbacks"] = stub
        from trl import DPOTrainer, DPOConfig
        return DPOTrainer, DPOConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class DistillPipeline:
    """Runs the stage chain for one method. Stage outputs are content-addressed."""

    def __init__(self, config: Config, logger: DistillLogger, method: str,
                 from_adapter: Optional[str] = None, force: bool = False):
        self.config = config
        self.logger = logger
        self.run_id = logger.run_id
        self.method = method
        self.from_adapter = from_adapter
        self.force = force
        self.paths = compute_artifacts(config, method, from_adapter)
        self.adapter_path: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self._training_start_time: Optional[datetime] = None

    # ── Chain ────────────────────────────────────────────────────────────────

    def chain(self) -> List[str]:
        """Method's stage chain filtered by config."""
        steps = list(METHODS[self.method])
        if not self.config.training.generate_teacher:
            steps = [s for s in steps if s != "generate"]
        if self.config.dataset.source != "preference":
            steps = [s for s in steps if s != "evaluate"]
        if not self.config.benchmark.enabled:
            steps = [s for s in steps if s != "benchmark"]
        if not (self.config.deploy.gguf.enabled or self.config.deploy.ollama.enabled):
            steps = [s for s in steps if s != "deploy"]
        return steps

    def run(self, steps: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Any]:
        chain = self.chain()
        if steps:
            unknown = [s for s in steps if s not in chain]
            if unknown:
                return {"run_id": self.run_id, "status": "failed",
                        "error": f"Unknown/excluded steps {unknown}. Chain for method "
                                 f"'{self.method}' with this config: {chain}"}
            chain = [s for s in chain if s in steps]

        self.logger.header("LOCALDISTILL PIPELINE")

        errs = validate_config(self.config)
        if errs:
            for e in errs:
                self.logger.error(f"Config error: {e}")
            return {"run_id": self.run_id, "status": "failed", "error": "; ".join(errs)}

        print_config(self.config, paths={k: v for k, v in self.paths.items()
                                         if k in ("curate_dir", "train_dataset", "adapter_dir")})
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Method: {self.method} | Stages: {' -> '.join(chain)}")
        if self.config.training.generate_teacher and self.method != "sft":
            self.logger.info("generate_teacher ignored (only applies to method=sft)")
        if self.from_adapter:
            self.logger.info(f"Warm start: {self.from_adapter}")
        if self.force:
            self.logger.info("--force: existing stage outputs will be regenerated")

        if dry_run:
            self.logger.info("DRY RUN - No changes")
            for step in chain:
                marker = self._stage_output_marker(step)
                state = "cached" if marker and marker.exists() and not self.force else "would run"
                self.logger.info(f"  {step:10s} {state}")
            return {"run_id": self.run_id, "status": "dry_run", "steps": chain}

        config_snapshot = self.logger.run_dir / "config.yaml"
        save_config(self.config, str(config_snapshot))

        try:
            for step in chain:
                getattr(self, f"_stage_{step}")()

            self.logger.complete("Pipeline finished successfully")
            self._write_training_summary()
            return {
                "run_id": self.run_id,
                "status": "completed",
                "adapter_path": self.adapter_path,
                "metrics": self.metrics,
            }

        except Exception as e:
            import traceback
            self.logger.fail(str(e))
            self.logger.error(traceback.format_exc())
            self._write_training_summary()
            return {"run_id": self.run_id, "status": "failed", "error": str(e)}

    def _stage_output_marker(self, step: str) -> Optional[Path]:
        """File whose existence means the stage completed (for dry-run display)."""
        p = self.paths
        return {
            "curate": p["train_data"],
            "generate": p["teacher_singleturn"],
            "collect": p["teacher_trajectories"],
            "weight": p["reopd_sampled"],
            "train": p["adapter_done"],
            "evaluate": None,
            "benchmark": None,
            "deploy": None,
        }.get(step)

    def _cached(self, marker: Path, what: str) -> bool:
        """Standard cache check. Returns True if the stage should be skipped."""
        if marker.exists() and not self.force:
            self.logger.info(f"{what} exists, skipping ({marker})")
            return True
        return False

    # ── Curate ────────────────────────────────────────────────────────────────

    def _stage_curate(self):
        self.logger.set_stage(Stage.CURATE)
        p = self.paths
        is_pref = self.config.dataset.source == "preference"
        complete = p["train_data"].exists() and (not is_pref or (p["holdout"].exists() and p["pairs"].exists()))
        if complete and self._cached(p["train_data"], "Curated dataset"):
            self.metrics["curated_examples"] = sum(1 for _ in open(p["train_data"]))
            return

        if self.force and p["curate_dir"].exists():
            shutil.rmtree(p["curate_dir"])
        p["curate_dir"].mkdir(parents=True, exist_ok=True)

        loader = DatasetLoader(self.config, self.logger)

        if is_pref:
            self.logger.info("Loading preference dataset...")
            pairs = loader.load_preference_pairs()
            self.logger.info(f"Loaded {len(pairs)} preference pairs")

            train_pairs, holdout_pairs = split_dataset(
                pairs, train_ratio=1.0 - self.config.dataset.holdout_ratio,
                seed=self.config.training.hyperparams.seed,
            )
            train_convs = [pr.to_chosen_conversation() for pr in train_pairs]
            count = loader.export(train_convs, str(p["train_data"]), "chatml")

            # Preference methods (dpo/orpo/kto) train on the full pairs
            with open(p["pairs"], "w") as f:
                for pr in train_pairs:
                    f.write(json.dumps({
                        "prompt": pr.prompt,
                        "chosen": pr.chosen,
                        "rejected": pr.rejected,
                    }) + "\n")

            with open(p["holdout"], "w") as f:
                for pr in holdout_pairs:
                    f.write(json.dumps({
                        "prompt": pr.prompt,
                        "chosen": pr.chosen,
                        "rejected": pr.rejected,
                        "score_chosen": pr.score_chosen,
                        "score_rejected": pr.score_rejected,
                    }) + "\n")

            self.metrics["train_examples"] = len(train_pairs)
            self.metrics["holdout_examples"] = len(holdout_pairs)
            self.logger.success(f"Train: {count} | Holdout: {len(holdout_pairs)} → {p['curate_dir']}")
        else:
            self.logger.info("Loading dataset...")
            conversations = loader.load()
            curated = loader.curate(conversations)
            count = loader.export(curated, str(p["train_data"]), "chatml")
            self.metrics["curated_examples"] = count
            self.logger.success(f"Dataset ready: {count} examples → {p['train_data']}")

        with open(p["curate_meta"], "w") as f:
            json.dump({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": self.config.dataset.source,
                "examples": self.metrics.get("train_examples") or self.metrics.get("curated_examples"),
                "run_id": self.run_id,
            }, f, indent=2)

    # ── Teacher generation helpers ────────────────────────────────────────────

    def _require_train_data(self, stage: str) -> List[Dict]:
        """Load curated conversations; fail with a clear message if missing."""
        train_data = self.paths["train_data"]
        if not train_data.exists():
            raise FileNotFoundError(
                f"{stage} needs {train_data} — run the curate stage first "
                f"(it is content-addressed, so one curate serves all methods)"
            )
        from datasets import load_dataset
        return load_dataset("json", data_files=str(train_data), split="train")

    def _teacher_complete(self, messages: List[Dict], max_tokens: int, retries: int = 2) -> Optional[str]:
        import litellm
        import time
        for attempt in range(retries + 1):
            try:
                resp = litellm.completion(
                    model=self.config.models.teacher,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < retries:
                    time.sleep(5 * (attempt + 1))
                else:
                    self.logger.warning(f"Teacher query failed after {retries + 1} attempts: {e}")
        return None

    def _resume_done_idxs(self, partial: Path) -> set:
        """Indices already in the partial file. Old-format lines (no idx field)
        were written in dataset order, so position == index."""
        done = set()
        if partial.exists():
            with open(partial) as f:
                for pos, line in enumerate(f):
                    try:
                        done.add(json.loads(line).get("idx", pos))
                    except json.JSONDecodeError:
                        continue
        if done:
            self.logger.info(f"Resuming from partial: {len(done)} examples already done")
        return done

    def _run_teacher_stage(self, final: Path, work_fn, label: str):
        """Shared driver for teacher-generation stages: cache check, idx-based
        resume, thread-pool execution, atomic partial -> final rename.

        work_fn(messages) -> conversation list or None (dropped, retried on rerun).
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        partial = final.with_suffix(".partial")
        if self._cached(final, label):
            return
        if self.force:
            final.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)

        dataset = self._require_train_data(label)
        done = self._resume_done_idxs(partial)
        work = [(idx, dataset[idx].get("messages", [])) for idx in range(len(dataset)) if idx not in done]
        workers = self.config.models.teacher_concurrency
        self.logger.info(f"{label}: {len(done)} cached, {len(work)} to go, concurrency={workers}")

        if not work and not partial.exists():
            partial.touch()

        lock = threading.Lock()
        completed = 0
        started_at = time.time()
        with open(partial, "a") as pf, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work_fn, msgs): idx for idx, msgs in work if msgs}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    conv = fut.result()
                except Exception as e:
                    self.logger.warning(f"{label} item {idx} failed: {e}")
                    continue
                if not conv:
                    continue
                with lock:
                    pf.write(json.dumps({"idx": idx, "messages": conv}) + "\n")
                    pf.flush()
                completed += 1
                total_done = len(done) + completed
                self.logger.set_progress((total_done / len(dataset)) * 100, f"{label} {total_done}/{len(dataset)}")
                if completed == 1 or completed % 10 == 0 or total_done == len(dataset):
                    elapsed = time.time() - started_at
                    rate = completed / max(elapsed, 1e-9) * 60
                    left = (len(work) - completed) / max(rate, 1e-9)
                    pct = total_done * 100 // len(dataset)
                    self.logger.info(f"{label}: {total_done}/{len(dataset)} ({pct}%) — {rate:.1f}/min, ~{left:.0f} min left")

        partial.rename(final)
        self.logger.success(f"{label}: {len(done) + completed} → {final}")

    # ── Generate (SFT, single-turn teacher completions) ──────────────────────

    def _stage_generate(self):
        self.logger.set_stage(Stage.GENERATE)
        final = self.paths["teacher_singleturn"]
        if self._cached(final, "Teacher completions"):
            return

        # Free reuse: reopd trajectories for the same curated data + teacher
        # already contain a teacher completion per turn. The first turn of each
        # trajectory is exactly the single-turn completion this stage would
        # otherwise pay an API call for.
        traj = self.paths["teacher_trajectories"]
        if traj.exists():
            self.logger.info(f"Deriving single-turn completions from {traj.name} (no API calls)")
            n = 0
            with open(traj) as f, open(final, "w") as out:
                for line in f:
                    msgs = json.loads(line).get("messages", [])
                    user = next((m for m in msgs if m.get("role") == "user"), None)
                    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
                    if user and asst:
                        out.write(json.dumps({"messages": [user, asst]}) + "\n")
                        n += 1
            self.logger.success(f"Derived {n} single-turn completions → {final}")
            return

        def single_turn(messages):
            prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), None)
            if not prompt:
                return None
            teacher_text = self._teacher_complete([{"role": "user", "content": prompt}], max_tokens=1024)
            if teacher_text is None:
                return None
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": teacher_text},
            ]

        self._run_teacher_stage(self.paths["teacher_singleturn"], single_turn, "generate")
        self.metrics["teacher_singleturn"] = str(self.paths["teacher_singleturn"])

    # ── Collect (ReOPD, multi-turn teacher trajectories) ─────────────────────

    def _stage_collect(self):
        self.logger.set_stage(Stage.COLLECT)
        max_tokens = self.config.reopd.max_teacher_tokens

        def trajectory(messages):
            # Turns within a conversation are sequential (context builds);
            # conversations run in parallel via the stage driver.
            conversation = []
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                prompt = msg.get("content", "")
                if not prompt:
                    continue
                teacher_text = self._teacher_complete(conversation + [{"role": "user", "content": prompt}], max_tokens)
                if teacher_text is None:
                    return None
                conversation.append({"role": "user", "content": prompt})
                conversation.append({"role": "assistant", "content": teacher_text})
            return conversation or None

        self._run_teacher_stage(self.paths["teacher_trajectories"], trajectory, "collect")
        self.metrics["teacher_trajectories"] = str(self.paths["teacher_trajectories"])

    # ── Weight (ReOPD, per-turn decay + weighted resampling) ──────────────────

    def _stage_weight(self):
        self.logger.set_stage(Stage.WEIGHT)
        p = self.paths
        if self._cached(p["reopd_sampled"], "Weighted ReOPD dataset"):
            return
        if not p["teacher_trajectories"].exists():
            raise FileNotFoundError(f"weight needs {p['teacher_trajectories']} — run collect first")

        kappa = self.config.reopd.kappa
        self.logger.info(f"Building decay-weighted examples (kappa={kappa})...")

        weighted_examples = []
        with open(p["teacher_trajectories"]) as f:
            for line in f:
                messages = json.loads(line).get("messages", [])
                assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
                for turn_idx, msg_idx in enumerate(assistant_indices):
                    weighted_examples.append({
                        "messages": messages[:msg_idx + 1],
                        "weight": kappa ** turn_idx,
                    })

        with open(p["reopd_weighted"], "w") as f:
            for item in weighted_examples:
                f.write(json.dumps(item) + "\n")

        # Replay via weighted sampling: high-weight (early) turns are duplicated,
        # low-weight (late) turns may drop out. Approximates a weighted loss.
        import random
        random.seed(self.config.training.hyperparams.seed)
        weights = [ex["weight"] for ex in weighted_examples]
        sampled_indices = random.choices(range(len(weighted_examples)), weights=weights, k=len(weighted_examples))

        with open(p["reopd_sampled"], "w") as f:
            for i in sampled_indices:
                f.write(json.dumps(weighted_examples[i]) + "\n")

        self.logger.success(f"Weighted pool: {len(weighted_examples)} → sampled {len(sampled_indices)} → {p['reopd_sampled']}")
        self.metrics["reopd_weighted_pool"] = len(weighted_examples)

    # ── Train ─────────────────────────────────────────────────────────────────

    def _stage_train(self):
        self.logger.set_stage(Stage.TRAIN)
        self._training_start_time = datetime.now(timezone.utc)
        p = self.paths
        adapter_dir = p["adapter_dir"]
        done_marker = p["adapter_done"]

        if self._cached(done_marker, "Trained adapter"):
            self.adapter_path = str(adapter_dir)
            return
        if self.force and adapter_dir.exists():
            shutil.rmtree(adapter_dir)

        dataset_path = p["train_dataset"]
        if not dataset_path.exists():
            producer = {"sft": "generate (or curate)", "reopd": "weight"}.get(self.method, "curate")
            raise FileNotFoundError(f"train needs {dataset_path} — run {producer} first")

        resume_ckpt = self._latest_checkpoint(adapter_dir)
        gpu_available, gpu_info, free_vram = self._check_gpu()
        if not gpu_available:
            raise RuntimeError(f"No GPU: {gpu_info}")
        self.logger.info(f"GPU: {gpu_info}")

        max_seq_length = self.config.training.hyperparams.max_seq_length
        if free_vram < 5.0 and max_seq_length > 1024:
            max_seq_length = 1024
            self.logger.warning(f"Low VRAM ({free_vram:.1f} GB) — max_seq_length reduced to {max_seq_length}")

        if self.method in PREFERENCE_METHODS:
            self._train_preference(adapter_dir, done_marker, dataset_path, resume_ckpt, max_seq_length, free_vram)
        else:
            self._train_sft(adapter_dir, done_marker, dataset_path, resume_ckpt, max_seq_length)

    def _load_model(self, resume_ckpt, max_seq_length):
        """Model for training: warm start from --from-adapter, else base + LoRA.
        (Checkpoint resume restores weights via the trainer, not here.)"""
        from unsloth import FastLanguageModel

        cfg = self.config.training
        h = cfg.hyperparams
        load_in_4bit = (cfg.quantization == "4bit")

        if self.from_adapter and not resume_ckpt:
            self.logger.info(f"Warm start from adapter: {self.from_adapter}")
            return FastLanguageModel.from_pretrained(
                model_name=self.from_adapter,
                max_seq_length=max_seq_length, dtype=None, load_in_4bit=load_in_4bit,
            )

        if resume_ckpt:
            self.logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
        else:
            self.logger.info(f"Loading base model: {self.config.models.student}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.models.student,
            max_seq_length=max_seq_length, dtype=None, load_in_4bit=load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=cfg.lora.rank, target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            bias="none", use_gradient_checkpointing="unsloth", random_state=h.seed,
        )
        return model, tokenizer

    def _common_training_args(self) -> Dict[str, Any]:
        cfg = self.config.training
        h = cfg.hyperparams
        from unsloth import is_bfloat16_supported
        chk = cfg.checkpoint
        return {
            "per_device_train_batch_size": h.batch_size,
            "gradient_accumulation_steps": h.gradient_accumulation_steps,
            "warmup_steps": h.warmup_steps,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "logging_steps": self.config.logging.metrics.log_every_n_steps,
            "optim": "adamw_8bit",
            "weight_decay": h.weight_decay,
            "seed": h.seed,
            "report_to": "none",
            "dataloader_num_workers": 0,
            "save_strategy": "steps" if chk.enabled else "no",
            "save_steps": chk.steps,
            "save_total_limit": chk.keep_last,
        }

    def _train_sft(self, adapter_dir, done_marker, dataset_path, resume_ckpt, max_seq_length):
        """SFT for methods sft and reopd (reopd differs only in data + lr/epochs)."""
        import torch
        # unsloth must import before trl/transformers — it patches SFTTrainer
        import unsloth  # noqa: F401
        from datasets import load_dataset
        from transformers import EarlyStoppingCallback
        from trl import SFTTrainer, SFTConfig

        cfg = self.config.training
        h = cfg.hyperparams
        model, tokenizer = self._load_model(resume_ckpt, max_seq_length)

        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        self.logger.info(f"Training data: {len(dataset)} examples ({dataset_path.name})")

        def format_chatml(examples):
            return {"text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in examples["messages"]
            ]}

        dataset = dataset.map(format_chatml, batched=True, num_proc=1)

        if self.method == "reopd":
            epochs = self.config.reopd.epochs
            lr = h.learning_rate * self.config.reopd.lr_scale
            scheduler = "constant"
        else:
            epochs = h.epochs
            lr = h.learning_rate
            scheduler = h.lr_scheduler

        # Early stopping needs eval loss: carve a small split off the train set
        es = cfg.early_stopping
        eval_dataset = None
        if es.enabled and len(dataset) >= 50:
            n_eval = max(8, int(len(dataset) * es.eval_fraction))
            split = dataset.train_test_split(test_size=n_eval, seed=h.seed)
            dataset, eval_dataset = split["train"], split["test"]
            self.logger.info(f"Early stopping: eval split={len(eval_dataset)} examples")

        adapter_dir.mkdir(parents=True, exist_ok=True)
        steps_per_epoch = max(1, len(dataset) // (h.batch_size * h.gradient_accumulation_steps))
        total_steps = steps_per_epoch * epochs
        self.logger.info(f"Steps: ~{total_steps} ({steps_per_epoch}/epoch x {epochs}), lr={lr:.2e}, scheduler={scheduler}")

        args_dict = {
            **self._common_training_args(),
            "num_train_epochs": epochs,
            "learning_rate": lr,
            "lr_scheduler_type": scheduler,
            "output_dir": str(adapter_dir),
            "eval_strategy": "steps" if eval_dataset is not None else "no",
            "eval_steps": cfg.checkpoint.steps if eval_dataset is not None else None,
        }

        callbacks = [_make_log_callback(self.logger, self.method, total_steps)]
        if eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=es.patience,
                early_stopping_threshold=es.min_delta,
            ))
            self.logger.info(f"Early stopping: patience={es.patience} evals, min_delta={es.min_delta}")

        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(
                **args_dict,
                dataset_text_field="text",
                max_length=max_seq_length,
                dataset_num_proc=1,
                packing=False,
            ),
            callbacks=callbacks,
        )
        self.logger.info("Training started")
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        self._finalize_training(model, tokenizer, trainer, adapter_dir, done_marker, len(dataset), total_steps)

    def _train_preference(self, adapter_dir, done_marker, dataset_path, resume_ckpt, max_seq_length, free_vram):
        """DPO / ORPO / KTO on chosen/rejected pairs (TRL 0.24)."""
        import unsloth  # noqa: F401  (must import before trl/transformers)
        from datasets import load_dataset, Dataset

        cfg = self.config.training
        h = cfg.hyperparams
        pref = self.config.preference
        model, tokenizer = self._load_model(resume_ckpt, max_seq_length)

        raw = load_dataset("json", data_files=str(dataset_path), split="train")
        self.logger.info(f"Preference pairs: {len(raw)} ({dataset_path.name})")

        if self.method == "kto":
            # KTO wants {prompt, completion, label} — one row per side of the pair
            rows = []
            for ex in raw:
                prompt = ex["chosen"][:-1]
                rows.append({"prompt": prompt, "completion": ex["chosen"][-1:], "label": True})
                rows.append({"prompt": prompt, "completion": ex["rejected"][-1:], "label": False})
            ds = Dataset.from_list(rows)
        else:
            def to_preference(ex):
                return {
                    "prompt": ex["chosen"][:-1],
                    "chosen": ex["chosen"][-1:],
                    "rejected": ex["rejected"][-1:],
                }
            ds = raw.map(to_preference)

        adapter_dir.mkdir(parents=True, exist_ok=True)
        lr = h.learning_rate * pref.lr_scale

        # Preference losses materialize fp32 logits for chosen+rejected
        # (~batch x seq x vocab x 4 bytes) — relieve memory pressure on small
        # GPUs. KTO needs batch_size > 1 (KL term), so clamp seq length instead.
        batch_size = h.batch_size
        grad_accum = h.gradient_accumulation_steps
        if free_vram < 7.0:
            if self.method == "kto":
                if max_seq_length > 512:
                    max_seq_length = 512
                    self.logger.warning(f"Low VRAM ({free_vram:.1f} GB) — KTO max_seq_length reduced to 512")
            elif batch_size > 1:
                grad_accum *= batch_size
                batch_size = 1
                self.logger.warning(
                    f"Low VRAM ({free_vram:.1f} GB) — batch_size 1 x{grad_accum} accum "
                    f"(preference losses need ~2x logits memory of SFT)"
                )

        steps_per_epoch = max(1, len(ds) // (batch_size * grad_accum))
        total_steps = steps_per_epoch * pref.epochs
        self.logger.info(f"Method: {self.method}, beta={pref.beta}, lr={lr:.2e}, steps: ~{total_steps}")

        args_dict = {
            **self._common_training_args(),
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "num_train_epochs": pref.epochs,
            "learning_rate": lr,
            "lr_scheduler_type": h.lr_scheduler,
            "output_dir": str(adapter_dir),
            "beta": pref.beta,
            "max_prompt_length": pref.max_prompt_length,
            "max_length": max_seq_length,
        }
        callbacks = [_make_log_callback(self.logger, self.method, total_steps)]

        if self.method == "dpo":
            DPOTrainer, DPOConfig = _import_dpo_trainer()
            # TRL writes model.warnings_issued[...] — unsloth's patched models
            # don't have it (transformers sets it in PreTrainedModel.__init__)
            model.warnings_issued = {}
            trainer = DPOTrainer(
                model=model,
                args=DPOConfig(**args_dict, precompute_ref_log_probs=pref.precompute_ref_log_probs),
                train_dataset=ds,
                processing_class=tokenizer,
                callbacks=callbacks,
            )
        elif self.method == "orpo":
            from trl import ORPOTrainer, ORPOConfig
            trainer = ORPOTrainer(
                model=model,
                args=ORPOConfig(**args_dict),
                train_dataset=ds,
                processing_class=tokenizer,
                callbacks=callbacks,
            )
        elif self.method == "kto":
            from trl import KTOTrainer, KTOConfig
            trainer = KTOTrainer(
                model=model,
                args=KTOConfig(**args_dict),
                train_dataset=ds,
                processing_class=tokenizer,
                callbacks=callbacks,
            )
        else:
            raise ValueError(f"Unknown preference method: {self.method}")

        self.logger.info("Training started")
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        self._finalize_training(model, tokenizer, trainer, adapter_dir, done_marker, len(ds), total_steps)

    def _finalize_training(self, model, tokenizer, trainer, adapter_dir, done_marker, n_examples, total_steps):
        """Metrics, final save, done marker, GGUF export, GPU cleanup."""
        import torch

        elapsed = datetime.now(timezone.utc) - self._training_start_time
        self.metrics["training_duration_sec"] = elapsed.total_seconds()
        self.metrics["completed_steps"] = trainer.state.global_step
        self.metrics["training_examples"] = n_examples
        self.metrics["total_steps"] = total_steps
        if trainer.state.log_history:
            self.metrics["final_loss"] = trainer.state.log_history[-1].get("loss", 0)

        self.logger.info("Saving final adapter...")
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        with open(done_marker, "w") as f:
            json.dump({
                "method": self.method,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "from_adapter": self.from_adapter,
                "metrics": {k: v for k, v in self.metrics.items()},
            }, f, indent=2, default=str)
        self.adapter_path = str(adapter_dir)
        self.logger.success(f"Adapter saved: {adapter_dir}")

        if self.config.deploy.gguf.enabled:
            self.logger.info("Exporting GGUF...")
            gguf_path = export_gguf(model, tokenizer, str(adapter_dir),
                                    self.config.deploy.gguf.quantization, self.logger)
            if gguf_path:
                self.metrics["gguf_path"] = gguf_path

        del trainer, model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _latest_checkpoint(self, adapter_dir: Path) -> Optional[Path]:
        """Latest resumable HF checkpoint in the keyed adapter dir, if training
        was interrupted (checkpoints present, no completion marker)."""
        if not adapter_dir.exists() or (adapter_dir / "training_done.json").exists():
            return None
        checkpoints = sorted(
            adapter_dir.glob("checkpoint-*"),
            key=lambda c: int(c.name.split("-")[1]),
        )
        for ckpt in reversed(checkpoints):
            if (ckpt / "trainer_state.json").exists():
                return ckpt
        return None

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def _stage_evaluate(self):
        self.logger.set_stage(Stage.EVALUATE)
        holdout = self.paths["holdout"]
        if not holdout.exists():
            self.logger.warning(f"No holdout: {holdout} — run curate first, skipping evaluation")
            return
        adapter = self.adapter_path or str(self.paths["adapter_dir"])
        if not (Path(adapter) / "adapter_config.json").exists():
            self.logger.warning(f"No adapter at {adapter}, skipping evaluation")
            return

        self.logger.info(f"Evaluating: {adapter}")
        import torch
        from unsloth import FastLanguageModel
        from lib.evaluate import evaluate_model

        max_seq_length = self.config.training.hyperparams.max_seq_length
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter, max_seq_length=max_seq_length, dtype=None,
                load_in_4bit=(self.config.training.quantization == "4bit"),
            )
        except ValueError as e:
            if "dispatched on the CPU or the disk" in str(e):
                self.logger.warning("OOM loading eval model, retrying with max_seq_length=512...")
                max_seq_length = 512
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=adapter, max_seq_length=max_seq_length, dtype=None,
                    load_in_4bit=(self.config.training.quantization == "4bit"),
                )
            else:
                raise
        FastLanguageModel.for_inference(model)

        judge = self.config.training.judge
        use_llm = judge.mode == "llm"
        self.logger.info(f"Judge: {'LLM (' + judge.llm_model + ')' if use_llm else 'heuristic'}, target: {judge.win_rate_target*100:.0f}%")

        eval_results = evaluate_model(
            model=model, tokenizer=tokenizer,
            holdout_path=str(holdout),
            use_llm_judge=use_llm,
            judge_model=judge.llm_model,
            max_examples=judge.max_examples,
            logger=self.logger,
        )

        self.metrics["eval_total"] = eval_results["total"]
        self.metrics["eval_student_wins"] = eval_results["wins"]["student"]
        self.metrics["eval_chosen_wins"] = eval_results["wins"]["chosen"]
        self.metrics["eval_ties"] = eval_results["wins"]["tie"]
        self.metrics["eval_student_win_rate"] = eval_results["student_win_rate"]

        win_pct = eval_results["student_win_rate"] * 100
        if eval_results["student_win_rate"] >= judge.win_rate_target:
            self.logger.success(f"GATE CLEARED: {win_pct:.1f}% ≥ {judge.win_rate_target*100:.0f}%")
        else:
            self.logger.warning(f"Gate not cleared: {win_pct:.1f}% < {judge.win_rate_target*100:.0f}%")

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def _stage_benchmark(self):
        self.logger.set_stage(Stage.BENCHMARK)
        adapter = self.adapter_path or str(self.paths["adapter_dir"])
        if not (Path(adapter) / "adapter_config.json").exists():
            self.logger.warning(f"No adapter at {adapter}, skipping benchmark")
            return
        self.logger.info(f"Benchmarking: {adapter}")

        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM

            device = self.config.benchmark.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model = HFLM(
                pretrained=self.config.models.student,
                peft=adapter,
                batch_size="auto",
                trust_remote_code=True,
                device=device,
            )
            results = simple_evaluate(
                model=model,
                tasks=self.config.benchmark.tasks,
                limit=self.config.benchmark.limit,
                batch_size="auto",
            )

            benchmark_results = {}
            for task, info in results.get("results", {}).items():
                score = (
                    info.get("exact_match,strict-match") or
                    info.get("acc,none") or
                    info.get("acc_norm,none")
                )
                if score is not None:
                    benchmark_results[task] = float(score)
                    self.logger.log_metric(f"benchmark_{task}", score)
                    self.logger.success(f"{task}: {score:.4f}")
            self.metrics["benchmark"] = benchmark_results

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            self.metrics["benchmark_error"] = str(e)

    # ── Deploy ────────────────────────────────────────────────────────────────

    def _stage_deploy(self):
        self.logger.set_stage(Stage.DEPLOY)
        adapter_dir = Path(self.adapter_path or str(self.paths["adapter_dir"]))
        if not (adapter_dir / "adapter_config.json").exists():
            self.logger.warning(f"No adapter at {adapter_dir}, skipping deployment")
            return

        gguf_dir = adapter_dir / "gguf"
        has_gguf = gguf_dir.exists() and any(gguf_dir.rglob("*.gguf"))

        if self.config.deploy.gguf.enabled and not has_gguf:
            self.logger.info("No GGUF in adapter — loading model to export...")
            import torch
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(adapter_dir),
                max_seq_length=self.config.training.hyperparams.max_seq_length,
                dtype=None,
                load_in_4bit=(self.config.training.quantization == "4bit"),
            )
            export_gguf(model, tokenizer, str(adapter_dir),
                        self.config.deploy.gguf.quantization, self.logger)
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            has_gguf = gguf_dir.exists() and any(gguf_dir.rglob("*.gguf"))

        if not self.config.deploy.ollama.enabled:
            self.logger.info("Ollama deployment disabled")
            return
        modelfiles = list(gguf_dir.rglob("Modelfile")) if gguf_dir.exists() else []
        if not has_gguf or not modelfiles:
            self.logger.warning(f"No Modelfile under {gguf_dir}")
            return
        modelfile = modelfiles[0]
        if not check_ollama_installed():
            self.logger.warning("Ollama not installed")
            return

        model_name = self.config.deploy.ollama.model_name
        if register_ollama_model(str(modelfile), model_name, self.logger):
            self.metrics["ollama_model"] = model_name
            self.logger.success(f"Deployed: ollama run {model_name}")
        else:
            self.logger.error("Ollama registration failed")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _check_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory // (1024**3)
                free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                return True, f"{props.name} ({total_gb} GB, {free_gb:.1f} GB free)", free_gb
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return True, "Apple MPS", 8.0
            else:
                return False, "No GPU", 0
        except ImportError:
            return False, "PyTorch not installed", 0

    def _write_training_summary(self):
        summary = {
            "run_id": self.run_id,
            "status": self.logger.status.stage.value,
            "started_at": self.logger.status.started_at,
            "completed_at": self.logger.status.completed_at,
            "metrics": self.metrics,
            "final_loss": self.metrics.get("final_loss"),
            "eval_win_rate": self.metrics.get("eval_student_win_rate"),
            "training_duration_sec": self.metrics.get("training_duration_sec"),
            "adapter_path": self.adapter_path,
            "config": {
                "method": self.method,
                "student": self.config.models.student,
                "teacher": self.config.models.teacher,
                "epochs": self.config.training.hyperparams.epochs,
                "learning_rate": self.config.training.hyperparams.learning_rate,
                "lora_rank": self.config.training.lora.rank,
            },
        }
        summary_path = self.logger.run_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info(f"Training summary: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Commands
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    config = load_config(args.config, run_mode=args.mode)

    if args.method:
        config.training.method = args.method
    if args.student:
        config.models.student = args.student
    if args.teacher:
        config.models.teacher = args.teacher
    if args.max_examples:
        config.curation.max_examples = args.max_examples
    if args.epochs:
        config.training.hyperparams.epochs = args.epochs

    method = config.training.method

    from_adapter = None
    if args.from_adapter:
        if args.from_adapter == "latest":
            from_adapter = latest_adapter()
            if not from_adapter:
                print("Error: no adapter found for --from-adapter latest")
                return 1
        else:
            candidate = Path(args.from_adapter).expanduser()
            if not (candidate / "adapter_config.json").exists():
                print(f"Error: invalid adapter (no adapter_config.json): {candidate}")
                return 1
            from_adapter = str(candidate)

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else None

    run_id = str(uuid.uuid4())
    logger = create_logger(run_id, config)

    pipeline = DistillPipeline(config, logger, method, from_adapter=from_adapter, force=args.force)
    result = pipeline.run(steps=steps, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] in ("completed", "dry_run") else 1


def cmd_status(args):
    logs_dir = Path(args.logs_dir or "./logs").expanduser()
    runs_dir = logs_dir / "runs"
    if not runs_dir.exists():
        print("No runs found")
        return 0

    print("\n" + "=" * 70)
    print("  RECENT RUNS")
    print("=" * 70)

    for run_path in sorted(runs_dir.iterdir(), reverse=True)[:10]:
        status_file = run_path / "status.json"
        if status_file.exists():
            with open(status_file) as f:
                st = json.load(f)
            stage = st.get("stage", "?")
            icon = {"complete": "✅", "failed": "❌", "train": "🏋️"}.get(stage, "·")
            progress = st.get("progress", 0)
            print(f"  {icon} {run_path.name:30s} {stage:10s} {progress:5.1f}%")
        else:
            print(f"  ?  {run_path.name}")
    print()
    return 0


def cmd_logs(args):
    logs_dir = Path(args.logs_dir or "./logs").expanduser()
    runs_dir = logs_dir / "runs"
    if not runs_dir.exists():
        print("No runs found")
        return 1
    runs = sorted(runs_dir.iterdir(), reverse=True)
    if not runs:
        print("No runs")
        return 1
    log_file = runs[0] / "run.log"
    if not log_file.exists():
        print(f"No log: {log_file}")
        return 1
    import subprocess
    try:
        subprocess.run(["tail", "-f", str(log_file)])
    except KeyboardInterrupt:
        pass
    return 0


def cmd_adapters(args):
    adapters = list_adapters()
    print("\n" + "=" * 70)
    print("  TRAINED ADAPTERS")
    print("=" * 70)
    if not adapters:
        print("  No adapters found")
    else:
        for a in adapters[:15]:
            gguf = "GGUF" if a["has_gguf"] else "    "
            print(f"  [{gguf}] {a['id']:24s}  {a['created_at'][:16]}")
    print()
    return 0


def cmd_clean(args):
    removed = cleanup_old_adapters(keep=args.keep)
    print(f"Removed {removed} adapter(s), kept {args.keep} most recent")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LocalDistill - LLM Distillation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  sft    curate -> [generate] -> train -> evaluate -> benchmark -> deploy
  reopd  curate -> collect -> weight -> train -> evaluate -> benchmark -> deploy
  dpo    curate -> train -> evaluate -> benchmark -> deploy  (preference pairs)
  orpo   curate -> train -> evaluate -> benchmark -> deploy  (no ref model)
  kto    curate -> train -> evaluate -> benchmark -> deploy  (binary labels)

Examples:
  python distill.py run --method sft --mode preference
  python distill.py run --method reopd                      # from base model
  python distill.py run --method reopd --from-adapter latest  # warm start from SFT
  python distill.py run --method orpo --mode preference
  python distill.py run --method reopd --steps curate,collect
  python distill.py run --method sft --force                # redo all stages
  python distill.py status
        """
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run pipeline")
    run_p.add_argument("--config", "-c", help="Config file")
    run_p.add_argument("--method", choices=sorted(METHODS), help="Training method (default: config training.method)")
    run_p.add_argument("--mode", "-m", choices=["demo", "full", "preference", "multiturn"], help="Preset to apply")
    run_p.add_argument("--from-adapter", metavar="latest|PATH", help="Warm-start training from an existing adapter")
    run_p.add_argument("--force", action="store_true", help="Regenerate all stage outputs (ignore cache)")
    run_p.add_argument("--steps", help="Comma-separated subset of the method's stages")
    run_p.add_argument("--student", help="Student model")
    run_p.add_argument("--teacher", help="Teacher model")
    run_p.add_argument("--max-examples", type=int)
    run_p.add_argument("--epochs", type=int)
    run_p.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="Show run status")
    sub.add_parser("logs", help="Tail latest logs")
    sub.add_parser("adapters", help="List trained adapters")

    clean_p = sub.add_parser("clean", help="Delete old adapters, keeping the most recent N")
    clean_p.add_argument("--keep", type=int, default=5)

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "logs":
        sys.exit(cmd_logs(args))
    elif args.command == "adapters":
        sys.exit(cmd_adapters(args))
    elif args.command == "clean":
        sys.exit(cmd_clean(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
