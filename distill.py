#!/usr/bin/env python3
"""
LocalDistill - Main Orchestrator

Central CLI for the distillation pipeline. V2: coherent, configurable,
with proper progress tracking, early stopping, and strategy selection.

Strategies (config.training.strategy.name):
  sft      - Supervised fine-tuning on chosen responses
  dpo      - Direct preference optimization (uses rejected pairs)
  on_policy - ReOPD: collect teacher trajectories, replay with decay

Usage:
    python distill.py run --mode preference
    python distill.py run --mode preference --steps curate,train,evaluate
    python distill.py status
    python distill.py logs
"""

import os
import sys
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
sys.path.insert(0, str(Path(__file__).parent))

from lib.config import load_config, save_config, print_config, Config, validate_config, _apply_preset
from lib.logger import DistillLogger, create_logger, PipelineStage as Stage
from lib.dataset import DatasetLoader, load_dataset_from_config
from lib.deploy import (
    export_gguf, register_ollama_model, list_adapters,
    get_latest_adapter, deploy_adapter, check_ollama_installed,
    cleanup_old_adapters
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class DistillPipeline:
    """Orchestrates: curate → train → evaluate → on_policy → benchmark → deploy"""

    def __init__(self, config: Config, logger: DistillLogger):
        self.config = config
        self.logger = logger
        self.run_id = logger.run_id
        self.adapter_path: Optional[str] = None
        self.dataset_path: Optional[str] = None
        self.holdout_path: Optional[str] = None
        self.resume_checkpoint: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self._training_start_time: Optional[datetime] = None

    def run(self, steps: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Any]:
        all_steps = ["curate", "train", "evaluate", "on_policy", "benchmark", "deploy"]
        steps = steps or all_steps

        # Filter based on config
        if not self.config.on_policy.enabled and self.config.training.strategy.name != "on_policy":
            steps = [s for s in steps if s != "on_policy"]
        if not self.config.benchmark.enabled:
            steps = [s for s in steps if s != "benchmark"]
        if not self.config.deploy.gguf.enabled and not self.config.deploy.ollama.enabled:
            steps = [s for s in steps if s != "deploy"]
        if self.config.dataset.source != "preference":
            steps = [s for s in steps if s != "evaluate"]

        self.logger.header("LOCALDISTILL PIPELINE")

        # Validate config before running
        errs = validate_config(self.config)
        if errs:
            for e in errs:
                self.logger.error(f"Config error: {e}")
            return {"run_id": self.run_id, "status": "failed", "error": "; ".join(errs)}

        print_config(self.config)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Steps: {' -> '.join(steps)}")

        if dry_run:
            self.logger.info("DRY RUN - No changes")
            return {"run_id": self.run_id, "status": "dry_run", "steps": steps}

        # Save frozen config
        config_snapshot = self.logger.run_dir / "config.yaml"
        save_config(self.config, str(config_snapshot))

        # Cleanup old adapters
        removed = cleanup_old_adapters(keep=3, logger=self.logger)
        if removed:
            self.logger.info(f"Cleaned up {removed} old adapter(s)")

        try:
            for step in steps:
                if step == "curate":
                    self._run_curate()
                elif step == "train":
                    self._run_train()
                elif step == "evaluate":
                    self._run_evaluate()
                elif step == "on_policy":
                    self._run_on_policy()
                elif step == "benchmark":
                    self._run_benchmark()
                elif step == "deploy":
                    self._run_deploy()

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
            return {
                "run_id": self.run_id,
                "status": "failed",
                "error": str(e),
            }

    # ── Curate ─────────────────────────────────────────────────────────────────

    def _run_curate(self):
        self.logger.set_stage(Stage.CURATE)
        loader = DatasetLoader(self.config, self.logger)
        base_path = Path(self.config.logging.dir).parent
        base_path.mkdir(parents=True, exist_ok=True)

        if self.config.dataset.source == "preference":
            self.logger.info("Loading preference dataset...")
            from lib.dataset import split_dataset

            pairs = loader.load_preference_pairs()
            self.logger.info(f"Loaded {len(pairs)} preference pairs")

            holdout_ratio = self.config.dataset.holdout_ratio
            train_pairs, holdout_pairs = split_dataset(
                pairs, train_ratio=1.0 - holdout_ratio,
                seed=self.config.training.hyperparams.seed,
            )
            self.logger.info(f"Split: {len(train_pairs)} train, {len(holdout_pairs)} holdout")

            train_convs = [p.to_chosen_conversation() for p in train_pairs]
            train_path = base_path / "train.jsonl"
            count = loader.export(train_convs, str(train_path), "chatml")

            holdout_path = base_path / "holdout.jsonl"
            with open(holdout_path, "w") as f:
                for p in holdout_pairs:
                    f.write(json.dumps({
                        "prompt": p.prompt,
                        "chosen": p.chosen,
                        "rejected": p.rejected,
                        "score_chosen": p.score_chosen,
                        "score_rejected": p.score_rejected,
                    }) + "\n")

            self.dataset_path = str(train_path)
            self.holdout_path = str(holdout_path)
            self.metrics["train_examples"] = len(train_pairs)
            self.metrics["holdout_examples"] = len(holdout_pairs)

            self.logger.success(f"Train: {count} examples → {train_path}")
            self.logger.success(f"Holdout: {len(holdout_pairs)} pairs → {holdout_path}")
        else:
            self.logger.info("Loading dataset...")
            conversations = loader.load()
            self.logger.info(f"Loaded {len(conversations)} raw conversations")

            self.logger.info("Applying curation filters...")
            curated = loader.curate(conversations)
            self.logger.info(f"Curated to {len(curated)} conversations")

            output_path = base_path / "train.jsonl"
            count = loader.export(curated, str(output_path), "chatml")
            self.dataset_path = str(output_path)
            self.metrics["curated_examples"] = count
            self.logger.success(f"Dataset ready: {count} examples → {output_path}")

    # ── Train ──────────────────────────────────────────────────────────────────

    def _resolve_train_dataset(self) -> Optional[str]:
        """Determine which dataset file to train on based on strategy config."""
        base_path = Path(self.config.logging.dir).parent
        strategy = self.config.training.strategy
        source = strategy.dataset_source

        # Sources in priority order
        curated_path = base_path / "train.jsonl"
        teacher_pool_path = base_path / "teacher_pool.jsonl"
        on_policy_path = base_path / "on_policy_weighted.jsonl"

        if source == "auto":
            # Auto: teacher_pool only if strategy is on_policy AND file exists
            if strategy.name == "on_policy" and teacher_pool_path.exists():
                self.logger.info(f"Auto-selected dataset: teacher_pool ({teacher_pool_path})")
                return str(teacher_pool_path)
            if curated_path.exists():
                self.logger.info(f"Auto-selected dataset: curated ({curated_path})")
                return str(curated_path)
        elif source == "teacher_pool":
            if teacher_pool_path.exists():
                return str(teacher_pool_path)
            raise FileNotFoundError(f"Dataset source='teacher_pool' but {teacher_pool_path} not found")
        elif source == "on_policy_weighted":
            if on_policy_path.exists():
                return str(on_policy_path)
            raise FileNotFoundError(f"Dataset source='on_policy_weighted' but {on_policy_path} not found")
        elif source == "curated":
            if curated_path.exists():
                return str(curated_path)
            raise FileNotFoundError(f"Dataset source='curated' but {curated_path} not found")

        # Fallback: whatever exists
        for p in [curated_path, teacher_pool_path, on_policy_path]:
            if p.exists():
                return str(p)
        return None

    def _resolve_resume_checkpoint(self, adapter_dir: Path) -> Optional[str]:
        """Find a valid checkpoint to resume from."""
        if self.resume_checkpoint:
            cp = Path(self.resume_checkpoint)
            if (cp / "adapter_config.json").exists():
                return str(cp)
            self.logger.warning(f"Invalid checkpoint: {cp} (no adapter_config.json)")

        # Auto-detect latest checkpoint in adapter_dir
        checkpoints = sorted(
            adapter_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        if checkpoints:
            latest = checkpoints[-1]
            if (latest / "adapter_config.json").exists():
                self.logger.info(f"Auto-resume from checkpoint: {latest}")
                return str(latest)
            self.logger.warning(f"Latest checkpoint missing adapter_config.json: {latest}")
        return None

    def _run_train(self):
        """Main training dispatcher. Routes to SFT, DPO, or on-policy based on strategy."""
        self.logger.set_stage(Stage.TRAIN)
        self._training_start_time = datetime.now(timezone.utc)

        strategy = self.config.training.strategy.name

        if strategy == "sft":
            self._train_sft()
        elif strategy == "dpo":
            self._train_dpo()
        elif strategy == "on_policy":
            # On-policy is a two-phase process; _run_on_policy handles it
            self.logger.info("Strategy=on_policy — use --steps on_policy or run full pipeline")
            self._train_sft()  # Fall back to SFT if called directly
        else:
            raise ValueError(f"Unknown training strategy: {strategy}")

    def _train_sft(self):
        """LoRA SFT with Unsloth. Includes progress tracking, checkpointing, early stopping."""
        dataset_path = self._resolve_train_dataset()
        if not dataset_path:
            raise FileNotFoundError("No training dataset resolved")

        gpu_available, gpu_info, free_vram = self._check_gpu()
        if not gpu_available:
            raise RuntimeError(f"No GPU: {gpu_info}")
        self.logger.info(f"GPU: {gpu_info}")

        # ── Imports ──
        import torch
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import load_dataset
        from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback
        from trl import SFTTrainer

        cfg = self.config.training
        h = cfg.hyperparams
        model_name = self.config.models.student
        max_seq_length = h.max_seq_length

        # ── VRAM adjustment ──
        if free_vram < 5.0:
            self.logger.warning(f"Low VRAM ({free_vram:.1f} GB free)")
            if max_seq_length > 1024:
                max_seq_length = 1024
                self.logger.info(f"Reduced max_seq_length to {max_seq_length}")
            if free_vram < 3.0:
                self.logger.warning("Very low VRAM — consider 1B model")

        # ── Load model ──
        self.logger.info(f"Loading {model_name}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=(cfg.quantization == "4bit"),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.rank,
            target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=h.seed,
        )
        self.logger.info(f"LoRA: rank={cfg.lora.rank}, alpha={cfg.lora.alpha}")

        # ── Dataset ──
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        self.logger.info(f"Dataset: {len(dataset)} examples")

        def format_chatml(examples):
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return {"text": texts}

        dataset = dataset.map(format_chatml, batched=True, num_proc=1)

        # ── Adapter dir ──
        adapter_dir = Path("~/localdistill/adapters").expanduser() / self.run_id[:8]
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # ── Calculate total steps for progress ──
        steps_per_epoch = max(1, len(dataset) // (h.batch_size * h.gradient_accumulation_steps))
        total_steps = steps_per_epoch * h.epochs
        self.logger.info(f"Estimated steps: {total_steps} ({steps_per_epoch}/epoch × {h.epochs})")

        # ── Training args ──
        training_args = TrainingArguments(
            per_device_train_batch_size=h.batch_size,
            gradient_accumulation_steps=h.gradient_accumulation_steps,
            warmup_steps=h.warmup_steps,
            num_train_epochs=h.epochs,
            learning_rate=h.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging.metrics.log_every_n_steps,
            optim="adamw_8bit",
            weight_decay=h.weight_decay,
            lr_scheduler_type=h.lr_scheduler,
            seed=h.seed,
            output_dir=str(adapter_dir),
            report_to="none",
            dataloader_num_workers=0,
            save_strategy="no",  # We handle checkpointing manually
        )

        # ── Callbacks ──
        class DistillCallback(TrainerCallback):
            def __init__(self, pipeline: "DistillPipeline", total_steps: int):
                self.pipeline = pipeline
                self.total_steps = total_steps

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    step = state.global_step
                    loss = logs.get("loss", 0)
                    lr = logs.get("learning_rate", 0)
                    self.pipeline.logger.log_training_step(
                        step, loss, lr, total_steps=self.total_steps
                    )

            def on_step_end(self, args, state, control, **kwargs):
                # Manual checkpointing (avoids HF save_strategy pickle issues)
                chk = self.pipeline.config.training.checkpoint
                if chk.enabled and state.global_step > 0 and state.global_step % chk.steps == 0:
                    ckpt_dir = adapter_dir / f"checkpoint-{state.global_step}"
                    model.save_pretrained(str(ckpt_dir))
                    self.pipeline.logger.info(f"Checkpoint saved: {state.global_step}")
                    # Cleanup old
                    checkpoints = sorted(
                        adapter_dir.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[1])
                    )
                    for old in checkpoints[:-chk.keep_last]:
                        shutil.rmtree(old, ignore_errors=True)

        callbacks = [DistillCallback(self, total_steps)]

        # Early stopping
        es = cfg.early_stopping
        if es.enabled:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=es.patience,
                early_stopping_threshold=es.min_delta,
            ))
            self.logger.info(f"Early stopping: patience={es.patience}, min_delta={es.min_delta}")

        # ── Train ──
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
            callbacks=callbacks,
        )

        resume_from = self._resolve_resume_checkpoint(adapter_dir)
        if resume_from:
            self.logger.info(f"Resuming from: {resume_from}")

        self.logger.info("Training started")
        self.metrics["training_start"] = datetime.now(timezone.utc).isoformat()
        trainer.train(resume_from_checkpoint=resume_from)

        elapsed = datetime.now(timezone.utc) - self._training_start_time
        self.metrics["training_duration_sec"] = elapsed.total_seconds()

        # ── Save final adapter ──
        if cfg.checkpoint.save_final:
            self.logger.info("Saving final adapter...")
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            self.adapter_path = str(adapter_dir)
            self.logger.success(f"Adapter saved: {adapter_dir}")

        # ── GGUF export ──
        if self.config.deploy.gguf.enabled:
            self.logger.info("Exporting GGUF...")
            gguf_path = export_gguf(
                model, tokenizer, str(adapter_dir),
                self.config.deploy.gguf.quantization, self.logger,
            )
            if gguf_path:
                self.metrics["gguf_path"] = gguf_path

        # ── Metrics ──
        self.metrics["adapter_path"] = str(adapter_dir)
        self.metrics["training_examples"] = len(dataset)
        self.metrics["total_steps"] = total_steps
        self.metrics["completed_steps"] = trainer.state.global_step
        self.metrics["final_loss"] = trainer.state.log_history[-1].get("loss", 0) if trainer.state.log_history else 0

    def _train_dpo(self):
        """DPO training using preference pairs. Stub — full implementation TBD."""
        self.logger.info("DPO strategy selected")
        if self.config.dataset.source != "preference":
            raise ValueError("DPO requires dataset.source='preference' (needs rejected pairs)")

        self.logger.warning("DPO training not yet fully implemented. Falling back to SFT.")
        self.logger.info("To implement DPO: use trl.DPOTrainer with chosen+rejected pairs.")
        # For now, fall back to SFT
        self._train_sft()

    # ── Evaluate ───────────────────────────────────────────────────────────────

    def _run_evaluate(self):
        """Evaluate student against chosen responses on holdout set."""
        self.logger.set_stage(Stage.EVALUATE)

        if not self.holdout_path:
            self.holdout_path = str(Path(self.config.logging.dir).parent / "holdout.jsonl")
        if not Path(self.holdout_path).exists():
            self.logger.warning(f"No holdout file: {self.holdout_path}, skipping evaluation")
            return
        if not self.adapter_path:
            self.adapter_path = get_latest_adapter()
        if not self.adapter_path:
            self.logger.warning("No adapter found, skipping evaluation")
            return

        self.logger.info(f"Evaluating: {self.adapter_path}")
        self.logger.info(f"Holdout: {self.holdout_path}")

        import torch
        from unsloth import FastLanguageModel
        from lib.evaluate import evaluate_model

        max_seq_length = self.config.training.hyperparams.max_seq_length
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.adapter_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=(self.config.training.quantization == "4bit"),
        )
        FastLanguageModel.for_inference(model)

        judge_cfg = self.config.training.judge if hasattr(self.config.training, 'judge') else None
        use_llm = judge_cfg.mode == "llm" if judge_cfg else False
        judge_model = judge_cfg.llm_model if judge_cfg else self.config.models.teacher
        max_examples = judge_cfg.max_examples if judge_cfg else 50
        win_target = judge_cfg.win_rate_target if judge_cfg else 0.6

        self.logger.info(f"Judge: {'LLM (' + judge_model + ')' if use_llm else 'heuristic'}")
        self.logger.info(f"Win rate target: {win_target*100:.0f}%")

        eval_results = evaluate_model(
            model=model, tokenizer=tokenizer,
            holdout_path=self.holdout_path,
            use_llm_judge=use_llm,
            judge_model=judge_model,
            max_examples=max_examples,
            logger=self.logger,
        )

        self.metrics["eval_total"] = eval_results["total"]
        self.metrics["eval_student_wins"] = eval_results["wins"]["student"]
        self.metrics["eval_chosen_wins"] = eval_results["wins"]["chosen"]
        self.metrics["eval_ties"] = eval_results["wins"]["tie"]
        self.metrics["eval_student_win_rate"] = eval_results["student_win_rate"]

        win_pct = eval_results["student_win_rate"] * 100
        if eval_results["student_win_rate"] >= win_target:
            self.logger.success(f"GATE CLEARED: {win_pct:.1f}% ≥ {win_target*100:.0f}%")
        else:
            self.logger.warning(f"Gate not cleared: {win_pct:.1f}% < {win_target*100:.0f}%")

    # ── On-Policy (ReOPD) ──────────────────────────────────────────────────────

    def _run_on_policy(self):
        """Two-phase ReOPD: collect teacher trajectories, train with decay weighting."""
        self.logger.set_stage(Stage.ON_POLICY)

        if not self.config.on_policy.enabled and self.config.training.strategy.name != "on_policy":
            self.logger.info("On-policy disabled")
            return

        import torch
        import litellm
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import load_dataset
        from transformers import TrainingArguments, TrainerCallback
        from trl import SFTTrainer

        cfg = self.config.on_policy
        teacher_model = self.config.models.teacher
        kappa = cfg.exponential_lambda if cfg.decay_function == "exponential" else (1.0 - 1.0 / cfg.teacher_query_interval)
        self.logger.info(f"ReOPD: teacher={teacher_model}, decay={cfg.decay_function}, kappa={kappa:.2f}")

        base_path = Path(self.config.logging.dir).parent
        teacher_pool_path = base_path / "teacher_pool.jsonl"
        dataset_path = self._resolve_train_dataset()

        # ── Phase 1: Collect teacher responses ──
        if teacher_pool_path.exists():
            self.logger.info(f"Phase 1: Using existing teacher pool ({teacher_pool_path})")
        else:
            self.logger.info("Phase 1: Collecting teacher trajectories...")
            if not dataset_path:
                raise FileNotFoundError("No dataset for on-policy collection")

            dataset = load_dataset("json", data_files=dataset_path, split="train")
            partial_path = Path(str(teacher_pool_path) + ".partial")
            teacher_pool = []
            start_idx = 0

            if partial_path.exists():
                with open(partial_path) as f:
                    for line in f:
                        teacher_pool.append(json.loads(line))
                start_idx = len(teacher_pool)
                self.logger.info(f"Resuming from {start_idx} conversations")

            for idx, example in enumerate(dataset):
                if idx < start_idx:
                    continue
                messages = example.get("messages", [])
                if not messages:
                    continue

                conversation = []
                for turn_idx, msg in enumerate(messages):
                    if msg.get("role") != "user":
                        continue
                    prompt = msg.get("content", "")
                    if not prompt:
                        continue

                    context = conversation + [{"role": "user", "content": prompt}]
                    try:
                        resp = litellm.completion(
                            model=teacher_model, messages=context, max_tokens=512
                        )
                        teacher_text = resp.choices[0].message.content
                        conversation.append({"role": "user", "content": prompt})
                        conversation.append({"role": "assistant", "content": teacher_text})
                    except Exception as e:
                        self.logger.warning(f"Teacher query failed: {e}")
                        break

                if conversation:
                    teacher_pool.append({"messages": conversation})
                    with open(partial_path, "a") as f:
                        f.write(json.dumps({"messages": conversation}) + "\n")

                if (idx + 1) % 5 == 0:
                    progress = ((idx + 1) / len(dataset)) * 50
                    self.logger.set_progress(progress, f"Phase 1: {idx+1}/{len(dataset)}")

            if partial_path.exists():
                partial_path.rename(teacher_pool_path)
            self.logger.info(f"Phase 1 complete: {len(teacher_pool)} conversations")

        # ── Phase 2: Create weighted examples ──
        self.logger.info("Phase 2: Creating step-decay weighted examples...")
        teacher_pool = load_dataset("json", data_files=str(teacher_pool_path), split="train")

        weighted_examples = []
        for example in teacher_pool:
            messages = example.get("messages", [])
            num_turns = len([m for m in messages if m.get("role") == "assistant"])
            for turn_idx in range(num_turns):
                prefix_end = (turn_idx + 1) * 2
                turn_messages = messages[:prefix_end]
                weight = kappa ** turn_idx
                weighted_examples.append({"messages": turn_messages, "weight": weight})

        self.logger.info(f"Created {len(weighted_examples)} weighted examples")

        weighted_path = base_path / "on_policy_weighted.jsonl"
        with open(weighted_path, "w") as f:
            for item in weighted_examples:
                f.write(json.dumps(item) + "\n")

        # ── Phase 3: Train with weighted sampling ──
        self.logger.info("Phase 3: Training with step-decay weights...")
        max_seq_length = self.config.training.hyperparams.max_seq_length
        tcfg = self.config.training

        # Load SFT adapter if available
        if self.adapter_path and Path(self.adapter_path).exists():
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.adapter_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=(tcfg.quantization == "4bit"),
            )
            model = FastLanguageModel.get_peft_model(
                model, r=tcfg.lora.rank, target_modules=tcfg.lora.target_modules,
                lora_alpha=tcfg.lora.alpha, lora_dropout=tcfg.lora.dropout,
                bias="none", use_gradient_checkpointing="unsloth",
                random_state=tcfg.hyperparams.seed,
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.models.student,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=(tcfg.quantization == "4bit"),
            )
            model = FastLanguageModel.get_peft_model(
                model, r=tcfg.lora.rank, target_modules=tcfg.lora.target_modules,
                lora_alpha=tcfg.lora.alpha, lora_dropout=tcfg.lora.dropout,
                bias="none", use_gradient_checkpointing="unsloth",
                random_state=tcfg.hyperparams.seed,
            )

        op_dataset = load_dataset("json", data_files=str(weighted_path), split="train")

        def format_chatml(examples):
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return {"text": texts}

        op_dataset = op_dataset.map(format_chatml, batched=True, num_proc=1)

        # Weighted sampling
        weights = [ex["weight"] for ex in weighted_examples]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        import random
        random.seed(tcfg.hyperparams.seed)
        num_samples = min(len(weighted_examples), len(op_dataset) * 2)
        sampled_indices = random.choices(range(len(weighted_examples)), weights=probs, k=num_samples)

        sampled_path = base_path / "on_policy_sampled.jsonl"
        with open(sampled_path, "w") as f:
            for i in sampled_indices:
                f.write(json.dumps(weighted_examples[i]) + "\n")

        sampled_dataset = load_dataset("json", data_files=str(sampled_path), split="train")
        sampled_dataset = sampled_dataset.map(format_chatml, batched=True, num_proc=1)
        self.logger.info(f"Sampled {len(sampled_dataset)} examples")

        # Adapter dir
        sft_name = Path(self.adapter_path).name if self.adapter_path else self.run_id[:8]
        adapter_dir = Path("~/localdistill/adapters").expanduser() / f"{sft_name}_onpolicy"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        h = tcfg.hyperparams
        steps = (len(sampled_dataset) // (h.batch_size * h.gradient_accumulation_steps)) or 1

        training_args = TrainingArguments(
            per_device_train_batch_size=h.batch_size,
            gradient_accumulation_steps=h.gradient_accumulation_steps,
            warmup_steps=2,
            num_train_epochs=1,
            learning_rate=h.learning_rate * 0.5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=h.weight_decay,
            lr_scheduler_type="constant",
            seed=h.seed,
            output_dir=str(adapter_dir),
            report_to="none",
            dataloader_num_workers=0,
            save_strategy="no",
        )

        class OPCallback(TrainerCallback):
            def __init__(self, logger, total_steps):
                self.logger = logger
                self.total_steps = total_steps

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    step = state.global_step
                    loss = logs.get("loss", 0)
                    lr = logs.get("learning_rate", 0)
                    self.logger.log_training_step(step, loss, lr, total_steps=self.total_steps, phase="on_policy")
                    if self.total_steps > 0:
                        progress = 50 + (step / self.total_steps) * 50
                        self.logger.set_progress(progress, f"Phase 2: {step}/{self.total_steps}")

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=sampled_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
            callbacks=[OPCallback(self.logger, steps)],
        )

        self.logger.set_progress(50, "Phase 2: training started")
        trainer.train()
        self.logger.set_progress(100, "Phase 2: complete")

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        self.adapter_path = str(adapter_dir)
        self.logger.success(f"On-policy adapter saved: {adapter_dir}")

        self.metrics["on_policy_examples"] = len(sampled_dataset)
        self.metrics["on_policy_weighted_pool"] = len(weighted_examples)

    # ── Benchmark ──────────────────────────────────────────────────────────────

    def _run_benchmark(self):
        self.logger.set_stage(Stage.BENCHMARK)
        if not self.config.benchmark.enabled:
            self.logger.info("Benchmarking disabled")
            return
        if not self.adapter_path:
            self.adapter_path = get_latest_adapter()
        if not self.adapter_path:
            self.logger.warning("No adapter found, skipping benchmark")
            return

        self.logger.info(f"Benchmarking: {self.adapter_path}")

        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM

            device = self.config.benchmark.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model = HFLM(
                pretrained=self.config.models.student,
                peft=self.adapter_path,
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

    # ── Deploy ─────────────────────────────────────────────────────────────────

    def _run_deploy(self):
        self.logger.set_stage(Stage.DEPLOY)
        if not self.adapter_path:
            self.adapter_path = get_latest_adapter()
        if not self.adapter_path:
            self.logger.warning("No adapter found, skipping deployment")
            return

        gguf_dir = Path(self.adapter_path) / "gguf"
        modelfile = gguf_dir / "Modelfile"

        if not modelfile.exists():
            self.logger.warning("No Modelfile found")
            self.logger.info(f"Manual: ollama create {self.config.deploy.ollama.model_name} -f {modelfile}")
            return

        if not self.config.deploy.ollama.enabled:
            self.logger.info("Ollama deployment disabled")
            return

        if not check_ollama_installed():
            self.logger.warning("Ollama not installed")
            return

        model_name = self.config.deploy.ollama.model_name
        if register_ollama_model(str(modelfile), model_name, self.logger):
            self.metrics["ollama_model"] = model_name
            self.logger.success(f"Deployed: ollama run {model_name}")
        else:
            self.logger.error("Ollama registration failed")

    # ── Utilities ──────────────────────────────────────────────────────────────

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
        """Write training_summary.json with key results."""
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
                "strategy": self.config.training.strategy.name,
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
    config = load_config(args.config)

    if args.mode:
        config.run_mode = args.mode
        if args.mode in config.presets:
            _apply_preset(config, config.presets[args.mode])
    if args.student:
        config.models.student = args.student
    if args.teacher:
        config.models.teacher = args.teacher
    if args.max_examples:
        config.curation.max_examples = args.max_examples
    if args.epochs:
        config.training.hyperparams.epochs = args.epochs
    if args.on_policy:
        config.on_policy.enabled = True
        config.training.strategy.name = "on_policy"

    # Handle --resume
    resume_adapter = None
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if resume_path.exists() and (resume_path / "adapter_config.json").exists():
            resume_adapter = str(resume_path)
        elif args.resume == "latest":
            resume_adapter = get_latest_adapter()
        else:
            print(f"Error: Invalid adapter: {args.resume}")
            return 1
        if not resume_adapter:
            print("Error: No adapter to resume from")
            return 1
        print(f"Resuming: {resume_adapter}")
        config.on_policy.enabled = True

    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    elif resume_adapter:
        steps = ["on_policy", "deploy"]

    run_id = str(uuid.uuid4())
    logger = create_logger(run_id, config)

    pipeline = DistillPipeline(config, logger)
    if resume_adapter:
        pipeline.adapter_path = resume_adapter
    if args.resume_checkpoint:
        pipeline.resume_checkpoint = args.resume_checkpoint

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
        for a in adapters[:10]:
            gguf = "GGUF" if a["has_gguf"] else "    "
            print(f"  [{gguf}] {a['id']}  {a['created_at'][:16]}")
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LocalDistill - LLM Distillation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python distill.py run --mode preference
  python distill.py run --mode preference --steps curate,train
  python distill.py run --epochs 5 --lr 1e-4
  python distill.py status
  python distill.py logs
        """
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run pipeline")
    run_p.add_argument("--config", "-c", help="Config file")
    run_p.add_argument("--mode", "-m", choices=["demo", "full", "preference", "custom"])
    run_p.add_argument("--student", help="Student model")
    run_p.add_argument("--teacher", help="Teacher model")
    run_p.add_argument("--max-examples", type=int)
    run_p.add_argument("--epochs", type=int)
    run_p.add_argument("--steps", help="curate,train,evaluate,on_policy,benchmark,deploy")
    run_p.add_argument("--on-policy", action="store_true")
    run_p.add_argument("--resume", help="Resume from adapter path or 'latest'")
    run_p.add_argument("--resume-checkpoint", help="Resume from checkpoint dir")
    run_p.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="Show run status")
    sub.add_parser("logs", help="Tail latest logs")
    sub.add_parser("adapters", help="List adapters")

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "logs":
        sys.exit(cmd_logs(args))
    elif args.command == "adapters":
        sys.exit(cmd_adapters(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
