#!/usr/bin/env python3
"""
LocalDistill - Main Orchestrator

The central CLI for running the distillation pipeline.

Usage:
    python distill.py run                     # Run with config.yaml defaults
    python distill.py run --mode demo         # Quick demo (50 examples, 1 epoch)
    python distill.py run --mode full         # Full training
    python distill.py run --dry-run           # Show plan without executing
    
    python distill.py status                  # Show current/recent run status
    python distill.py logs                    # Tail latest logs
    python distill.py adapters                # List trained adapters
"""

import os
import sys
import uuid
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Load .env file if exists
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.config import load_config, save_config, print_config, Config
from lib.logger import (
    DistillLogger, PipelineStage, create_logger, 
    get_logger, set_logger, ProgressBar
)
from lib.dataset import DatasetLoader, load_dataset_from_config
from lib.deploy import (
    export_gguf, register_ollama_model, list_adapters,
    get_latest_adapter, deploy_adapter, check_ollama_installed
)


class DistillPipeline:
    """
    Main pipeline orchestrator.
    
    Coordinates: Curate -> Train -> On-Policy -> Benchmark -> Deploy
    """
    
    def __init__(self, config: Config, logger: DistillLogger):
        self.config = config
        self.logger = logger
        self.run_id = logger.run_id
        self.adapter_path: Optional[str] = None
        self.dataset_path: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
    
    def run(self, steps: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Args:
            steps: List of steps to run, or None for all.
                   Valid: ["curate", "train", "on_policy", "benchmark", "deploy"]
            dry_run: If True, show plan without executing.
        
        Returns:
            Result dict with status and metrics.
        """
        all_steps = ["curate", "train", "on_policy", "benchmark", "deploy"]
        steps = steps or all_steps
        
        # Filter based on config
        if not self.config.on_policy.enabled:
            steps = [s for s in steps if s != "on_policy"]
        if not self.config.benchmark.enabled:
            steps = [s for s in steps if s != "benchmark"]
        if not self.config.deploy.gguf.enabled and not self.config.deploy.ollama.enabled:
            steps = [s for s in steps if s != "deploy"]
        
        self.logger.header("LOCALDISTILL PIPELINE")
        print_config(self.config)
        
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Steps: {' -> '.join(steps)}")
        
        if dry_run:
            self.logger.info("DRY RUN - No changes will be made")
            return {"run_id": self.run_id, "status": "dry_run", "steps": steps}
        
        # Pre-flight checks
        self._validate_on_policy_requirements()
        
        # Save frozen config
        config_snapshot = self.logger.run_dir / "config.yaml"
        save_config(self.config, str(config_snapshot))
        
        try:
            for step in steps:
                if step == "curate":
                    self._run_curate()
                elif step == "train":
                    self._run_train()
                elif step == "on_policy":
                    self._run_on_policy()
                elif step == "benchmark":
                    self._run_benchmark()
                elif step == "deploy":
                    self._run_deploy()
            
            self.logger.complete("Pipeline finished successfully")
            
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
            return {
                "run_id": self.run_id,
                "status": "failed",
                "error": str(e),
            }
    
    def _run_curate(self):
        """Curate step: Load and filter dataset."""
        self.logger.set_stage(PipelineStage.CURATE)
        
        loader = DatasetLoader(self.config, self.logger)
        
        self.logger.info("Loading dataset...")
        conversations = loader.load()
        self.logger.info(f"Loaded {len(conversations)} raw conversations")
        
        self.logger.info("Applying curation filters...")
        curated = loader.curate(conversations)
        self.logger.info(f"Curated to {len(curated)} conversations")
        
        # Export to training file
        output_path = Path(self.config.logging.dir).parent / "train.jsonl"
        count = loader.export(curated, str(output_path), "chatml")
        
        self.dataset_path = str(output_path)
        self.metrics["curated_examples"] = count
        
        self.logger.success(f"Dataset ready: {count} examples -> {output_path}")
    
    def _run_train(self):
        """Train step: LoRA fine-tuning with Unsloth."""
        self.logger.set_stage(PipelineStage.TRAIN)
        
        if not self.dataset_path:
            self.dataset_path = str(Path(self.config.logging.dir).parent / "train.jsonl")
        
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Training dataset not found: {self.dataset_path}")
        
        # Check GPU
        self.logger.info("Checking GPU availability...")
        gpu_available, gpu_info, free_vram = self._check_gpu()
        
        if not gpu_available:
            raise RuntimeError(f"No GPU available: {gpu_info}")
        
        self.logger.info(f"GPU: {gpu_info}")
        
        # Import training dependencies
        self.logger.info("Loading model and tokenizer...")
        
        import torch
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import load_dataset
        from transformers import TrainingArguments, TrainerCallback
        from trl import SFTTrainer
        
        cfg = self.config.training
        model_name = self.config.models.student
        max_seq_length = cfg.hyperparams.max_seq_length
        
        # Auto-adjust for low VRAM (< 5GB free)
        if free_vram < 5.0:
            self.logger.warning(f"Low VRAM ({free_vram:.1f} GB free). Adjusting settings...")
            # Use smaller sequence length
            if max_seq_length > 1024:
                max_seq_length = 1024
                self.logger.info(f"Reduced max_seq_length to {max_seq_length}")
            # Suggest smaller model if still likely to fail
            if free_vram < 3.0:
                self.logger.warning("Very low VRAM. Consider closing other apps or using a smaller model.")
                self.logger.info("Tip: Set models.student to 'unsloth/Llama-3.2-1B-Instruct' for low VRAM")
        
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=(cfg.quantization == "4bit"),
        )
        
        self.logger.info(f"Loaded base model: {model_name}")
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.rank,
            target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.hyperparams.seed,
        )
        
        self.logger.info(f"LoRA applied: rank={cfg.lora.rank}, alpha={cfg.lora.alpha}")
        
        # Load dataset
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        self.logger.info(f"Training dataset: {len(dataset)} examples")
        
        # Format for training
        def format_chatml(examples):
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return {"text": texts}
        
        dataset = dataset.map(format_chatml, batched=True, num_proc=1)
        
        # Setup training
        adapter_dir = Path("~/localdistill/adapters").expanduser() / self.run_id[:8]
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            per_device_train_batch_size=cfg.hyperparams.batch_size,
            gradient_accumulation_steps=cfg.hyperparams.gradient_accumulation_steps,
            warmup_steps=cfg.hyperparams.warmup_steps,
            num_train_epochs=cfg.hyperparams.epochs,
            learning_rate=cfg.hyperparams.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging.metrics.log_every_n_steps,
            optim="adamw_8bit",
            weight_decay=cfg.hyperparams.weight_decay,
            lr_scheduler_type=cfg.hyperparams.lr_scheduler,
            seed=cfg.hyperparams.seed,
            output_dir=str(adapter_dir),
            report_to="none",
            dataloader_num_workers=0,
            save_strategy="no",
        )
        
        # Custom callback for logging
        class LogCallback(TrainerCallback):
            def __init__(cb_self, logger):
                cb_self.logger = logger
                cb_self.step = 0
            
            def on_log(cb_self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    cb_self.step = state.global_step
                    loss = logs.get("loss", 0)
                    lr = logs.get("learning_rate", 0)
                    cb_self.logger.log_training_step(cb_self.step, loss, lr)
        
        log_callback = LogCallback(self.logger)
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
            callbacks=[log_callback],
        )
        
        # Train
        self.logger.info(f"Starting training: {len(dataset)} examples, {cfg.hyperparams.epochs} epochs")
        
        trainer.train()
        
        # Save adapter
        self.logger.info("Saving adapter...")
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        
        self.adapter_path = str(adapter_dir)
        self.logger.success(f"Adapter saved: {adapter_dir}")
        
        # Export GGUF if enabled
        if self.config.deploy.gguf.enabled:
            self.logger.info("Exporting GGUF...")
            gguf_path = export_gguf(
                model, tokenizer, str(adapter_dir),
                self.config.deploy.gguf.quantization,
                self.logger,
            )
            if gguf_path:
                self.metrics["gguf_path"] = gguf_path
        
        self.metrics["adapter_path"] = str(adapter_dir)
        self.metrics["training_examples"] = len(dataset)
        self.metrics["epochs"] = cfg.hyperparams.epochs
    
    def _validate_on_policy_requirements(self):
        """Fail fast if on-policy is enabled but API key missing."""
        if not self.config.on_policy.enabled:
            return
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "on_policy.enabled=true but OPENROUTER_API_KEY not set. "
                "Add it to .env or export it."
            )
    
    def _run_on_policy(self):
        """On-policy distillation (ReOPD): Two-phase offline approach.
        
        Phase 1: Collect teacher trajectories (one-time, stores pool)
        Phase 2: Train student with step-decay weighting (no API calls)
        """
        self.logger.set_stage(PipelineStage.ON_POLICY)
        
        if not self.config.on_policy.enabled:
            self.logger.info("On-policy distillation disabled, skipping")
            return
        
        import torch
        import litellm
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        
        cfg = self.config.on_policy
        teacher_model = self.config.models.teacher
        kappa = 1.0 - (1.0 / cfg.teacher_query_interval)  # Decay base from interval
        # ponytail: using interval as proxy for kappa, proper kappa config can be added later
        
        self.logger.info(f"On-policy (ReOPD): teacher={teacher_model}, kappa={kappa:.2f}")
        
        # Load dataset
        if not self.dataset_path:
            self.dataset_path = str(Path(self.config.logging.dir).parent / "train.jsonl")
        
        if not Path(self.dataset_path).exists():
            self.logger.warning(f"No dataset found at {self.dataset_path}, skipping on-policy")
            return
        
        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        self.logger.info(f"Loaded {len(dataset)} conversations")
        
        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1: Collect teacher pool (one-time)
        # ══════════════════════════════════════════════════════════════════════
        teacher_pool_path = Path(self.config.logging.dir).parent / "teacher_pool.jsonl"
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        if teacher_pool_path.exists():
            self.logger.info(f"Teacher pool exists: {teacher_pool_path}, skipping collection")
        else:
            self.logger.info("Phase 1: Collecting teacher trajectories...")
            
            teacher_pool = []
            
            for idx, example in enumerate(dataset):
                messages = example.get("messages", [])
                if not messages:
                    continue
                
                # Build conversation with teacher responses for each turn
                conversation = []
                user_turns = [m for m in messages if m.get("role") == "user"]
                
                for turn_idx, user_msg in enumerate(user_turns):
                    prompt = user_msg.get("content", "")
                    if not prompt:
                        continue
                    
                    # Build context: all prior turns + current user message
                    context = conversation + [{"role": "user", "content": prompt}]
                    
                    try:
                        self.logger.info(f"[{idx+1}/{len(dataset)}] Turn {turn_idx+1}: querying {teacher_model}")
                        teacher_resp = litellm.completion(
                            model=teacher_model,
                            messages=context,
                            max_tokens=512,
                        )
                        teacher_response = teacher_resp.choices[0].message.content
                        
                        # Track token usage
                        usage = teacher_resp.usage
                        if usage:
                            total_prompt_tokens += usage.prompt_tokens
                            total_completion_tokens += usage.completion_tokens
                            self.logger.info(f"  -> {usage.prompt_tokens}+{usage.completion_tokens} tokens")
                        
                        # Add to conversation
                        conversation.append({"role": "user", "content": prompt})
                        conversation.append({"role": "assistant", "content": teacher_response})
                        
                    except Exception as e:
                        self.logger.warning(f"Teacher query failed: {e}")
                        break
                
                if conversation:
                    teacher_pool.append({"messages": conversation})
                
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Collected {idx + 1}/{len(dataset)} conversations")
            
            # Save teacher pool
            with open(teacher_pool_path, "w") as f:
                for item in teacher_pool:
                    f.write(json.dumps(item) + "\n")
            
            total_tokens = total_prompt_tokens + total_completion_tokens
            self.logger.info(f"Phase 1 complete: {len(teacher_pool)} conversations")
            self.logger.info(f"Total teacher tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion = {total_tokens}")
            
            self.metrics["teacher_prompt_tokens"] = total_prompt_tokens
            self.metrics["teacher_completion_tokens"] = total_completion_tokens
            self.metrics["teacher_total_tokens"] = total_tokens
        
        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2: Train with step-decay weighting (offline, no API calls)
        # ══════════════════════════════════════════════════════════════════════
        self.logger.info("Phase 2: Training with step-decay weighting...")
        
        # Load teacher pool and create weighted training examples
        teacher_pool = load_dataset("json", data_files=str(teacher_pool_path), split="train")
        
        weighted_examples = []
        for example in teacher_pool:
            messages = example.get("messages", [])
            if not messages:
                continue
            
            # Count turns (each user+assistant pair is one turn)
            num_turns = len([m for m in messages if m.get("role") == "assistant"])
            
            # Create training example for each turn with step-decay weight
            for turn_idx in range(num_turns):
                # Extract prefix up to this turn
                prefix_end = (turn_idx + 1) * 2  # user + assistant pairs
                turn_messages = messages[:prefix_end]
                
                # Compute weight: w_t = kappa^t (exponential decay)
                # Earlier turns (lower t) get higher weight
                weight = kappa ** turn_idx
                
                weighted_examples.append({
                    "messages": turn_messages,
                    "weight": weight,
                    "turn": turn_idx + 1,
                })
        
        self.logger.info(f"Created {len(weighted_examples)} weighted training examples")
        
        # Save weighted dataset
        weighted_path = Path(self.config.logging.dir).parent / "on_policy_weighted.jsonl"
        with open(weighted_path, "w") as f:
            for item in weighted_examples:
                f.write(json.dumps(item) + "\n")
        
        # Load for training — continue from SFT adapter if available
        max_seq_length = self.config.training.hyperparams.max_seq_length
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.models.student,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=(self.config.training.quantization == "4bit"),
        )
        
        # Load SFT adapter weights if we have them, then apply LoRA on top
        tcfg = self.config.training
        if self.adapter_path and Path(self.adapter_path).exists():
            self.logger.info(f"Loading SFT adapter: {self.adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()  # Merge weights so we can apply fresh LoRA
            self.logger.info("Merged SFT adapter weights into base model")
        
        # Apply fresh LoRA for on-policy training
        model = FastLanguageModel.get_peft_model(
            model,
            r=tcfg.lora.rank,
            target_modules=tcfg.lora.target_modules,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=tcfg.hyperparams.seed,
        )
        
        # Load weighted dataset
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
        
        # Sample examples proportional to weight (ReOPD sampling approach)
        # ponytail: simple weighted sampling, can add importance weighting later
        weights = [ex["weight"] for ex in weighted_examples]
        total_weight = sum(weights)
        sample_probs = [w / total_weight for w in weights]
        
        import random
        random.seed(tcfg.hyperparams.seed)
        num_samples = min(len(weighted_examples), len(dataset) * 2)  # Sample ~2x original size
        sampled_indices = random.choices(range(len(weighted_examples)), weights=sample_probs, k=num_samples)
        
        # Create sampled dataset
        sampled_data = [weighted_examples[i] for i in sampled_indices]
        sampled_path = Path(self.config.logging.dir).parent / "on_policy_sampled.jsonl"
        with open(sampled_path, "w") as f:
            for item in sampled_data:
                f.write(json.dumps(item) + "\n")
        
        sampled_dataset = load_dataset("json", data_files=str(sampled_path), split="train")
        sampled_dataset = sampled_dataset.map(format_chatml, batched=True, num_proc=1)
        
        self.logger.info(f"Sampled {len(sampled_dataset)} examples (weighted by step-decay)")
        
        # Train
        adapter_dir = Path("~/localdistill/adapters").expanduser() / f"{self.run_id[:8]}_onpolicy"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            per_device_train_batch_size=tcfg.hyperparams.batch_size,
            gradient_accumulation_steps=tcfg.hyperparams.gradient_accumulation_steps,
            warmup_steps=2,
            num_train_epochs=1,
            learning_rate=tcfg.hyperparams.learning_rate * 0.5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=tcfg.hyperparams.weight_decay,
            lr_scheduler_type="constant",
            seed=tcfg.hyperparams.seed,
            output_dir=str(adapter_dir),
            report_to="none",
            dataloader_num_workers=0,
            save_strategy="no",
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=sampled_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )
        
        self.logger.info(f"Training on {len(sampled_dataset)} step-decay weighted examples")
        trainer.train()
        
        # Save adapter
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        
        self.adapter_path = str(adapter_dir)
        self.logger.success(f"On-policy adapter saved: {adapter_dir}")
        
        self.metrics["on_policy_examples"] = len(sampled_dataset)
        self.metrics["on_policy_weighted_pool"] = len(weighted_examples)
    
    def _run_benchmark(self):
        """Benchmark step: Evaluate adapter with lm-eval."""
        self.logger.set_stage(PipelineStage.BENCHMARK)
        
        if not self.config.benchmark.enabled:
            self.logger.info("Benchmarking disabled, skipping")
            return
        
        if not self.adapter_path:
            self.adapter_path = get_latest_adapter()
        
        if not self.adapter_path:
            self.logger.warning("No adapter found, skipping benchmark")
            return
        
        self.logger.info(f"Benchmarking adapter: {self.adapter_path}")
        
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
            
            tasks = self.config.benchmark.tasks
            limit = self.config.benchmark.limit
            device = self.config.benchmark.device
            
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Running lm-eval on {tasks} (limit={limit}, device={device})")
            
            model = HFLM(
                pretrained=self.config.models.student,
                peft=self.adapter_path,
                batch_size="auto",
                trust_remote_code=True,
                device=device,
            )
            
            results = simple_evaluate(
                model=model,
                tasks=tasks,
                limit=limit,
                batch_size="auto",
            )
            
            # Extract scores
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
    
    def _run_deploy(self):
        """Deploy step: Register with Ollama."""
        self.logger.set_stage(PipelineStage.DEPLOY)
        
        if not self.adapter_path:
            self.adapter_path = get_latest_adapter()
        
        if not self.adapter_path:
            self.logger.warning("No adapter found, skipping deployment")
            return
        
        # Check for GGUF
        gguf_dir = Path(self.adapter_path) / "gguf"
        modelfile = gguf_dir / "Modelfile"
        
        if not modelfile.exists():
            self.logger.warning("No Modelfile found, skipping Ollama registration")
            return
        
        if not self.config.deploy.ollama.enabled:
            self.logger.info("Ollama deployment disabled")
            self.logger.info(f"Manual: ollama create {self.config.deploy.ollama.model_name} -f {modelfile}")
            return
        
        if not check_ollama_installed():
            self.logger.warning("Ollama not installed, skipping registration")
            self.logger.info(f"Manual: ollama create {self.config.deploy.ollama.model_name} -f {modelfile}")
            return
        
        model_name = self.config.deploy.ollama.model_name
        
        if register_ollama_model(str(modelfile), model_name, self.logger):
            self.metrics["ollama_model"] = model_name
            self.logger.success(f"Deployed to Ollama: ollama run {model_name}")
        else:
            self.logger.error("Ollama registration failed")
    
    def _check_gpu(self):
        """Check GPU availability and return free VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory // (1024**3)
                # Get free memory
                free_bytes = torch.cuda.mem_get_info()[0]
                free_gb = free_bytes / (1024**3)
                return True, f"{props.name} ({total_gb} GB, {free_gb:.1f} GB free)", free_gb
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return True, "Apple MPS", 8.0  # Assume 8GB for MPS
            else:
                return False, "No GPU detected", 0
        except ImportError:
            return False, "PyTorch not installed", 0


def cmd_run(args):
    """Run the pipeline."""
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.mode:
        config.run_mode = args.mode
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
    
    # Parse steps
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    
    # Create logger
    run_id = str(uuid.uuid4())
    logger = create_logger(run_id, config)
    
    # Run pipeline
    pipeline = DistillPipeline(config, logger)
    result = pipeline.run(steps=steps, dry_run=args.dry_run)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    
    return 0 if result["status"] in ("completed", "dry_run") else 1


def cmd_status(args):
    """Show status of recent runs."""
    logs_dir = Path(args.logs_dir or "./logs").expanduser()
    runs_dir = logs_dir / "runs"
    
    if not runs_dir.exists():
        print("No runs found")
        return 0
    
    runs = sorted(runs_dir.iterdir(), reverse=True)[:10]
    
    print("\n" + "=" * 70)
    print("  RECENT RUNS")
    print("=" * 70)
    
    for run_path in runs:
        status_file = run_path / "status.json"
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
            
            stage = status.get("stage", "unknown")
            stage_icon = {"complete": "OK", "failed": "ERR", "running": "..."}.get(stage, stage[:3].upper())
            
            print(f"  [{stage_icon:^5}] {run_path.name}  {status.get('progress', 0):.0f}%")
        else:
            print(f"  [???] {run_path.name}")
    
    print()
    return 0


def cmd_logs(args):
    """Tail logs from latest run."""
    logs_dir = Path(args.logs_dir or "./logs").expanduser()
    runs_dir = logs_dir / "runs"
    
    if not runs_dir.exists():
        print("No runs found")
        return 1
    
    # Find latest run
    runs = sorted(runs_dir.iterdir(), reverse=True)
    if not runs:
        print("No runs found")
        return 1
    
    log_file = runs[0] / "run.log"
    if not log_file.exists():
        print(f"No log file: {log_file}")
        return 1
    
    # Tail the file
    import subprocess
    try:
        subprocess.run(["tail", "-f", str(log_file)])
    except KeyboardInterrupt:
        pass
    
    return 0


def cmd_adapters(args):
    """List trained adapters."""
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
  python distill.py run                    # Run with defaults (demo mode)
  python distill.py run --mode full        # Full training
  python distill.py run --dry-run          # Show plan only
  python distill.py run --steps curate     # Only run curation
  python distill.py status                 # Check run status
  python distill.py logs                   # Tail latest logs
  python distill.py adapters               # List adapters
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--config", "-c", help="Config file path")
    run_parser.add_argument("--mode", "-m", choices=["demo", "full", "custom"],
                           help="Run mode (overrides config)")
    run_parser.add_argument("--student", help="Student model")
    run_parser.add_argument("--teacher", help="Teacher model")
    run_parser.add_argument("--max-examples", type=int, help="Max training examples")
    run_parser.add_argument("--epochs", type=int, help="Training epochs")
    run_parser.add_argument("--steps", help="Comma-separated steps: curate,train,benchmark,deploy")
    run_parser.add_argument("--on-policy", action="store_true", help="Enable on-policy distillation")
    run_parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show run status")
    status_parser.add_argument("--logs-dir", help="Logs directory")
    
    # logs command
    logs_parser = subparsers.add_parser("logs", help="Tail latest logs")
    logs_parser.add_argument("--logs-dir", help="Logs directory")
    
    # adapters command
    adapters_parser = subparsers.add_parser("adapters", help="List trained adapters")
    
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
        sys.exit(0)


if __name__ == "__main__":
    main()
