"""
localdistill — Unsloth LoRA Training Pipeline.

Trains a LoRA adapter on curated conversations from SQLite.
Supports local GPU training and cloud GPU rental (via config).

Usage:
  python train.py                      # Train with defaults
  python train.py --base unsloth/Llama-3.2-3B-Instruct  # Different model
  python train.py --cloud runpod       # Train on cloud GPU
  python train.py --dry-run            # Export dataset only, don't train
"""

import sys
import os
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from db import get_db, init_db
from dataset_exporter import export_dataset

# ── Training configuration ──

DEFAULT_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = Path("~/localdistill/adapters").expanduser()
DATASET_PATH = Path("~/localdistill/train.jsonl").expanduser()

# Cloud GPU providers (future)
CLOUD_PROVIDERS = {
    "runpod": {"gpu": "A6000", "image": "unsloth/unsloth:latest"},
    "vastai": {"gpu": "RTX 4090", "image": "unsloth/unsloth:latest"},
    "lambda": {"gpu": "A100", "image": "unsloth/unsloth:latest"},
}


def check_gpu() -> Tuple[bool, str]:
    """Check if a GPU is available for training."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem // 1024**3} GB)"
        elif torch.backends.mps.is_available():
            return True, "Apple MPS (Metal Performance Shaders)"
        else:
            return False, "No GPU detected"
    except ImportError:
        return False, "PyTorch not installed"


def train_lora(
    dataset_path: str,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    output_dir: Optional[str] = None,
    cloud_provider: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Train a LoRA adapter on the given dataset.
    
    Returns (run_id, adapter_path).
    """
    run_id = str(uuid.uuid4())
    
    if cloud_provider:
        return _train_on_cloud(run_id, dataset_path, base_model, cloud_provider,
                               lora_rank, lora_alpha, learning_rate, num_epochs)
    
    # ── Local GPU Training ──
    has_gpu, gpu_info = check_gpu()
    if not has_gpu:
        print(f"[localdistill] WARNING: {gpu_info}")
        print("[localdistill] Training requires a GPU. Use --cloud <provider> for cloud training.")
        print("[localdistill] Supported cloud providers:", ", ".join(CLOUD_PROVIDERS.keys()))
        return run_id, ""
    
    print(f"[localdistill] GPU: {gpu_info}")
    print(f"[localdistill] Base model: {base_model}")
    print(f"[localdistill] LoRA rank={lora_rank}, alpha={lora_alpha}, lr={learning_rate}")
    
    import torch
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load dataset
    if dataset_path.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    elif dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")
    
    print(f"[localdistill] Dataset: {len(dataset)} examples")
    
    # ChatML formatting function
    def format_chatml(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(format_chatml, batched=True)
    
    # Training arguments
    adapter_dir = output_dir or str(OUTPUT_DIR / run_id[:8])
    
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=adapter_dir,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    print(f"[localdistill] Starting training ({len(dataset)} examples, {num_epochs} epochs)...")
    trainer.train()
    
    # Save adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[localdistill] Adapter saved to {adapter_dir}")
    
    # Export to GGUF for Ollama (optional)
    _export_gguf(model, tokenizer, adapter_dir)
    
    return run_id, adapter_dir


def _export_gguf(model, tokenizer, adapter_dir: str):
    """Export LoRA adapter as GGUF for Ollama."""
    try:
        from unsloth import save_pretrained_gguf
        gguf_dir = Path(adapter_dir) / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        
        # Save merged model as GGUF (Q4_K_M quantization)
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"[localdistill] GGUF exported to {gguf_dir}")
        
        # Create Ollama Modelfile
        modelfile = gguf_dir / "Modelfile"
        gguf_file = list(gguf_dir.glob("*.gguf"))[0] if list(gguf_dir.glob("*.gguf")) else None
        if gguf_file:
            modelfile.write_text(
                f"FROM {gguf_file.name}\n"
                f"TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>\n{{{{ .System }}}}<|end|>\n{{{{ end }}}}"
                f"{{{{ if .Prompt }}}}<|user|>\n{{{{ .Prompt }}}}<|end|>\n{{{{ end }}}}"
                f"<|assistant|>\n\"\"\"\n"
            )
            print(f"[localdistill] Ollama Modelfile created: {modelfile}")
            print(f"[localdistill] Run: ollama create localdistill -f {modelfile}")
    except Exception as e:
        print(f"[localdistill] GGUF export skipped: {e}")


def _train_on_cloud(run_id, dataset_path, base_model, provider, *args) -> Tuple[str, str]:
    """Train on cloud GPU provider."""
    print(f"[localdistill] Cloud training via {provider} — not yet implemented")
    print(f"[localdistill] Upload dataset ({dataset_path}) to cloud storage")
    print(f"[localdistill] Launch GPU instance with {CLOUD_PROVIDERS.get(provider, {})}")
    print(f"[localdistill] Run train.py on remote instance")
    print(f"[localdistill] Download adapter back to local")
    return run_id, "(cloud)"


def run_training_pipeline(
    base_model: str = DEFAULT_BASE_MODEL,
    min_score: float = 0.0,
    max_examples: Optional[int] = None,
    cloud_provider: Optional[str] = None,
    dry_run: bool = False,
    mark_used: bool = True,
) -> dict:
    """
    Full training pipeline: export → train → record → mark.
    Returns a dict with run details.
    """
    init_db()
    db = get_db()
    
    # Create training run record
    run_id = str(uuid.uuid4())
    db.execute(
        """INSERT INTO training_runs 
           (id, base_model, lora_rank, lora_alpha, learning_rate, num_epochs, status, started_at)
           VALUES (?, ?, ?, ?, ?, ?, 'running', ?)""",
        (run_id, base_model, DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA,
         DEFAULT_LEARNING_RATE, DEFAULT_NUM_EPOCHS,
         datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    
    # Step 1: Export dataset
    count = export_dataset(
        output_path=str(DATASET_PATH),
        fmt="chatml",
        max_examples=max_examples,
        min_score=min_score,
    )
    
    if count == 0:
        db.execute("UPDATE training_runs SET status = 'failed', error_log = 'No examples to train on' WHERE id = ?", (run_id,))
        db.commit()
        db.close()
        return {"run_id": run_id, "status": "failed", "error": "No training examples"}
    
    db.execute("UPDATE training_runs SET num_examples = ? WHERE id = ?", (count, run_id))
    db.commit()
    
    if dry_run:
        db.execute("UPDATE training_runs SET status = 'completed' WHERE id = ?", (run_id,))
        db.commit()
        db.close()
        print(f"[localdistill] Dry run complete. {count} examples exported to {DATASET_PATH}")
        return {"run_id": run_id, "status": "completed", "examples": count, "dry_run": True}
    
    # Step 2: Train
    try:
        train_run_id, adapter_path = train_lora(
            dataset_path=str(DATASET_PATH),
            base_model=base_model,
            cloud_provider=cloud_provider,
        )
    except Exception as e:
        db.execute(
            "UPDATE training_runs SET status = 'failed', error_log = ?, completed_at = ? WHERE id = ?",
            (str(e), datetime.now(timezone.utc).isoformat(), run_id)
        )
        db.commit()
        db.close()
        return {"run_id": run_id, "status": "failed", "error": str(e)}
    
    # Step 3: Mark as used
    if mark_used and not cloud_provider:
        curated = db.execute(
            "SELECT conversation_id FROM curated_training WHERE used_in_training = 0 ORDER BY quality_score DESC LIMIT ?",
            (count,)
        ).fetchall()
        cids = [r["conversation_id"] for r in curated]
        for cid in cids:
            db.execute(
                "UPDATE curated_training SET used_in_training = 1, training_run_id = ? WHERE conversation_id = ?",
                (run_id, cid)
            )
    
    # Step 4: Update run record
    db.execute(
        """UPDATE training_runs 
           SET status = 'completed', adapter_path = ?, completed_at = ?
           WHERE id = ?""",
        (adapter_path, datetime.now(timezone.utc).isoformat(), run_id)
    )
    db.commit()
    db.close()
    
    result = {
        "run_id": run_id,
        "status": "completed",
        "examples": count,
        "adapter_path": adapter_path,
    }
    print(f"[localdistill] Training pipeline complete: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="localdistill LoRA training")
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL, help="Base model")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum quality score for training examples")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum training examples")
    parser.add_argument("--cloud", choices=list(CLOUD_PROVIDERS.keys()),
                        help="Cloud GPU provider")
    parser.add_argument("--dry-run", action="store_true",
                        help="Export dataset without training")
    parser.add_argument("--no-mark", action="store_true",
                        help="Don't mark examples as used after training")
    
    args = parser.parse_args()
    
    result = run_training_pipeline(
        base_model=args.base,
        min_score=args.min_score,
        max_examples=args.max_examples,
        cloud_provider=args.cloud,
        dry_run=args.dry_run,
        mark_used=not args.no_mark,
    )
    
    print(f"\n{'='*50}")
    print(f"Training result: {json.dumps(result, indent=2)}")