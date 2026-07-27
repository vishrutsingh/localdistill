"""
LocalDistill Configuration Loader

Loads and validates config.yaml, applies presets, and provides typed access.
All new training strategies, judge modes, and monitoring options are configurable here.
"""

import os
import sys
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from copy import deepcopy


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


# ── Config Dataclasses ────────────────────────────────────────────────────────

@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class HyperparamsConfig:
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_steps: int = 5
    weight_decay: float = 0.01
    lr_scheduler: str = "linear"
    seed: int = 42


@dataclass
class CheckpointConfig:
    """Checkpoint behavior."""
    enabled: bool = False
    steps: int = 100          # Save every N steps
    keep_last: int = 2        # Keep only last N checkpoints (disk space)
    save_final: bool = True   # Always save final adapter on completion


@dataclass
class EarlyStoppingConfig:
    """Stop training if loss stops improving."""
    enabled: bool = False
    patience: int = 50        # Steps without improvement before stopping
    min_delta: float = 0.001  # Improvement threshold
    monitor: str = "loss"     # Metric to watch
    mode: str = "min"         # "min" or "max"


@dataclass
class DatasetMixConfig:
    """Blend multiple dataset sources for training."""
    curated: float = 1.0      # train.jsonl (preference chosen or curated)
    teacher_pool: float = 0.0 # teacher_pool.jsonl (on-policy phase 1)
    on_policy_weighted: float = 0.0  # on_policy_weighted.jsonl
    # Normalized to sum=1 at load time


@dataclass
class TrainingStrategyConfig:
    """Which training algorithm to use."""
    name: str = "sft"         # sft | dpo | on_policy | spin
    # SFT: standard supervised fine-tuning on chosen responses
    # DPO: direct preference optimization (uses rejected pairs too)
    # On-policy: ReOPD (replay-based offline policy distillation)
    # SPIN: synthetic preference injection (future)
    dataset_source: str = "auto"  # auto | curated | teacher_pool | on_policy_weighted
    # "auto" = use teacher_pool if exists AND strategy=on_policy, else curated


@dataclass
class JudgeConfig:
    """Evaluation judge selection."""
    mode: str = "heuristic"   # heuristic | llm | human
    llm_model: str = "openrouter/deepseek/deepseek-chat"
    max_examples: int = 50    # Limit eval examples (LLM judge costs money)
    win_rate_target: float = 0.6  # Success gate: >60%


@dataclass
class TrainingConfig:
    lora: LoraConfig = field(default_factory=LoraConfig)
    hyperparams: HyperparamsConfig = field(default_factory=HyperparamsConfig)
    strategy: TrainingStrategyConfig = field(default_factory=TrainingStrategyConfig)
    dataset_mix: DatasetMixConfig = field(default_factory=DatasetMixConfig)
    quantization: str = "4bit"
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


@dataclass
class CurationConfig:
    min_quality_score: float = 0.0
    max_examples: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=lambda: {
        "min_turns": 2,
        "max_turns": 50,
        "min_chars": 100,
        "max_chars": 50000,
        "exclude_code_heavy": False,
    })


@dataclass
class HuggingFaceConfig:
    dataset_id: str = "RyokoAI/ShareGPT52K"
    split: str = "train"
    config: Optional[str] = None
    field_mapping: Dict[str, str] = field(default_factory=lambda: {
        "conversations": "conversations",
        "role_field": "from",
        "content_field": "value",
        "human_role": "human",
        "assistant_role": "gpt",
    })


@dataclass
class DatasetConfig:
    source: str = "file"
    path: str = "./curated_train.jsonl"
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    format: str = "chatml"
    holdout_ratio: float = 0.1


@dataclass
class ModelsConfig:
    student: str = "unsloth/Llama-3.2-3B-Instruct"
    teacher: str = "openrouter/deepseek/deepseek-chat"


@dataclass
class OnPolicyConfig:
    enabled: bool = False
    teacher_query_interval: int = 10
    decay_function: str = "linear"
    exponential_lambda: float = 0.9


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    tasks: List[str] = field(default_factory=lambda: ["gsm8k"])
    limit: Optional[int] = 50
    compare_base: bool = True
    device: str = "auto"


@dataclass
class GGUFConfig:
    enabled: bool = True
    quantization: str = "q4_k_m"


@dataclass
class OllamaConfig:
    enabled: bool = False
    model_name: str = "localdistill"
    auto_register: bool = True


@dataclass
class DeployConfig:
    gguf: GGUFConfig = field(default_factory=GGUFConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class ConsoleConfig:
    enabled: bool = True
    colors: bool = True
    progress_bars: bool = True


@dataclass
class MetricsConfig:
    log_every_n_steps: int = 1
    include_loss: bool = True
    include_lr: bool = True
    include_grad_norm: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "./logs"
    max_run_logs: int = 20
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


@dataclass
class MonitorConfig:
    enabled: bool = True
    port: int = 8080
    refresh_interval: int = 5


@dataclass
class Config:
    """Main configuration object."""
    run_mode: str = "demo"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    on_policy: OnPolicyConfig = field(default_factory=OnPolicyConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    presets: Dict[str, Any] = field(default_factory=dict)

    # Non-persistent state
    config_path: Optional[str] = None
    run_id: Optional[str] = None


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_training_config(data: dict) -> TrainingConfig:
    return TrainingConfig(
        lora=_dict_to_dataclass(LoraConfig, data.get("lora", {})),
        hyperparams=_dict_to_dataclass(HyperparamsConfig, data.get("hyperparams", {})),
        strategy=_dict_to_dataclass(TrainingStrategyConfig, data.get("strategy", {})),
        dataset_mix=_dict_to_dataclass(DatasetMixConfig, data.get("dataset_mix", {})),
        quantization=data.get("quantization", "4bit"),
        checkpoint=_dict_to_dataclass(CheckpointConfig, data.get("checkpoint", {})),
        early_stopping=_dict_to_dataclass(EarlyStoppingConfig, data.get("early_stopping", {})),
        judge=_dict_to_dataclass(JudgeConfig, data.get("judge", {})),
    )


def _build_deploy_config(data: dict) -> DeployConfig:
    return DeployConfig(
        gguf=_dict_to_dataclass(GGUFConfig, data.get("gguf", {})),
        ollama=_dict_to_dataclass(OllamaConfig, data.get("ollama", {})),
    )


def _build_logging_config(data: dict) -> LoggingConfig:
    return LoggingConfig(
        level=data.get("level", "INFO"),
        dir=data.get("dir", "./logs"),
        max_run_logs=data.get("max_run_logs", 20),
        console=_dict_to_dataclass(ConsoleConfig, data.get("console", {})),
        metrics=_dict_to_dataclass(MetricsConfig, data.get("metrics", {})),
    )


def _dict_to_dataclass(cls, data: dict):
    if data is None:
        return cls()
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        field_type = field_types[key]
        if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _expand_env_vars(obj):
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            return os.environ.get(obj[2:-1], "")
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


# ── Validation ────────────────────────────────────────────────────────────────

VALID_STRATEGIES = {"sft", "dpo", "on_policy", "spin"}
VALID_JUDGES = {"heuristic", "llm", "human"}
VALID_LR_SCHEDULERS = {"linear", "cosine", "constant", "cosine_with_restarts"}


def validate_config(config: Config) -> list[str]:
    """Return list of warnings/errors. Empty list = valid."""
    errs = []

    if config.training.strategy.name not in VALID_STRATEGIES:
        errs.append(f"Unknown strategy '{config.training.strategy.name}'. Valid: {VALID_STRATEGIES}")

    if config.training.strategy.dataset_source not in {"auto", "curated", "teacher_pool", "on_policy_weighted"}:
        errs.append(f"Unknown dataset_source '{config.training.strategy.dataset_source}'")

    if config.training.hyperparams.epochs <= 0:
        errs.append(f"epochs={config.training.hyperparams.epochs} must be > 0")
    if config.training.hyperparams.learning_rate <= 0:
        errs.append(f"learning_rate={config.training.hyperparams.learning_rate} must be > 0")
    if config.training.hyperparams.batch_size <= 0:
        errs.append(f"batch_size={config.training.hyperparams.batch_size} must be > 0")
    if config.training.hyperparams.warmup_steps < 0:
        errs.append(f"warmup_steps={config.training.hyperparams.warmup_steps} must be >= 0")

    if config.training.hyperparams.lr_scheduler not in VALID_LR_SCHEDULERS:
        errs.append(f"Unknown lr_scheduler '{config.training.hyperparams.lr_scheduler}'")

    mix = config.training.dataset_mix
    total = mix.curated + mix.teacher_pool + mix.on_policy_weighted
    if total <= 0 and config.training.strategy.name == "sft":
        errs.append("dataset_mix sums to <= 0 — no training data selected")
    if total > 0 and abs(total - 1.0) > 0.01:
        errs.append(f"dataset_mix sums to {total:.2f} (should be ~1.0)")

    # DPO requires preference dataset
    if config.training.strategy.name == "dpo" and config.dataset.source != "preference":
        errs.append("Strategy 'dpo' requires dataset.source='preference' (needs rejected pairs)")

    return errs


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        for p in [Path.cwd() / "config.yaml", Path(__file__).parent.parent / "config.yaml", Path.home() / "localdistill" / "config.yaml"]:
            if p.exists():
                config_path = str(p)
                break

    if config_path is None or not Path(config_path).exists():
        print("[config] No config file found, using defaults")
        return Config()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Apply preset
    run_mode = raw.get("run_mode", "demo")
    presets = raw.get("presets", {})
    if run_mode in presets:
        preset = presets[run_mode]
        for key, value in preset.items():
            if key in raw and isinstance(raw[key], dict) and isinstance(value, dict):
                raw[key] = deep_merge(raw[key], value)
            else:
                raw[key] = value

    raw = _expand_env_vars(raw)

    config = Config(
        run_mode=raw.get("run_mode", "demo"),
        dataset=_dict_to_dataclass(DatasetConfig, raw.get("dataset", {})),
        models=_dict_to_dataclass(ModelsConfig, raw.get("models", {})),
        curation=_dict_to_dataclass(CurationConfig, raw.get("curation", {})),
        training=_build_training_config(raw.get("training", {})),
        on_policy=_dict_to_dataclass(OnPolicyConfig, raw.get("on_policy", {})),
        benchmark=_dict_to_dataclass(BenchmarkConfig, raw.get("benchmark", {})),
        deploy=_build_deploy_config(raw.get("deploy", {})),
        logging=_build_logging_config(raw.get("logging", {})),
        monitor=_dict_to_dataclass(MonitorConfig, raw.get("monitor", {})),
        presets=presets,
        config_path=config_path,
    )

    if "huggingface" in raw.get("dataset", {}):
        config.dataset.huggingface = _dict_to_dataclass(HuggingFaceConfig, raw["dataset"]["huggingface"])

    return config


def _apply_preset(config: Config, preset: dict):
    """Apply a preset dict to config object (used by CLI)."""
    for key, value in preset.items():
        if key == "dataset" and isinstance(value, dict):
            for dk, dv in value.items():
                if dk == "huggingface" and isinstance(dv, dict):
                    for hk, hv in dv.items():
                        setattr(config.dataset.huggingface, hk, hv)
                else:
                    setattr(config.dataset, dk, dv)
        elif key == "curation" and isinstance(value, dict):
            for ck, cv in value.items():
                if ck == "filters" and isinstance(cv, dict):
                    config.curation.filters.update(cv)
                else:
                    setattr(config.curation, ck, cv)
        elif key == "training" and isinstance(value, dict):
            for tk, tv in value.items():
                if tk == "hyperparams" and isinstance(tv, dict):
                    for hk, hv in tv.items():
                        setattr(config.training.hyperparams, hk, hv)
                elif tk == "lora" and isinstance(tv, dict):
                    for lk, lv in tv.items():
                        setattr(config.training.lora, lk, lv)
                elif tk == "strategy" and isinstance(tv, dict):
                    for sk, sv in tv.items():
                        setattr(config.training.strategy, sk, sv)
                elif tk == "checkpoint" and isinstance(tv, dict):
                    for ck, cv in tv.items():
                        setattr(config.training.checkpoint, ck, cv)
                elif tk == "early_stopping" and isinstance(tv, dict):
                    for ek, ev in tv.items():
                        setattr(config.training.early_stopping, ek, ev)
                elif tk == "judge" and isinstance(tv, dict):
                    for jk, jv in tv.items():
                        setattr(config.training.judge, jk, jv)
                elif tk == "dataset_mix" and isinstance(tv, dict):
                    for mk, mv in tv.items():
                        setattr(config.training.dataset_mix, mk, mv)
                else:
                    setattr(config.training, tk, tv)
        elif key == "on_policy" and isinstance(value, dict):
            for ok, ov in value.items():
                setattr(config.on_policy, ok, ov)
        elif key == "benchmark" and isinstance(value, dict):
            for bk, bv in value.items():
                setattr(config.benchmark, bk, bv)
        elif key == "deploy" and isinstance(value, dict):
            if "gguf" in value:
                for gk, gv in value["gguf"].items():
                    setattr(config.deploy.gguf, gk, gv)
            if "ollama" in value:
                for ok, ov in value["ollama"].items():
                    setattr(config.deploy.ollama, ok, ov)


# ── Serialization ─────────────────────────────────────────────────────────────

def to_dict(obj) -> Any:
    """Recursively convert dataclass to dict."""
    if hasattr(obj, '__dataclass_fields__'):
        return {
            k: to_dict(v) for k, v in obj.__dict__.items()
            if not k.startswith('_') and k not in ('config_path', 'run_id')
        }
    elif isinstance(obj, list):
        return [to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def save_config(config: Config, path: str):
    """Save config to YAML (for freezing run config)."""
    with open(path, 'w') as f:
        yaml.dump(to_dict(config), f, default_flow_style=False, sort_keys=False)


def print_config(config: Config):
    """Pretty print configuration."""
    print("\n" + "=" * 60)
    print("  LOCALDISTILL CONFIGURATION")
    print("=" * 60)
    print(f"  Run mode:      {config.run_mode}")
    print(f"  Strategy:      {config.training.strategy.name}")
    print(f"  Dataset src:   {config.training.strategy.dataset_source}")
    print(f"  Config file:   {config.config_path}")
    print()
    print(f"  Dataset:       {config.dataset.source}")
    if config.dataset.source == "file":
        print(f"                 {config.dataset.path}")
    else:
        print(f"                 {config.dataset.huggingface.dataset_id}")
    print()
    print(f"  Student:       {config.models.student}")
    print(f"  Teacher:       {config.models.teacher}")
    print()
    print(f"  Max examples:  {config.curation.max_examples or 'all'}")
    print(f"  Epochs:        {config.training.hyperparams.epochs}")
    print(f"  LoRA rank:     {config.training.lora.rank}")
    print(f"  Learning rate: {config.training.hyperparams.learning_rate}")
    print(f"  LR scheduler:  {config.training.hyperparams.lr_scheduler}")
    print()
    print(f"  Checkpoint:    {config.training.checkpoint.enabled} (every {config.training.checkpoint.steps} steps)")
    print(f"  Early stop:    {config.training.early_stopping.enabled} (patience={config.training.early_stopping.patience})")
    print()
    print(f"  On-policy:     {config.on_policy.enabled}")
    print(f"  Benchmark:     {config.benchmark.enabled} ({', '.join(config.benchmark.tasks)})")
    print(f"  Judge:         {config.deploy.gguf.enabled}")
    print(f"  Deploy Ollama: {config.deploy.ollama.enabled}")
    print("=" * 60 + "\n")
