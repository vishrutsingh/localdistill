"""
LocalDistill Configuration Loader

Loads and validates config.yaml, applies presets, and provides typed access.

Model: one method selector (`training.method: sft | reopd | dpo | orpo | kto`).
Each method is an explicit chain of stages (see METHODS). There are no hidden
enable flags — if a stage is in the chain it runs, unless its output already
exists (resume).
"""

import os
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


# ── Methods ───────────────────────────────────────────────────────────────────
# Each method declares its full stage chain. Stages are filtered at runtime by
# config (e.g. evaluate only for preference datasets, generate only when
# generate_teacher=true). Adding a new training method = add an entry here and
# implement its stages in distill.py.

METHODS: Dict[str, List[str]] = {
    # Supervised fine-tuning on curated data (optionally on teacher completions)
    "sft": ["curate", "generate", "train", "evaluate", "benchmark", "deploy"],
    # ReOPD: replay-based policy distillation with per-turn decay weighting
    "reopd": ["curate", "collect", "weight", "train", "evaluate", "benchmark", "deploy"],
    # Preference methods (dataset.source must be "preference"):
    "dpo": ["curate", "train", "evaluate", "benchmark", "deploy"],   # Rafailov et al. 2023
    "orpo": ["curate", "train", "evaluate", "benchmark", "deploy"],  # Hong et al. 2024, no ref model
    "kto": ["curate", "train", "evaluate", "benchmark", "deploy"],   # Ethayarajh et al. 2024
}

# Methods that train on chosen/rejected pairs (curate exports pairs.jsonl)
PREFERENCE_METHODS = {"dpo", "orpo", "kto"}


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
    """HF-native checkpointing during training (enables true resume)."""
    enabled: bool = True
    steps: int = 100          # Save every N steps
    keep_last: int = 2        # Keep only last N checkpoints


@dataclass
class EarlyStoppingConfig:
    """Stop training if eval loss stops improving. Uses a small holdout split."""
    enabled: bool = False
    patience: int = 3         # Evals without improvement before stopping
    min_delta: float = 0.001
    eval_fraction: float = 0.02  # Fraction of train data used for eval loss


@dataclass
class JudgeConfig:
    """Evaluation judge selection."""
    mode: str = "heuristic"   # heuristic | llm | human
    # Keep in a different family from models.teacher — see config.yaml
    llm_model: str = "openrouter/openai/gpt-4o-mini"
    max_examples: int = 200
    win_rate_target: float = 0.6
    batch_size: int = 8            # generation batch size (halves on OOM)
    concurrency: int = 8           # parallel judge calls
    compare_teacher: bool = True   # also measure how much of the gap was closed
    regurgitation_examples: int = 50   # memorisation probe size; 0 disables


@dataclass
class TrainingConfig:
    method: str = "sft"       # sft | reopd (see METHODS)
    generate_teacher: bool = False  # SFT only: generate teacher completions first
    lora: LoraConfig = field(default_factory=LoraConfig)
    hyperparams: HyperparamsConfig = field(default_factory=HyperparamsConfig)
    quantization: str = "4bit"
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


@dataclass
class ReopdConfig:
    """ReOPD method settings (used only when training.method=reopd)."""
    kappa: float = 0.9            # Per-turn exponential decay: weight = kappa^turn
    epochs: int = 1               # Reopd train phase epochs
    lr_scale: float = 0.5         # Reopd LR = base LR * lr_scale
    max_teacher_tokens: int = 512 # Per-turn teacher completion budget


@dataclass
class PreferenceConfig:
    """Shared settings for dpo/orpo/kto."""
    beta: float = 0.1             # DPO/KTO beta; ORPO odds-ratio weight
    lr_scale: float = 1.0         # Method LR = base LR * lr_scale
    epochs: int = 1
    max_prompt_length: int = 1024
    # DPO on small GPUs: precompute reference log probs in a single pass
    # instead of keeping a second (reference) model resident.
    precompute_ref_log_probs: bool = True


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
    source: str = "file"          # file | huggingface | preference
    path: str = "./curated_train.jsonl"
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    format: str = "chatml"
    holdout_ratio: float = 0.1


@dataclass
class ModelsConfig:
    student: str = "unsloth/Llama-3.2-3B-Instruct"
    teacher: str = "openrouter/deepseek/deepseek-chat"
    # Concurrent teacher API calls (collect/generate stages). Calls within a
    # conversation stay sequential; conversations run in parallel.
    teacher_concurrency: int = 8
    # Response budget for the teacher, and for the student at eval time. One
    # value on purpose: if the student gets a smaller budget than the responses
    # it is compared against, it is judged as incomplete for a harness reason.
    teacher_max_tokens: int = 1024


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    tasks: List[str] = field(default_factory=lambda: ["gsm8k"])
    limit: Optional[int] = 50
    device: str = "auto"


@dataclass
class GGUFConfig:
    enabled: bool = True
    quantization: str = "q4_k_m"


@dataclass
class OllamaConfig:
    enabled: bool = False
    model_name: str = "localdistill"


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
    reopd: ReopdConfig = field(default_factory=ReopdConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
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
        method=data.get("method", "sft"),
        generate_teacher=data.get("generate_teacher", False),
        lora=_dict_to_dataclass(LoraConfig, data.get("lora", {})),
        hyperparams=_dict_to_dataclass(HyperparamsConfig, data.get("hyperparams", {})),
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

VALID_JUDGES = {"heuristic", "llm", "human"}
VALID_LR_SCHEDULERS = {"linear", "cosine", "constant", "cosine_with_restarts"}
VALID_SOURCES = {"file", "huggingface", "preference"}


def validate_config(config: Config) -> List[str]:
    """Return list of errors. Empty list = valid."""
    errs = []

    if config.training.method not in METHODS:
        errs.append(f"Unknown method '{config.training.method}'. Valid: {sorted(METHODS)}")

    if config.dataset.source not in VALID_SOURCES:
        errs.append(f"Unknown dataset.source '{config.dataset.source}'. Valid: {sorted(VALID_SOURCES)}")

    h = config.training.hyperparams
    if h.epochs <= 0:
        errs.append(f"epochs={h.epochs} must be > 0")
    if h.learning_rate <= 0:
        errs.append(f"learning_rate={h.learning_rate} must be > 0")
    if h.batch_size <= 0:
        errs.append(f"batch_size={h.batch_size} must be > 0")
    if h.warmup_steps < 0:
        errs.append(f"warmup_steps={h.warmup_steps} must be >= 0")
    if h.lr_scheduler not in VALID_LR_SCHEDULERS:
        errs.append(f"Unknown lr_scheduler '{h.lr_scheduler}'")

    r = config.reopd
    if not (0.0 < r.kappa <= 1.0):
        errs.append(f"reopd.kappa={r.kappa} must be in (0, 1]")
    if r.epochs <= 0:
        errs.append(f"reopd.epochs={r.epochs} must be > 0")
    if r.lr_scale <= 0:
        errs.append(f"reopd.lr_scale={r.lr_scale} must be > 0")

    pref = config.preference
    if pref.beta <= 0:
        errs.append(f"preference.beta={pref.beta} must be > 0")
    if pref.lr_scale <= 0:
        errs.append(f"preference.lr_scale={pref.lr_scale} must be > 0")

    if config.training.method in PREFERENCE_METHODS and config.dataset.source != "preference":
        errs.append(f"method={config.training.method} requires dataset.source='preference' (chosen/rejected pairs)")

    if config.training.judge.mode not in VALID_JUDGES:
        errs.append(f"Unknown judge mode '{config.training.judge.mode}'")

    if not (1 <= config.models.teacher_concurrency <= 64):
        errs.append(f"teacher_concurrency={config.models.teacher_concurrency} must be in [1, 64]")

    return errs


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None, run_mode: Optional[str] = None) -> Config:
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

    # Apply preset (CLI --mode overrides the yaml run_mode)
    presets = raw.get("presets", {})
    mode = run_mode or raw.get("run_mode", "demo")
    if mode in presets:
        raw = deep_merge(raw, presets[mode])
    raw["run_mode"] = mode

    raw = _expand_env_vars(raw)

    config = Config(
        run_mode=mode,
        dataset=_dict_to_dataclass(DatasetConfig, raw.get("dataset", {})),
        models=_dict_to_dataclass(ModelsConfig, raw.get("models", {})),
        curation=_dict_to_dataclass(CurationConfig, raw.get("curation", {})),
        training=_build_training_config(raw.get("training", {})),
        reopd=_dict_to_dataclass(ReopdConfig, raw.get("reopd", {})),
        preference=_dict_to_dataclass(PreferenceConfig, raw.get("preference", {})),
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


def print_config(config: Config, paths: Optional[Dict[str, Path]] = None):
    """Pretty print configuration."""
    print("\n" + "=" * 60)
    print("  LOCALDISTILL CONFIGURATION")
    print("=" * 60)
    print(f"  Run mode:      {config.run_mode}")
    print(f"  Method:        {config.training.method}")
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
    if config.training.method == "sft":
        print(f"  Gen teacher:   {config.training.generate_teacher}")
    else:
        print(f"  ReOPD:         kappa={config.reopd.kappa}, epochs={config.reopd.epochs}, lr_scale={config.reopd.lr_scale}")
    print()
    print(f"  Checkpoint:    {config.training.checkpoint.enabled} (every {config.training.checkpoint.steps} steps)")
    print(f"  Early stop:    {config.training.early_stopping.enabled}")
    print(f"  Benchmark:     {config.benchmark.enabled} ({', '.join(config.benchmark.tasks)})")
    print(f"  Judge:         {config.training.judge.mode}")
    print(f"  Deploy GGUF:   {config.deploy.gguf.enabled}")
    print(f"  Deploy Ollama: {config.deploy.ollama.enabled}")
    if paths:
        print()
        print("  Artifacts:")
        for name, p in paths.items():
            print(f"    {name:16s} {p}")
    print("=" * 60 + "\n")
