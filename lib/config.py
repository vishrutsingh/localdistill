"""
LocalDistill Configuration Loader

Loads and validates config.yaml, applies presets, and provides typed access.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
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
class TrainingConfig:
    lora: LoraConfig = field(default_factory=LoraConfig)
    hyperparams: HyperparamsConfig = field(default_factory=HyperparamsConfig)
    quantization: str = "4bit"
    save_checkpoints: bool = False
    checkpoint_steps: int = 100


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
    source: str = "file"  # file | huggingface | preference
    path: str = "./curated_train.jsonl"
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    format: str = "chatml"  # chatml | sharegpt | alpaca
    holdout_ratio: float = 0.1  # Fraction held out for evaluation


@dataclass
class ModelsConfig:
    student: str = "unsloth/Llama-3.2-3B-Instruct"
    teacher: str = "openrouter/deepseek/deepseek-chat"


@dataclass
class OnPolicyConfig:
    enabled: bool = False
    teacher_query_interval: int = 10      # Teacher corrects every N turns
    decay_function: str = "linear"        # "linear" or "exponential"
    exponential_lambda: float = 0.9       # Only for exponential decay


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
    
    # Runtime info (not from config file)
    config_path: Optional[str] = None
    run_id: Optional[str] = None


def _dict_to_dataclass(cls, data: dict):
    """Convert nested dict to dataclass, handling nested dataclasses."""
    if data is None:
        return cls()
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for key, value in data.items():
        if key not in field_types:
            continue
        
        field_type = field_types[key]
        
        # Handle nested dataclasses
        if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value
    
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for config.yaml
                    in current directory and parent directories.
    
    Returns:
        Config object with all settings.
    """
    # Find config file
    if config_path is None:
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / "localdistill" / "config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path is None or not Path(config_path).exists():
        print(f"[config] No config file found, using defaults")
        return Config()
    
    # Load YAML
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    
    # Apply preset if run_mode matches
    run_mode = raw.get("run_mode", "demo")
    presets = raw.get("presets", {})
    
    if run_mode in presets:
        preset = presets[run_mode]
        # Deep merge preset into raw config
        for key, value in preset.items():
            if key in raw and isinstance(raw[key], dict) and isinstance(value, dict):
                raw[key] = deep_merge(raw[key], value)
            else:
                raw[key] = value
    
    # Expand environment variables in string values
    raw = _expand_env_vars(raw)
    
    # Build config object
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
    
    # Handle HuggingFace config
    if "huggingface" in raw.get("dataset", {}):
        config.dataset.huggingface = _dict_to_dataclass(
            HuggingFaceConfig, raw["dataset"]["huggingface"]
        )
    
    return config


def _build_training_config(data: dict) -> TrainingConfig:
    """Build TrainingConfig with nested dataclasses."""
    return TrainingConfig(
        lora=_dict_to_dataclass(LoraConfig, data.get("lora", {})),
        hyperparams=_dict_to_dataclass(HyperparamsConfig, data.get("hyperparams", {})),
        quantization=data.get("quantization", "4bit"),
        save_checkpoints=data.get("save_checkpoints", False),
        checkpoint_steps=data.get("checkpoint_steps", 100),
    )


def _build_deploy_config(data: dict) -> DeployConfig:
    """Build DeployConfig with nested dataclasses."""
    return DeployConfig(
        gguf=_dict_to_dataclass(GGUFConfig, data.get("gguf", {})),
        ollama=_dict_to_dataclass(OllamaConfig, data.get("ollama", {})),
    )


def _build_logging_config(data: dict) -> LoggingConfig:
    """Build LoggingConfig with nested dataclasses."""
    return LoggingConfig(
        level=data.get("level", "INFO"),
        dir=data.get("dir", "./logs"),
        max_run_logs=data.get("max_run_logs", 20),
        console=_dict_to_dataclass(ConsoleConfig, data.get("console", {})),
        metrics=_dict_to_dataclass(MetricsConfig, data.get("metrics", {})),
    )


def _expand_env_vars(obj):
    """Recursively expand ${VAR} environment variables in strings."""
    if isinstance(obj, str):
        # Handle ${VAR} syntax
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def save_config(config: Config, path: str):
    """Save config to YAML file (for freezing run config)."""
    # Convert dataclass to dict recursively
    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items() 
                    if not k.startswith('_') and k not in ('config_path', 'run_id')}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj
    
    data = to_dict(config)
    
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def print_config(config: Config):
    """Pretty print configuration."""
    print("\n" + "=" * 60)
    print("  LOCALDISTILL CONFIGURATION")
    print("=" * 60)
    print(f"  Run mode:      {config.run_mode}")
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
    print()
    print(f"  On-policy:     {config.on_policy.enabled}")
    print(f"  Benchmark:     {config.benchmark.enabled} ({', '.join(config.benchmark.tasks)})")
    print(f"  Deploy GGUF:   {config.deploy.gguf.enabled}")
    print(f"  Deploy Ollama: {config.deploy.ollama.enabled}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test loading
    config = load_config()
    print_config(config)
