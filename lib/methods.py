"""
Artifact keys and path resolution.

All pipeline artifacts are content-addressed: their paths derive from a hash
of the config that produces them. Rerunning with the same config reuses
existing artifacts (automatic resume); changing any relevant setting produces
a new path, so runs never stomp each other.

Layout:
  data/curated_<key>/train.jsonl        curated conversations (method input)
  data/curated_<key>/holdout.jsonl      preference holdout pairs (eval input)
  data/curated_<key>/meta.json          what produced this dir
  data/teacher_singleturn_<key>.jsonl   SFT teacher completions (1 per prompt)
  data/teacher_trajectories_<key>.jsonl ReOPD multi-turn teacher trajectories
  data/reopd_weighted_<key>.jsonl       per-turn prefix examples with weights
  data/reopd_sampled_<key>.jsonl        weighted-resampled ReOPD train set
  adapters/<method>_<key>/              trained adapter (keyed by full config)

Files ending in .partial are append-only resume state for teacher generation;
they are renamed atomically on completion and never deleted except by --force.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from lib.config import Config, to_dict, PREFERENCE_METHODS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ADAPTERS_DIR = ROOT / "adapters"


def _key(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def compute_artifacts(config: Config, method: str, from_adapter: Optional[str] = None) -> Dict[str, Path]:
    """Resolve every artifact path for a run. Pure function of config."""
    curate_key = _key({"dataset": to_dict(config.dataset), "curation": to_dict(config.curation)})
    teacher_base = {"curate": curate_key, "teacher": config.models.teacher}
    single_key = _key({**teacher_base, "kind": "singleturn"})
    traj_key = _key({**teacher_base, "kind": "trajectories"})
    reopd_key = _key({"traj": traj_key, "kappa": config.reopd.kappa})

    curate_dir = DATA_DIR / f"curated_{curate_key}"
    teacher_single = DATA_DIR / f"teacher_singleturn_{single_key}.jsonl"
    teacher_traj = DATA_DIR / f"teacher_trajectories_{traj_key}.jsonl"
    reopd_weighted = DATA_DIR / f"reopd_weighted_{reopd_key}.jsonl"
    reopd_sampled = DATA_DIR / f"reopd_sampled_{reopd_key}.jsonl"

    # The adapter key binds: method, model, training config, and the exact
    # training data. Same everything -> same dir -> trainer checkpoints resume.
    if method == "sft":
        data_key = single_key if config.training.generate_teacher else curate_key
        train_dataset = teacher_single if config.training.generate_teacher else curate_dir / "train.jsonl"
        method_cfg = None
    elif method == "reopd":
        data_key = reopd_key
        train_dataset = reopd_sampled
        method_cfg = to_dict(config.reopd)
    elif method in PREFERENCE_METHODS:  # train on pairs.jsonl from curate
        data_key = curate_key
        train_dataset = curate_dir / "pairs.jsonl"
        method_cfg = to_dict(config.preference)
    else:
        raise ValueError(f"No artifact mapping for method '{method}'")

    adapter_key = _key({
        "method": method,
        "student": config.models.student,
        "lora": to_dict(config.training.lora),
        "hyperparams": to_dict(config.training.hyperparams),
        "method_cfg": method_cfg,
        "quantization": config.training.quantization,
        "data": data_key,
        "warm_start": Path(from_adapter).name if from_adapter else "",
    })
    adapter_dir = ADAPTERS_DIR / f"{method}_{adapter_key}"

    return {
        "curate_dir": curate_dir,
        "train_data": curate_dir / "train.jsonl",
        "pairs": curate_dir / "pairs.jsonl",
        "holdout": curate_dir / "holdout.jsonl",
        "curate_meta": curate_dir / "meta.json",
        "teacher_singleturn": teacher_single,
        "teacher_trajectories": teacher_traj,
        "reopd_weighted": reopd_weighted,
        "reopd_sampled": reopd_sampled,
        "train_dataset": Path(train_dataset),
        "adapter_dir": adapter_dir,
        "adapter_done": adapter_dir / "training_done.json",
    }


def latest_adapter(adapters_dir: Optional[Path] = None) -> Optional[str]:
    """Most recently modified adapter dir containing adapter_config.json."""
    d = adapters_dir or ADAPTERS_DIR
    if not d.exists():
        return None
    candidates = [
        p for p in d.iterdir()
        if p.is_dir() and (p / "adapter_config.json").exists()
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))
