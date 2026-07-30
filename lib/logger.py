"""
LocalDistill Logging Framework

Structured logging with:
- Console output with colors and optional progress display
- Per-run file logging (plain text + JSONL events)
- Real-time status.json for dashboard/monitor
- Automatic GPU stat logging
- Progress tracking (step-based, not just stage-based)
- Training step metrics with loss convergence detection
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from queue import Queue


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    CRITICAL = "CRITICAL"


class PipelineStage(Enum):
    INIT = "init"
    GENERATE = "generate"
    CURATE = "curate"
    COLLECT = "collect"
    WEIGHT = "weight"
    TRAIN = "train"
    EVALUATE = "evaluate"
    BENCHMARK = "benchmark"
    DEPLOY = "deploy"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class LogEvent:
    """Structured event for dashboard/events.jsonl."""
    timestamp: str
    level: str
    stage: str
    message: str
    run_id: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class RunStatus:
    """Serializable run state."""
    run_id: str
    stage: PipelineStage
    started_at: str
    progress: float = 0.0  # 0-100, updated at start of each stage
    current_step: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    completed_at: Optional[str] = None
    gpu_info: str = ""


# ── Colors ────────────────────────────────────────────────────────────────────

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


LEVEL_COLORS = {
    LogLevel.DEBUG: Colors.DIM,
    LogLevel.INFO: Colors.CYAN,
    LogLevel.WARNING: Colors.YELLOW,
    LogLevel.ERROR: Colors.RED,
    LogLevel.SUCCESS: Colors.GREEN,
    LogLevel.CRITICAL: Colors.RED + Colors.BOLD,
}

STAGE_ICONS = {
    PipelineStage.INIT: "🔧",
    PipelineStage.GENERATE: "✨",
    PipelineStage.CURATE: "📋",
    PipelineStage.COLLECT: "📡",
    PipelineStage.WEIGHT: "🔀",
    PipelineStage.TRAIN: "🏋️",
    PipelineStage.EVALUATE: "⚖️",
    PipelineStage.BENCHMARK: "📊",
    PipelineStage.DEPLOY: "🚀",
    PipelineStage.COMPLETE: "✅",
    PipelineStage.FAILED: "❌",
}


# ── GPU Helpers ───────────────────────────────────────────────────────────────

def _get_gpu_info() -> str:
    """Get brief GPU status string."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            p = [x.strip() for x in r.stdout.split(",")]
            if len(p) >= 4:
                return f"T:{p[0]}°C U:{p[1]}% M:{p[2]}MiB P:{p[3]}W"
        return "gpu-ok"
    except Exception:
        return "no-gpu"


def _get_loss_trend(metrics_history: List[Dict], window: int = 10) -> Optional[str]:
    """Analyze recent loss history. Returns: 'improving' | 'plateau' | 'worsening' | 'nan' | None"""
    if len(metrics_history) < window:
        return None
    recent = metrics_history[-window:]
    losses = [m.get("loss", float('nan')) for m in recent]
    if any(str(l) == 'nan' for l in losses if isinstance(l, float)):
        return "nan"
    # Simple linear regression slope
    n = len(losses)
    x_mean = (n - 1) / 2
    y_mean = sum(losses) / n
    numerator = sum((i - x_mean) * (losses[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return None
    slope = numerator / denominator
    if slope < -0.001:
        return "improving"
    elif slope > 0.002:
        return "worsening"
    else:
        return "plateau"


def analyze_training(rows: List[Dict]) -> Dict[str, Any]:
    """Read a run's metrics.jsonl back and name what the loss curve did.

    _get_loss_trend fits a line over a 20-step window, which reports
    "improving" throughout textbook memorisation — train loss falls smoothly
    the whole time. The two signatures that actually matter are:

      * divergence: held-out loss rising while training loss falls
      * epoch-boundary drops: a discontinuity in training loss each time the
        model starts re-reading data it has already seen, growing with each
        epoch

    Both are computed here, plus the best held-out checkpoint, which is the
    answer to "how many epochs should I have run".
    """
    train = [(r.get("step", 0), r["loss"], r.get("epoch", 0.0))
             for r in rows if "loss" in r and r.get("loss") == r.get("loss")]
    holdout = [(r.get("step", 0), r["eval_holdout_loss"])
               for r in rows if "eval_holdout_loss" in r]
    subset = [(r.get("step", 0), r["eval_train_subset_loss"])
              for r in rows if "eval_train_subset_loss" in r]

    out: Dict[str, Any] = {
        "train_points": len(train),
        "holdout_points": len(holdout),
        "final_train_loss": train[-1][1] if train else None,
        # A single final batch is noise; the tail mean is what to compare runs on.
        "mean_last_10_train_loss": (sum(l for _, l, _ in train[-10:]) / len(train[-10:])) if train else None,
    }

    if holdout:
        best_step, best_loss = min(holdout, key=lambda p: p[1])
        out["best_holdout_loss"] = best_loss
        out["best_holdout_step"] = best_step
        out["final_holdout_loss"] = holdout[-1][1]
        out["holdout_series"] = holdout
        # Rose on 2+ consecutive evals => past the useful point
        rises = 0
        max_rises = 0
        for (_, prev), (_, cur) in zip(holdout, holdout[1:]):
            rises = rises + 1 if cur > prev else 0
            max_rises = max(max_rises, rises)
        out["holdout_consecutive_rises"] = max_rises
        out["divergence_detected"] = max_rises >= 2
        out["holdout_regression_from_best"] = holdout[-1][1] - best_loss

    if subset:
        out["final_train_subset_loss"] = subset[-1][1]
        out["train_subset_series"] = subset
    if subset and holdout:
        gaps = [h - s for (_, s), (_, h) in zip(subset, holdout)]
        out["generalization_gap"] = gaps[-1]
        out["generalization_gap_series"] = gaps
        # A widening gap is only evidence of overfitting once held-out loss has
        # actually regressed. While held-out loss is still falling, the gap
        # widens simply because training loss falls faster — that is normal.
        out["generalization_gap_widened"] = (
            len(gaps) > 1
            and gaps[-1] > gaps[0]
            and out.get("holdout_regression_from_best", 0.0) > 0
        )

    # Epoch-boundary drops: mean train loss at the end of epoch N vs the start
    # of epoch N+1. A model that has memorised recognises its data immediately.
    by_epoch: Dict[int, List[float]] = {}
    for _, loss, epoch in train:
        by_epoch.setdefault(int(epoch), []).append(loss)
    epoch_means = {e: sum(v) / len(v) for e, v in sorted(by_epoch.items()) if v}
    out["epoch_mean_train_loss"] = epoch_means
    drops = []
    for e in sorted(by_epoch)[:-1]:
        tail = by_epoch[e][-5:]
        head = by_epoch[e + 1][:5] if (e + 1) in by_epoch else []
        if tail and head:
            drops.append({"boundary": f"{e}->{e+1}",
                          "drop": sum(tail) / len(tail) - sum(head) / len(head)})
    out["epoch_boundary_drops"] = drops
    out["epoch_boundary_drops_growing"] = (
        len(drops) > 1 and all(b["drop"] > a["drop"] for a, b in zip(drops, drops[1:]))
    )
    return out


def _package_versions() -> Dict[str, str]:
    """Versions of the packages that change results between runs."""
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for pkg in ("torch", "transformers", "trl", "peft", "unsloth", "datasets",
                "accelerate", "bitsandbytes", "litellm"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def write_provenance(run_dir: Path, extra: Dict[str, Any] = None) -> Path:
    """Record what produced a run, so a result can be reproduced or bisected.

    Cheap insurance: without the code SHA and the library versions, two runs
    that report the same config can have trained and evaluated differently and
    there is no way to find out afterwards.
    """
    import platform
    import subprocess

    def _git(*args) -> str:
        try:
            r = subprocess.run(["git", *args], capture_output=True, text=True,
                               timeout=5, cwd=str(Path(__file__).resolve().parent.parent))
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    gpu = "none"
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu = f"{p.name} ({p.total_memory // (1024**3)} GB)"
    except Exception:
        pass

    prov = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "packages": _package_versions(),
        **(extra or {}),
    }
    path = run_dir / "provenance.json"
    with open(path, "w") as f:
        json.dump(prov, f, indent=2, default=str)
    return path


# ── Logger ────────────────────────────────────────────────────────────────────

class DistillLogger:
    """
    Structured logger with console + file + event streaming + status tracking.
    """

    def __init__(
        self,
        run_id: str,
        log_dir: str = "./logs",
        level: str = "INFO",
        console_colors: bool = True,
        console_enabled: bool = True,
    ):
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.console_colors = console_colors and sys.stdout.isatty()
        self.console_enabled = console_enabled

        self.current_stage = PipelineStage.INIT
        self.started_at = datetime.now(timezone.utc).isoformat()

        # Event streaming
        self.event_queue: Queue = Queue()
        self.subscribers: List[Callable[[LogEvent], None]] = []

        # Status + history
        self.status = RunStatus(
            run_id=run_id,
            stage=PipelineStage.INIT,
            started_at=self.started_at,
        )
        self.metrics_history: List[Dict] = []  # For trend analysis
        self._last_progress_update = 0

        # Setup dirs and files
        self._setup_directories()
        self._setup_file_logging()

    def _setup_directories(self):
        self.run_dir = self.log_dir / "runs" / f"{datetime.now().strftime('%Y-%m-%d')}_{self.run_id[:8]}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_files = {
            "main": self.run_dir / "run.log",
            "events": self.run_dir / "events.jsonl",
            "metrics": self.run_dir / "metrics.jsonl",
            "curate": self.run_dir / "curate.log",
            "train": self.run_dir / "train.log",
            "evaluate": self.run_dir / "evaluate.log",
            "benchmark": self.run_dir / "benchmark.log",
            "deploy": self.run_dir / "deploy.log",
        }

    def _setup_file_logging(self):
        self._pylogger = logging.getLogger(f"distill.{self.run_id[:8]}")
        self._pylogger.setLevel(self.level)
        self._pylogger.handlers.clear()
        fh = logging.FileHandler(self.log_files["main"])
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self._pylogger.addHandler(fh)

    def _format_console(self, level: LogLevel, stage: PipelineStage, message: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        if self.console_colors:
            color = LEVEL_COLORS.get(level, "")
            icon = STAGE_ICONS.get(stage, "")
            stage_str = f"{Colors.BOLD}[{stage.value:^10}]{Colors.RESET}"
            return f"{Colors.DIM}{ts}{Colors.RESET} {icon} {stage_str} {color}{message}{Colors.RESET}"
        return f"{ts} [{stage.value:^10}] {message}"

    def _emit_event(self, level: LogLevel, message: str, data: Dict[str, Any] = None):
        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            stage=self.current_stage.value,
            message=message,
            run_id=self.run_id,
            data=data or {},
        )
        with open(self.log_files["events"], "a") as f:
            f.write(event.to_json() + "\n")

        # Keep last 100 logs in status
        self.status.logs.append(f"[{event.timestamp}] {message}")
        if len(self.status.logs) > 100:
            self.status.logs = self.status.logs[-100:]

        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception:
                pass
        self.event_queue.put(event)

    def log(self, level: LogLevel, message: str, data: Dict[str, Any] = None):
        if self.console_enabled:
            print(self._format_console(level, self.current_stage, message))

        py_level = getattr(logging, level.value, logging.INFO)
        self._pylogger.log(py_level, f"[{self.current_stage.value}] {message}")
        self._emit_event(level, message, data)

    def debug(self, message: str, **data):
        self.log(LogLevel.DEBUG, message, data)

    def info(self, message: str, **data):
        self.log(LogLevel.INFO, message, data)

    def warning(self, message: str, **data):
        self.log(LogLevel.WARNING, message, data)

    def error(self, message: str, **data):
        self.log(LogLevel.ERROR, message, data)

    def success(self, message: str, **data):
        self.log(LogLevel.SUCCESS, message, data)

    def critical(self, message: str, **data):
        self.log(LogLevel.CRITICAL, message, data)

    def set_stage(self, stage: PipelineStage):
        """Update pipeline stage and reset progress."""
        self.current_stage = stage
        self.status.stage = stage
        self.status.progress = 0.0
        self.status.current_step = ""
        self.status.gpu_info = _get_gpu_info()
        self.info(f"Entering stage: {stage.value}")
        self._save_status()

    def set_progress(self, progress: float, step: str = ""):
        """Update progress (0-100) and current step. Save status periodically."""
        self.status.progress = min(100.0, max(0.0, progress))
        self.status.current_step = step
        self.status.gpu_info = _get_gpu_info()
        now = datetime.now(timezone.utc).timestamp()
        # Save status at most every 5 seconds to avoid excessive I/O
        if now - self._last_progress_update >= 5.0:
            self._emit_event(LogLevel.INFO, f"Progress: {progress:.1f}% - {step}", {
                "progress": progress,
                "step": step,
            })
            self._save_status()
            self._last_progress_update = now

    def log_metrics_row(self, row: Dict[str, Any], **extra):
        """Persist one trainer log row verbatim to metrics.jsonl.

        Every key the trainer emits is kept: eval_loss, grad_norm, epoch, and
        the preference-method diagnostics (rewards/accuracies, rewards/margins,
        rewards/chosen) that say far more than `loss` does. status.json only
        ever holds the latest values, and events.jsonl is throttled, so this is
        the only full-resolution record of a run.
        """
        with open(self.log_files["metrics"], "a") as f:
            f.write(json.dumps({
                "wall_clock": datetime.now(timezone.utc).isoformat(),
                **row,
            }, default=str) + "\n")

        if "loss" in row:
            self.log_training_step(
                row.get("step", 0), row["loss"], row.get("learning_rate"), **extra
            )
        elif "eval_loss" in row:
            self.status.metrics["eval_loss"] = row["eval_loss"]
            self.info(f"Step {row.get('step', 0)}: eval_loss={row['eval_loss']:.4f}", **row)

    def log_training_step(self, step: int, loss: float, lr: float = None, **extra):
        """Log a training step, detect NaN, track trends, update progress."""
        metrics = {"step": step, "loss": loss}
        if lr is not None:
            metrics["lr"] = lr
        metrics.update(extra)

        self.status.metrics.update(metrics)
        self.metrics_history.append(metrics.copy())

        # NaN / explosion detection
        if loss != loss:  # NaN check
            self.critical(f"Step {step}: LOSS IS NaN! Training broken.", **metrics)
            return
        if loss > 50.0:
            self.error(f"Step {step}: Loss exploded ({loss:.2f}). Check LR or data.", **metrics)

        # Trend detection (every 10 steps)
        trend = None
        if step % 10 == 0:
            trend = _get_loss_trend(self.metrics_history, window=20)
            if trend == "plateau":
                self.warning(f"Step {step}: Loss plateau detected (last 20 steps)", **metrics)
            elif trend == "worsening":
                self.error(f"Step {step}: Loss increasing over last 20 steps", **metrics)

        # Format for console (throttle: every N steps based on config)
        msg = f"Step {step}: loss={loss:.4f}"
        if lr:
            msg += f", lr={lr:.2e}"
        if trend:
            msg += f" [{trend}]"

        if step % self._log_every_n == 0:
            self.debug(msg, **metrics)

        # Update progress in TRAIN stage
        if self.current_stage == PipelineStage.TRAIN:
            total_steps = extra.get("total_steps")
            if total_steps and total_steps > 0:
                progress = (step / total_steps) * 100
                self.set_progress(progress, f"step {step}/{total_steps}")
            self._save_status()

    def log_metric(self, name: str, value: Any):
        self.status.metrics[name] = value
        self._emit_event(LogLevel.INFO, f"Metric: {name}={value}", {
            "metric_name": name,
            "metric_value": value,
        })

    def complete(self, message: str = "Pipeline completed successfully"):
        self.status.completed_at = datetime.now(timezone.utc).isoformat()
        self.set_stage(PipelineStage.COMPLETE)
        self.success(message)
        self._save_status()

    def fail(self, error: str):
        self.status.error = error
        self.status.completed_at = datetime.now(timezone.utc).isoformat()
        self.set_stage(PipelineStage.FAILED)
        self.error(f"Pipeline failed: {error}")
        self._save_status()

    def _save_status(self):
        status_file = self.run_dir / "status.json"
        with open(status_file, "w") as f:
            json.dump({
                "run_id": self.status.run_id,
                "stage": self.status.stage.value,
                "started_at": self.status.started_at,
                "completed_at": self.status.completed_at,
                "progress": round(self.status.progress, 1),
                "current_step": self.status.current_step,
                "metrics": self.status.metrics,
                "error": self.status.error,
                "gpu_info": self.status.gpu_info,
            }, f, indent=2, default=str)

    def get_status(self) -> Dict[str, Any]:
        return {
            "run_id": self.status.run_id,
            "stage": self.status.stage.value,
            "started_at": self.status.started_at,
            "completed_at": self.status.completed_at,
            "progress": round(self.status.progress, 1),
            "current_step": self.status.current_step,
            "metrics": self.status.metrics,
            "error": self.status.error,
            "gpu_info": self.status.gpu_info,
            "log_dir": str(self.run_dir),
        }

    def subscribe(self, callback: Callable[[LogEvent], None]):
        self.subscribers.append(callback)

    def header(self, title: str):
        if self.console_enabled:
            width = 60
            if self.console_colors:
                print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * width}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.CYAN}{'═' * width}{Colors.RESET}\n")
            else:
                print(f"\n{'═' * width}")
                print(f"  {title}")
                print(f"{'═' * width}\n")

    @property
    def _log_every_n(self) -> int:
        # Throttle console logging: every 1 step at first 50, then every 10
        last_step = self.status.metrics.get("step", 0)
        if last_step < 50:
            return 1
        return 10


# ── Global instance ───────────────────────────────────────────────────────────

_logger: Optional[DistillLogger] = None


def get_logger() -> Optional[DistillLogger]:
    return _logger


def set_logger(logger: DistillLogger):
    global _logger
    _logger = logger


def create_logger(run_id: str, config) -> DistillLogger:
    logger = DistillLogger(
        run_id=run_id,
        log_dir=config.logging.dir,
        level=config.logging.level,
        console_colors=config.logging.console.colors,
        console_enabled=config.logging.console.enabled,
    )
    set_logger(logger)
    return logger


if __name__ == "__main__":
    # Self-check for analyze_training. Run: python -m lib.logger
    def _rows(train, holdout=None, subset=None, epochs=3):
        rows, per = [], max(1, len(train) // epochs)
        for i, l in enumerate(train):
            rows.append({"step": i + 1, "epoch": i // per, "loss": l})
        for j, hv in enumerate(holdout or []):
            row = {"step": (j + 1) * max(1, len(train) // len(holdout)),
                   "eval_holdout_loss": hv}
            if subset:
                row["eval_train_subset_loss"] = subset[j]
            rows.append(row)
        return rows

    # Healthy: both losses fall. The gap widens slightly because training loss
    # falls faster — that must NOT be reported as overfitting.
    ok = analyze_training(_rows([1.8, 1.6, 1.5, 1.4, 1.3, 1.25, 1.2, 1.15, 1.1],
                                holdout=[1.5, 1.35, 1.3], subset=[1.45, 1.28, 1.20]))
    assert ok["divergence_detected"] is False, ok
    assert ok["generalization_gap_widened"] is False, ok
    assert ok["holdout_regression_from_best"] == 0.0

    # Memorising: training loss collapses with a growing drop at each epoch
    # boundary while held-out loss climbs.
    bad = analyze_training(_rows([1.8, 1.7, 1.6, 1.2, 1.1, 1.0, 0.3, 0.2, 0.1],
                                 holdout=[1.40, 1.45, 1.62], subset=[1.35, 0.95, 0.35]))
    assert bad["divergence_detected"] is True, bad
    assert bad["holdout_consecutive_rises"] == 2, bad
    assert bad["generalization_gap_widened"] is True, bad
    assert bad["epoch_boundary_drops_growing"] is True, bad["epoch_boundary_drops"]
    assert bad["best_holdout_step"] == 3 and bad["holdout_regression_from_best"] > 0.2, bad

    # No held-out data: degrade without crashing, and claim nothing
    none = analyze_training(_rows([1.5, 1.4, 1.3]))
    assert none["holdout_points"] == 0 and "divergence_detected" not in none
    assert none["mean_last_10_train_loss"] is not None

    # The trailing trainer summary row has no `loss` key. The old code read
    # log_history[-1].get("loss", 0) and reported final_loss 0.0 — a perfect run.
    tail = analyze_training([{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.9},
                             {"train_runtime": 12.0}])
    assert tail["mean_last_10_train_loss"] == 0.95, tail
    # NaN steps are excluded rather than poisoning the mean
    nan = analyze_training([{"step": 1, "loss": 1.0}, {"step": 2, "loss": float("nan")},
                            {"step": 3, "loss": 0.8}])
    assert nan["train_points"] == 2, nan

    print("lib/logger.py self-check passed")
