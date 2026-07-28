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
