"""
LocalDistill Logging Framework

Provides structured logging with:
- Console output with colors and progress bars
- File logging per run
- JSON event logging for dashboard
- Real-time log streaming via WebSocket
"""

import os
import sys
import json
import logging
import threading
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
    SUCCESS = "SUCCESS"  # Custom level for completed steps


class PipelineStage(Enum):
    INIT = "init"
    CURATE = "curate"
    TRAIN = "train"
    ON_POLICY = "on_policy"
    BENCHMARK = "benchmark"
    DEPLOY = "deploy"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class LogEvent:
    """Structured log event for dashboard consumption."""
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
    """Current status of a pipeline run."""
    run_id: str
    stage: PipelineStage
    started_at: str
    progress: float = 0.0  # 0-100
    current_step: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    completed_at: Optional[str] = None


# ANSI color codes
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
    WHITE = "\033[37m"
    
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


LEVEL_COLORS = {
    LogLevel.DEBUG: Colors.DIM,
    LogLevel.INFO: Colors.CYAN,
    LogLevel.WARNING: Colors.YELLOW,
    LogLevel.ERROR: Colors.RED,
    LogLevel.SUCCESS: Colors.GREEN,
}

STAGE_ICONS = {
    PipelineStage.INIT: "🔧",
    PipelineStage.CURATE: "📋",
    PipelineStage.TRAIN: "🏋️",
    PipelineStage.ON_POLICY: "🎓",
    PipelineStage.BENCHMARK: "📊",
    PipelineStage.DEPLOY: "🚀",
    PipelineStage.COMPLETE: "✅",
    PipelineStage.FAILED: "❌",
}


class DistillLogger:
    """
    Main logger for LocalDistill pipeline.
    
    Handles console output, file logging, and event streaming.
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
        
        # Event queue for dashboard streaming
        self.event_queue: Queue = Queue()
        self.subscribers: List[Callable[[LogEvent], None]] = []
        
        # Status tracking
        self.status = RunStatus(
            run_id=run_id,
            stage=PipelineStage.INIT,
            started_at=self.started_at,
        )
        
        # Setup logging directories
        self._setup_directories()
        
        # Setup file handlers
        self._setup_file_logging()
    
    def _setup_directories(self):
        """Create log directories."""
        self.run_dir = self.log_dir / "runs" / f"{datetime.now().strftime('%Y-%m-%d')}_{self.run_id[:8]}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create stage-specific log files
        self.log_files = {
            "main": self.run_dir / "run.log",
            "events": self.run_dir / "events.jsonl",
            "curate": self.run_dir / "curate.log",
            "train": self.run_dir / "train.log",
            "benchmark": self.run_dir / "benchmark.log",
            "deploy": self.run_dir / "deploy.log",
        }
    
    def _setup_file_logging(self):
        """Setup Python logging handlers."""
        self.logger = logging.getLogger(f"distill.{self.run_id[:8]}")
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()
        
        # Main log file
        file_handler = logging.FileHandler(self.log_files["main"])
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)
    
    def _format_console(self, level: LogLevel, stage: PipelineStage, message: str) -> str:
        """Format message for console output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if self.console_colors:
            color = LEVEL_COLORS.get(level, "")
            icon = STAGE_ICONS.get(stage, "")
            stage_str = f"{Colors.BOLD}[{stage.value:^10}]{Colors.RESET}"
            return f"{Colors.DIM}{timestamp}{Colors.RESET} {icon} {stage_str} {color}{message}{Colors.RESET}"
        else:
            return f"{timestamp} [{stage.value:^10}] {message}"
    
    def _emit_event(self, level: LogLevel, message: str, data: Dict[str, Any] = None):
        """Emit a log event for dashboard consumption."""
        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            stage=self.current_stage.value,
            message=message,
            run_id=self.run_id,
            data=data or {},
        )
        
        # Write to events file
        with open(self.log_files["events"], "a") as f:
            f.write(event.to_json() + "\n")
        
        # Add to status logs (keep last 100)
        self.status.logs.append(f"[{event.timestamp}] {message}")
        if len(self.status.logs) > 100:
            self.status.logs = self.status.logs[-100:]
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception:
                pass
        
        # Put in queue for streaming
        self.event_queue.put(event)
    
    def log(self, level: LogLevel, message: str, data: Dict[str, Any] = None):
        """Log a message at the specified level."""
        # Console output
        if self.console_enabled:
            print(self._format_console(level, self.current_stage, message))
        
        # File logging
        log_level = getattr(logging, level.value, logging.INFO)
        self.logger.log(log_level, f"[{self.current_stage.value}] {message}")
        
        # Event emission
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
    
    def set_stage(self, stage: PipelineStage):
        """Update the current pipeline stage."""
        self.current_stage = stage
        self.status.stage = stage
        self.info(f"Entering stage: {stage.value}")
        self._save_status()  # Save status on every stage change
    
    def set_progress(self, progress: float, step: str = ""):
        """Update progress (0-100) and current step."""
        self.status.progress = progress
        self.status.current_step = step
        self._emit_event(LogLevel.INFO, f"Progress: {progress:.1f}% - {step}", {
            "progress": progress,
            "step": step,
        })
        self._save_status()  # Save status on progress update
    
    def log_metric(self, name: str, value: Any):
        """Log a training metric."""
        self.status.metrics[name] = value
        self._emit_event(LogLevel.INFO, f"Metric: {name}={value}", {
            "metric_name": name,
            "metric_value": value,
        })
    
    def log_training_step(self, step: int, loss: float, lr: float = None, **extra):
        """Log a training step with metrics."""
        metrics = {"step": step, "loss": loss}
        if lr is not None:
            metrics["lr"] = lr
        metrics.update(extra)
        
        self.status.metrics.update(metrics)
        
        # Format for console
        msg = f"Step {step}: loss={loss:.4f}"
        if lr:
            msg += f", lr={lr:.2e}"
        
        self.debug(msg, **metrics)
        
        # Save status periodically (every 10 steps to avoid too many writes)
        if step % 10 == 0:
            self._save_status()
    
    def complete(self, message: str = "Pipeline completed successfully"):
        """Mark pipeline as complete."""
        self.status.completed_at = datetime.now(timezone.utc).isoformat()
        self.set_stage(PipelineStage.COMPLETE)
        self.success(message)
        self._save_status()
    
    def fail(self, error: str):
        """Mark pipeline as failed."""
        self.status.error = error
        self.status.completed_at = datetime.now(timezone.utc).isoformat()
        self.set_stage(PipelineStage.FAILED)
        self.error(f"Pipeline failed: {error}")
        self._save_status()
    
    def _save_status(self):
        """Save current status to file."""
        status_file = self.run_dir / "status.json"
        with open(status_file, "w") as f:
            json.dump({
                "run_id": self.status.run_id,
                "stage": self.status.stage.value,
                "started_at": self.status.started_at,
                "completed_at": self.status.completed_at,
                "progress": self.status.progress,
                "current_step": self.status.current_step,
                "metrics": self.status.metrics,
                "error": self.status.error,
            }, f, indent=2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status as dict."""
        return {
            "run_id": self.status.run_id,
            "stage": self.status.stage.value,
            "started_at": self.status.started_at,
            "completed_at": self.status.completed_at,
            "progress": self.status.progress,
            "current_step": self.status.current_step,
            "metrics": self.status.metrics,
            "error": self.status.error,
            "log_dir": str(self.run_dir),
        }
    
    def subscribe(self, callback: Callable[[LogEvent], None]):
        """Subscribe to log events."""
        self.subscribers.append(callback)
    
    def header(self, title: str):
        """Print a section header."""
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


# Global logger instance (set by orchestrator)
_logger: Optional[DistillLogger] = None


def get_logger() -> Optional[DistillLogger]:
    """Get the global logger instance."""
    return _logger


def set_logger(logger: DistillLogger):
    """Set the global logger instance."""
    global _logger
    _logger = logger


def create_logger(run_id: str, config) -> DistillLogger:
    """Create and set the global logger from config."""
    logger = DistillLogger(
        run_id=run_id,
        log_dir=config.logging.dir,
        level=config.logging.level,
        console_colors=config.logging.console.colors,
        console_enabled=config.logging.console.enabled,
    )
    set_logger(logger)
    return logger


class ProgressBar:
    """Simple progress bar for console output."""
    
    def __init__(self, total: int, desc: str = "", width: int = 40, enabled: bool = True):
        self.total = total
        self.desc = desc
        self.width = width
        self.enabled = enabled and sys.stdout.isatty()
        self.current = 0
    
    def update(self, n: int = 1):
        self.current += n
        if self.enabled:
            self._render()
    
    def _render(self):
        pct = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        print(f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({pct*100:.1f}%)", end="", flush=True)
    
    def close(self):
        if self.enabled:
            print()  # Newline after progress bar


if __name__ == "__main__":
    # Test the logger
    import uuid
    
    logger = DistillLogger(
        run_id=str(uuid.uuid4()),
        log_dir="./logs",
        level="DEBUG",
    )
    
    logger.header("LOCALDISTILL TEST RUN")
    
    logger.set_stage(PipelineStage.INIT)
    logger.info("Initializing pipeline")
    
    logger.set_stage(PipelineStage.CURATE)
    logger.info("Loading dataset", examples=1000)
    logger.set_progress(50, "Filtering examples")
    
    logger.set_stage(PipelineStage.TRAIN)
    for i in range(5):
        logger.log_training_step(i, loss=1.0 - i * 0.1, lr=2e-4)
    
    logger.set_stage(PipelineStage.BENCHMARK)
    logger.log_metric("gsm8k_accuracy", 0.65)
    
    logger.complete()
    
    print("\nStatus:", json.dumps(logger.get_status(), indent=2))
