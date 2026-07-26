#!/usr/bin/env python3
"""
LocalDistill Setup Wizard

Interactive setup with Rich TUI for:
- Prerequisites checking (Docker, GPU)
- API key configuration
- Model selection (student & teacher)
- Config generation
- Docker build
- Quick demo run

Usage:
    python setup.py              # Full interactive setup
    python setup.py --quick      # Quick setup (skip optional steps)
    python setup.py --check      # Check prerequisites only
    python setup.py --demo       # Run demo after setup
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
except ImportError:
    print("Installing rich...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DIR = Path(__file__).resolve().parent
ENV_FILE = DIR / ".env"
CONFIG_FILE = DIR / "config.yaml"

console = Console()

# API Providers
PROVIDERS = {
    "openrouter": {
        "key": "OPENROUTER_API_KEY",
        "hint": "sk-or-...",
        "url": "https://openrouter.ai/keys",
        "models": [
            ("openrouter/deepseek/deepseek-chat", "Best value, fast"),
            ("openrouter/anthropic/claude-3.5-sonnet", "High quality"),
            ("openrouter/openai/gpt-4o", "OpenAI flagship"),
            ("openrouter/google/gemini-2.0-flash-exp", "Fast, free tier"),
            ("openrouter/meta-llama/llama-3.3-70b-instruct", "Open source"),
        ],
    },
    "openai": {
        "key": "OPENAI_API_KEY",
        "hint": "sk-...",
        "url": "https://platform.openai.com/api-keys",
        "models": [
            ("openai/gpt-4o", "Flagship"),
            ("openai/gpt-4o-mini", "Fast, cheap"),
            ("openai/o1-mini", "Reasoning"),
        ],
    },
    "anthropic": {
        "key": "ANTHROPIC_API_KEY",
        "hint": "sk-ant-...",
        "url": "https://console.anthropic.com/settings/keys",
        "models": [
            ("anthropic/claude-3.5-sonnet", "Best balance"),
            ("anthropic/claude-3-haiku", "Fast"),
        ],
    },
}

# Student models for fine-tuning
STUDENT_MODELS = [
    ("unsloth/Llama-3.2-3B-Instruct", "3B", "6GB", "Fast, good for testing"),
    ("unsloth/Qwen2.5-3B-Instruct", "3B", "6GB", "Fast, multilingual"),
    ("unsloth/Phi-3.5-mini-instruct", "4B", "6GB", "Microsoft, efficient"),
    ("unsloth/Mistral-7B-Instruct-v0.3", "7B", "10GB", "Balanced"),
    ("unsloth/Llama-3.1-8B-Instruct", "8B", "12GB", "High quality"),
    ("unsloth/Qwen2.5-7B-Instruct", "7B", "10GB", "Balanced, multilingual"),
]

# Run mode presets
RUN_MODES = {
    "demo": {"examples": 50, "epochs": 1, "desc": "Quick test (~5 min)"},
    "small": {"examples": 500, "epochs": 2, "desc": "Small training (~30 min)"},
    "full": {"examples": 5000, "epochs": 3, "desc": "Full training (~3 hours)"},
}


@dataclass
class SetupState:
    """Track setup progress."""
    has_docker: bool = False
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_vram: int = 0
    provider: str = ""
    api_key: str = ""
    teacher_model: str = ""
    student_model: str = "unsloth/Llama-3.2-3B-Instruct"
    run_mode: str = "demo"
    

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_env() -> dict:
    """Load existing .env file."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def save_env(env: dict):
    """Save environment variables to .env file."""
    lines = []
    # Preserve comments from example
    if (DIR / ".env.example").exists():
        for line in (DIR / ".env.example").read_text().splitlines():
            if line.startswith("#"):
                lines.append(line)
    
    lines.append("")
    for k, v in env.items():
        if v:
            lines.append(f"{k}={v}")
    
    ENV_FILE.write_text("\n".join(lines) + "\n")


def update_config(updates: dict):
    """Update config.yaml with new values."""
    import yaml
    
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Deep merge updates
    def merge(base, update):
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                merge(base[k], v)
            else:
                base[k] = v
    
    merge(config, updates)
    
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_cmd(cmd: List[str], capture: bool = True) -> Tuple[int, str]:
    """Run a command and return (returncode, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 1, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# Setup Steps
# ═══════════════════════════════════════════════════════════════════════════════

def show_banner():
    """Display welcome banner."""
    console.print()
    console.print(Panel(
        Text("LOCALDISTILL SETUP\n\nDistill cloud LLMs into local models", 
             style="bold cyan", justify="center"),
        box=box.DOUBLE,
        width=60,
    ))
    console.print()


def check_prerequisites(state: SetupState) -> bool:
    """Check Docker, GPU, and other requirements."""
    console.print("\n[bold cyan][1/5] Checking Prerequisites[/bold cyan]\n")
    
    all_ok = True
    
    # Docker
    code, out = run_cmd(["docker", "--version"])
    if code == 0:
        version = out.strip().split()[-1] if out else "?"
        console.print(f"  [green]✓[/green] Docker {version}")
        state.has_docker = True
    else:
        console.print("  [red]✗[/red] Docker not found")
        console.print("    [dim]Install: https://docs.docker.com/get-docker/[/dim]")
        all_ok = False
    
    # Docker Compose
    code, out = run_cmd(["docker", "compose", "version"])
    if code == 0:
        console.print(f"  [green]✓[/green] Docker Compose")
    else:
        console.print("  [red]✗[/red] Docker Compose not found")
        all_ok = False
    
    # GPU
    code, out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if code == 0 and out.strip():
        parts = out.strip().split(",")
        state.gpu_name = parts[0].strip()
        try:
            mem_str = parts[1].strip().replace("MiB", "").strip()
            state.gpu_vram = int(mem_str) // 1024  # Convert to GB
        except:
            state.gpu_vram = 0
        state.has_gpu = True
        console.print(f"  [green]✓[/green] GPU: {state.gpu_name} ({state.gpu_vram}GB)")
    else:
        console.print("  [yellow]![/yellow] No NVIDIA GPU detected")
        console.print("    [dim]Training will require cloud GPU or CPU (slow)[/dim]")
    
    # Python packages
    try:
        import yaml
        console.print("  [green]✓[/green] PyYAML")
    except ImportError:
        console.print("  [yellow]![/yellow] PyYAML not installed, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], 
                      capture_output=True)
    
    # Check existing data
    if (DIR / "curated_train.jsonl").exists():
        size_mb = (DIR / "curated_train.jsonl").stat().st_size // (1024 * 1024)
        console.print(f"  [green]✓[/green] Training data: curated_train.jsonl ({size_mb}MB)")
    else:
        console.print("  [yellow]![/yellow] No training data found")
        console.print("    [dim]Will need to provide dataset or use HuggingFace[/dim]")
    
    return all_ok


def configure_api(state: SetupState):
    """Configure API provider and key."""
    console.print("\n[bold cyan][2/5] API Configuration[/bold cyan]\n")
    
    env = load_env()
    
    # Check existing keys
    configured = []
    for name, cfg in PROVIDERS.items():
        if env.get(cfg["key"]):
            configured.append(name)
    
    if configured:
        console.print(f"  [dim]Existing keys: {', '.join(configured)}[/dim]")
        if not Confirm.ask("  Configure a different provider?", default=False):
            state.provider = configured[0]
            state.api_key = env.get(PROVIDERS[configured[0]]["key"], "")
            # Still let them pick a model
            pick_teacher_model(state, configured[0])
            return
    
    # Provider selection
    console.print("  [bold]Select API provider:[/bold]")
    console.print()
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, (name, cfg) in enumerate(PROVIDERS.items(), 1):
        status = "[green](configured)[/green]" if name in configured else ""
        table.add_row(f"[cyan]{i})[/cyan]", name.title(), status)
    table.add_row(f"[dim]{len(PROVIDERS)+1})[/dim]", "Skip", "[dim](use existing)[/dim]")
    console.print(table)
    console.print()
    
    choice = IntPrompt.ask("  Select", default=1)
    
    if choice > len(PROVIDERS):
        console.print("  [dim]Skipping API configuration[/dim]")
        return
    
    provider_name = list(PROVIDERS.keys())[choice - 1]
    provider = PROVIDERS[provider_name]
    state.provider = provider_name
    
    # Get API key
    current = env.get(provider["key"], "")
    hint = f"[dim]({provider['hint']})[/dim]"
    if current:
        hint = f"[dim](Enter to keep {current[:12]}...)[/dim]"
    
    console.print(f"\n  Get key: {provider['url']}")
    key = Prompt.ask(f"  {provider['key']} {hint}", default=current)
    
    if key:
        state.api_key = key
        env[provider["key"]] = key
        save_env(env)
        console.print(f"  [green]✓[/green] API key saved")
    
    # Pick teacher model
    pick_teacher_model(state, provider_name)


def pick_teacher_model(state: SetupState, provider: str):
    """Select teacher model for on-policy distillation."""
    models = PROVIDERS[provider]["models"]
    
    console.print(f"\n  [bold]Select teacher model ({provider}):[/bold]")
    console.print()
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, (model, desc) in enumerate(models, 1):
        table.add_row(f"[cyan]{i})[/cyan]", model.split("/")[-1], f"[dim]{desc}[/dim]")
    table.add_row(f"[dim]{len(models)+1})[/dim]", "Custom", "")
    console.print(table)
    console.print()
    
    choice = IntPrompt.ask("  Select", default=1)
    
    if choice <= len(models):
        state.teacher_model = models[choice - 1][0]
    else:
        state.teacher_model = Prompt.ask("  Model ID")
    
    console.print(f"  [green]✓[/green] Teacher: {state.teacher_model}")


def configure_models(state: SetupState):
    """Select student model for fine-tuning."""
    console.print("\n[bold cyan][3/5] Model Selection[/bold cyan]\n")
    
    console.print("  [bold]Select student model (to fine-tune):[/bold]")
    console.print()
    
    # Filter models based on VRAM
    available_models = []
    for model, size, vram, desc in STUDENT_MODELS:
        vram_gb = int(vram.replace("GB", ""))
        if not state.has_gpu or state.gpu_vram >= vram_gb:
            available_models.append((model, size, vram, desc, True))
        else:
            available_models.append((model, size, vram, desc, False))
    
    table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
    table.add_column("#", style="cyan", width=3)
    table.add_column("Model", width=30)
    table.add_column("Size", width=5)
    table.add_column("VRAM", width=6)
    table.add_column("Notes", style="dim")
    
    for i, (model, size, vram, desc, fits) in enumerate(available_models, 1):
        name = model.split("/")[-1]
        style = "" if fits else "dim red"
        warn = "" if fits else " (needs more VRAM)"
        table.add_row(str(i), name, size, vram, desc + warn, style=style)
    
    console.print(table)
    console.print()
    
    # Default to first model that fits
    default = 1
    for i, (_, _, _, _, fits) in enumerate(available_models, 1):
        if fits:
            default = i
            break
    
    choice = IntPrompt.ask("  Select", default=default)
    
    if 1 <= choice <= len(available_models):
        state.student_model = available_models[choice - 1][0]
    
    console.print(f"  [green]✓[/green] Student: {state.student_model}")


def configure_run_mode(state: SetupState):
    """Select default run mode."""
    console.print("\n[bold cyan][4/5] Run Mode[/bold cyan]\n")
    
    console.print("  [bold]Select default run mode:[/bold]")
    console.print()
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, (mode, cfg) in enumerate(RUN_MODES.items(), 1):
        table.add_row(
            f"[cyan]{i})[/cyan]",
            mode.title(),
            f"{cfg['examples']} examples, {cfg['epochs']} epoch(s)",
            f"[dim]{cfg['desc']}[/dim]"
        )
    console.print(table)
    console.print()
    
    choice = IntPrompt.ask("  Select", default=1)
    
    mode_names = list(RUN_MODES.keys())
    if 1 <= choice <= len(mode_names):
        state.run_mode = mode_names[choice - 1]
    
    console.print(f"  [green]✓[/green] Mode: {state.run_mode}")


def build_and_configure(state: SetupState):
    """Build Docker images and write config."""
    console.print("\n[bold cyan][5/5] Setup[/bold cyan]\n")
    
    # Update config.yaml
    import yaml
    
    mode_cfg = RUN_MODES[state.run_mode]
    config_updates = {
        "run_mode": state.run_mode,
        "models": {
            "student": state.student_model,
            "teacher": state.teacher_model or "openrouter/deepseek/deepseek-chat",
        },
        "curation": {
            "max_examples": mode_cfg["examples"],
        },
        "training": {
            "hyperparams": {
                "epochs": mode_cfg["epochs"],
            },
        },
    }
    
    update_config(config_updates)
    console.print("  [green]✓[/green] config.yaml updated")
    
    # Create directories
    for d in ["logs", "adapters", "data", "logs/runs"]:
        (DIR / d).mkdir(parents=True, exist_ok=True)
    console.print("  [green]✓[/green] Directories created")
    
    # Build Docker images
    if state.has_docker:
        if Confirm.ask("  Build Docker images?", default=True):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Building monitor...", total=None)
                run_cmd(["docker", "compose", "build", "monitor"])
                progress.update(task, description="Building trainer...")
                run_cmd(["docker", "compose", "build", "trainer"])
                progress.update(task, description="Done!")
            
            console.print("  [green]✓[/green] Docker images built")
    
    # Summary
    console.print()


def show_summary(state: SetupState):
    """Show setup summary and next steps."""
    mode_cfg = RUN_MODES[state.run_mode]
    
    summary = f"""
[bold]Configuration Summary[/bold]

  Student Model:  {state.student_model}
  Teacher Model:  {state.teacher_model or 'Not configured'}
  Run Mode:       {state.run_mode} ({mode_cfg['examples']} examples, {mode_cfg['epochs']} epochs)
  GPU:            {state.gpu_name or 'None'} ({state.gpu_vram}GB)

[bold]Quick Start[/bold]

  [cyan]./distill run[/cyan]              Run training pipeline ({state.run_mode} mode)
  [cyan]./distill run --mode full[/cyan]  Full training (5000 examples)
  [cyan]./distill monitor[/cyan]          Start dashboard at http://localhost:8080
  [cyan]./distill status[/cyan]           Check run status
  [cyan]./distill logs[/cyan]             View training logs

[bold]Files[/bold]

  config.yaml     Main configuration (edit to customize)
  .env            API keys
  adapters/       Trained model outputs
  logs/           Run logs
"""
    
    console.print(Panel(summary, title="[green bold]SETUP COMPLETE", width=70))


def run_demo(state: SetupState):
    """Optionally run a demo."""
    if Confirm.ask("\n  Run quick demo now?", default=False):
        console.print("\n  [bold]Starting demo run...[/bold]\n")
        os.chdir(DIR)
        subprocess.run([sys.executable, "distill.py", "run", "--mode", "demo"])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the setup wizard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LocalDistill Setup Wizard")
    parser.add_argument("--quick", action="store_true", help="Quick setup (skip optional)")
    parser.add_argument("--check", action="store_true", help="Check prerequisites only")
    parser.add_argument("--demo", action="store_true", help="Run demo after setup")
    args = parser.parse_args()
    
    state = SetupState()
    
    show_banner()
    
    # Check prerequisites
    ok = check_prerequisites(state)
    
    if args.check:
        sys.exit(0 if ok else 1)
    
    if not ok and not Confirm.ask("\n  Continue anyway?", default=False):
        console.print("\n  [dim]Setup cancelled.[/dim]\n")
        sys.exit(1)
    
    # Run setup steps
    configure_api(state)
    configure_models(state)
    
    if not args.quick:
        configure_run_mode(state)
    
    build_and_configure(state)
    show_summary(state)
    
    # Optional demo
    if args.demo:
        run_demo(state)
    elif Confirm.ask("\n  Run quick demo now?", default=False):
        run_demo(state)


if __name__ == "__main__":
    main()
