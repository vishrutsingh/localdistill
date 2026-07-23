#!/usr/bin/env python3
"""localdistill — one-command setup. Rich TUI, auto-detect, skip configured steps."""

import os, sys, subprocess, re, json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text

C = Console()
DIR = Path(__file__).resolve().parent
ENV = DIR / ".env"

PROVIDERS = {
    "openai":    {"key": "OPENAI_API_KEY",    "hint": "sk-...",     "models": [
        ("openai/gpt-4o", "flagship"),
        ("openai/gpt-4o-mini", "fast, cheap"),
        ("openai/o3-mini", "reasoning")]},
    "anthropic": {"key": "ANTHROPIC_API_KEY",  "hint": "sk-ant-...", "models": [
        ("anthropic/claude-sonnet-4", "best balance"),
        ("anthropic/claude-haiku-4", "fast"),
        ("anthropic/claude-opus-4", "strongest")]},
    "openrouter":{"key": "OPENROUTER_API_KEY", "hint": "sk-or-...",  "models": [
        ("deepseek/deepseek-v4-pro", "best all-around"),
        ("anthropic/claude-sonnet-4", ""),
        ("openai/gpt-4o", ""),
        ("google/gemini-2.5-flash", ""),
        ("meta-llama/llama-4-maverick", "")]},
}

TRAINING_MODELS = [
    ("unsloth/Llama-3.2-3B-Instruct", "3B", "fast"),
    ("unsloth/Qwen2.5-3B-Instruct", "3B", "fast"),
    ("unsloth/Phi-3.5-mini-instruct", "4B", "fast"),
    ("unsloth/Mistral-7B-Instruct-v0.3", "7B", "balanced"),
    ("unsloth/Llama-3.1-8B-Instruct", "8B", "balanced"),
    ("unsloth/Qwen2.5-7B-Instruct", "7B", "balanced"),
    ("unsloth/Llama-3.3-70B-Instruct", "70B", "cloud"),
]

MCP_CLIENTS = {
    "hermes":     lambda: os.system("hermes mcp add localdistill --command 'python %s' 2>/dev/null" % (DIR/"mcp"/"mcp_server.py")),
    "claude-desktop": lambda: _write_mcp_json(Path.home()/"Library/Application Support/Claude/claude_desktop_config.json"),
    "claude-code":    lambda: _write_mcp_json(Path.home()/".claude"/"mcp.json"),
    "vscode":         lambda: _write_mcp_json(Path.home()/".vscode"/"mcp.json"),
    "cursor":         lambda: _write_mcp_json(Path.home()/".cursor"/"mcp.json"),
    "codex":          lambda: _write_mcp_json(Path.home()/".codex"/"mcp.json"),
}

MCP_JSON = {"mcpServers": {"localdistill": {"command": "python", "args": [str(DIR/"mcp"/"mcp_server.py")]}}}

def _write_mcp_json(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(path.read_text()) if path.exists() else {}
    cfg.setdefault("mcpServers", {})["localdistill"] = MCP_JSON["mcpServers"]["localdistill"]
    path.write_text(json.dumps(cfg, indent=2))

# ── env helpers ──
def load_env():
    env = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v.strip('"').strip("'")
    return env

def save_env(env):
    lines = [f"{k}={v}" for k, v in env.items()]
    ENV.write_text("\n".join(lines) + "\n")

# ── banner ──
def banner():
    C.print()
    C.print(Panel(Text("LOCALDISTILL\nCapture → Curate → Train Your Own Model", style="bold cyan", justify="center"), width=58))
    C.print()

# ── fast-path ──
def fast_path(env):
    has_keys = any(env.get(p["key"]) for p in PROVIDERS.values())
    has_api = bool(env.get("LOCALDISTILL_API_MODEL"))
    has_train = bool(env.get("LOCALDISTILL_MODEL"))
    if not (has_keys and has_api and has_train):
        return False
    
    C.print("[green bold]Everything configured.[/]")
    C.print(f"  API model:  {env['LOCALDISTILL_API_MODEL']}")
    C.print(f"  Training:   {env['LOCALDISTILL_MODEL']}")
    if Confirm.ask("Start services?", default=True):
        subprocess.run(["docker", "compose", "build", "proxy", "api"], check=False)
        subprocess.run(["docker", "compose", "up", "-d", "proxy", "api"], check=False)
        C.print("[green]✓ Proxy: http://localhost:8787  Dashboard: http://localhost:8000[/]")
    return True

# ── step 1: prerequisites ──
def check_prereqs():
    C.print("\n[bold][1/6] Checking prerequisites[/]")
    for cmd in ["docker", "git"]:
        r = subprocess.run(["which", cmd], capture_output=True)
        if r.returncode == 0:
            C.print(f"  [green]✓[/] {cmd}")
        else:
            C.print(f"  [red]✗[/] {cmd} not found")
            sys.exit(1)
    try:
        subprocess.run(["docker", "compose", "version"], capture_output=True, check=True)
        C.print("  [green]✓[/] docker compose")
    except:
        C.print("  [red]✗[/] docker compose not found")
        sys.exit(1)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True)
    if gpu.returncode == 0:
        C.print(f"  [green]✓[/] GPU: {gpu.stdout.decode().strip().split(chr(10))[0]}")
        return True
    C.print("  [dim]→ No GPU — training will need --cloud[/]")
    return False

# ── step 2: provider + model ──
def configure_provider(env):
    C.print("\n[bold][2/6] Configure API provider[/]")
    
    configured = [p for p, cfg in PROVIDERS.items() if env.get(cfg["key"])]
    
    if len(configured) == 1:
        p = configured[0]
        C.print(f"  [dim]Using {p} (key detected)[/]")
        pick_model(env, p)
        return
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, p in enumerate(PROVIDERS, 1):
        tag = " [dim](configured)[/]" if p in configured else ""
        table.add_row(f"[green]{i})[/] {p}{tag}")
    table.add_row(f"[dim]{len(PROVIDERS)+1})[/] Skip")
    C.print(table)
    
    choice = IntPrompt.ask("Pick", default=1, choices=[str(i) for i in range(1, len(PROVIDERS)+3)])
    if choice <= len(PROVIDERS):
        p = list(PROVIDERS)[choice-1]
        cfg = PROVIDERS[p]
        new_key = Prompt.ask(f"  {cfg['key']} ({cfg['hint']})", default=env.get(cfg["key"], ""))
        if new_key.strip():
            env[cfg["key"]] = new_key.strip()
        pick_model(env, p)

def pick_model(env, provider):
    cfg = PROVIDERS[provider]
    current = env.get("LOCALDISTILL_API_MODEL", "")
    models = cfg["models"]
    
    if current:
        for i, (mid, _) in enumerate(models):
            if mid == current:
                C.print(f"  [dim]Model: {mid} (already set, Enter to keep)[/]")
                if not Confirm.ask("  Change?"):
                    return
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, (mid, desc) in enumerate(models, 1):
        marker = " [dim]▶[/]" if mid == current else ""
        tag = f" [dim]({desc})[/]" if desc else ""
        table.add_row(f"[green]{i})[/] {mid}{marker}{tag}")
    table.add_row(f"[dim]{len(models)+1})[/] Custom ID")
    C.print(f"\n  Select {provider} model:")
    C.print(table)
    
    c = IntPrompt.ask("Pick", default=1)
    if 1 <= c <= len(models):
        env["LOCALDISTILL_API_MODEL"] = models[c-1][0]
    elif c == len(models) + 1:
        env["LOCALDISTILL_API_MODEL"] = Prompt.ask("Model ID")
    
    save_env(env)
    C.print(f"  [green]✓[/] {env['LOCALDISTILL_API_MODEL']}")

# ── step 3: training model ──
def configure_training(env, has_gpu):
    C.print("\n[bold][3/6] Select base model for fine-tuning[/]")
    current = env.get("LOCALDISTILL_MODEL", "")
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, (hf, size, tier) in enumerate(TRAINING_MODELS, 1):
        marker = " [dim]▶ currently selected[/]" if hf == current else ""
        tag = {"fast": "⚡ fast", "balanced": "⚖ balanced", "cloud": "☁ cloud"}[tier]
        table.add_row(f"[green]{i})[/] {hf.split('/')[-1]:35} [dim]{size:4}[/] {tag}{marker}")
    table.add_row(f"[dim]{len(TRAINING_MODELS)+1})[/] Custom HuggingFace ID")
    C.print(table)
    
    default = next((i+1 for i, (hf, _, _) in enumerate(TRAINING_MODELS) if hf == current), 1)
    c = IntPrompt.ask("Select", default=default)
    
    if 1 <= c <= len(TRAINING_MODELS):
        selected, size, tier = TRAINING_MODELS[c-1]
    elif c == len(TRAINING_MODELS) + 1:
        selected = Prompt.ask("HuggingFace model ID")
        size, tier = "?", "custom"
    else:
        selected, size, tier = TRAINING_MODELS[0]
    
    env["LOCALDISTILL_MODEL"] = selected
    save_env(env)
    C.print(f"  [green]✓[/] {selected} ({size})")
    
    if tier == "cloud":
        C.print("  [yellow]⚠ Cloud-only — needs A100. Use ./scripts/cloud_train.sh[/]")
    elif not has_gpu:
        C.print("  [yellow]⚠ No GPU — use ./scripts/cloud_train.sh[/]")

# ── step 4: build ──
def build(has_gpu):
    C.print("\n[bold][4/6] Building Docker images[/]")
    subprocess.run(["docker", "compose", "build", "proxy", "api"], check=False)
    C.print("  [green]✓[/] proxy + api built")
    if has_gpu:
        subprocess.run(["docker", "compose", "--profile", "training", "build", "trainer"], check=False)
        C.print("  [green]✓[/] trainer built")

# ── step 5: db + start ──
def start_services():
    C.print("\n[bold][5/6] Initializing & starting[/]")
    subprocess.run(["mkdir", "-p", str(DIR/"data")], check=False)
    if not (DIR/"data"/"localdistill.db").exists():
        subprocess.run(["docker", "compose", "run", "--rm", "--entrypoint", "python", "proxy",
                        "-c", "from db import init_db; init_db()"], check=False)
    C.print("  [green]✓[/] Database ready")
    subprocess.run(["docker", "compose", "up", "-d", "proxy", "api"], check=False)
    C.print("  [green]✓[/] Proxy: http://localhost:8787")
    C.print("  [green]✓[/] Dashboard: http://localhost:8000")

# ── step 6: MCP ──
def configure_mcp():
    C.print("\n[bold][6/6] Configure MCP server[/]")
    detected = [c for c in MCP_CLIENTS if c == "hermes" and subprocess.run(["which", "hermes"], capture_output=True).returncode == 0]
    if not detected:
        C.print("  [dim]No MCP clients detected. Config:[/]")
        C.print(json.dumps(MCP_JSON, indent=2))
        return
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, c in enumerate(detected, 1):
        table.add_row(f"[green]{i})[/] {c}")
    table.add_row(f"[dim]{len(detected)+1})[/] Skip")
    C.print(table)
    
    c = IntPrompt.ask("Select", default=1)
    if 1 <= c <= len(detected):
        MCP_CLIENTS[detected[c-1]]()
        C.print(f"  [green]✓[/] {detected[c-1]} configured")

# ── done ──
def done_box(env):
    m = env.get("LOCALDISTILL_API_MODEL", "?")
    t = env.get("LOCALDISTILL_MODEL", "?")
    C.print()
    C.print(Panel(f"Proxy:  http://localhost:8787\nDashboard: http://localhost:8000\nModel: {m}\nTraining: {t}\n\nTrain: ./train.sh\nCloud:  ./scripts/cloud_train.sh runpod\nLogs:   docker compose logs -f proxy\nStop:   docker compose down", title="[green bold]READY", width=58))

# ── main ──
def main():
    banner()
    env = load_env()
    
    if fast_path(env):
        return
    
    has_gpu = check_prereqs()
    configure_provider(env)
    configure_training(env, has_gpu)
    build(has_gpu)
    start_services()
    configure_mcp()
    done_box(env)

if __name__ == "__main__":
    main()