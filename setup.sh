#!/usr/bin/env bash
#  ██╗      ██████╗  ██████╗ █████╗ ██╗     ██████╗ ██╗███████╗████████╗██╗██╗     ██╗
#  ██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔══██╗██║██╔════╝╚══██╔══╝██║██║     ██║
#  ██║     ██║   ██║██║     ███████║██║     ██║  ██║██║███████╗   ██║   ██║██║     ██║
#  ██║     ██║   ██║██║     ██╔══██║██║     ██║  ██║██║╚════██║   ██║   ██║██║     ██║
#  ███████╗╚██████╔╝╚██████╗██║  ██║███████╗██████╔╝██║███████║   ██║   ██║███████╗███████╗
#  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝╚══════╝
#
#  Capture → Curate → Train Your Own Model
#  https://github.com/vishrutsingh/localdistill

set -euo pipefail
shopt -s inherit_errexit

# ═══════════════════════════════════════════════════════════════
# Colors & styling
# ═══════════════════════════════════════════════════════════════

if [[ -t 1 ]]; then
  BOLD='\033[1m'
  DIM='\033[2m'
  GREEN='\033[32m'
  YELLOW='\033[33m'
  RED='\033[31m'
  BLUE='\033[34m'
  CYAN='\033[36m'
  NC='\033[0m'
else
  BOLD='' DIM='' GREEN='' YELLOW='' RED='' BLUE='' CYAN='' NC=''
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STEP=0
TOTAL_STEPS=7
HAS_GPU=false
SKIP_BUILD=false
SKIP_MCP=false

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

_step() {
  STEP=$((STEP + 1))
  echo ""
  printf "  ${BLUE}${BOLD}[%d/%d]${NC} ${BOLD}%s${NC}\n" "$STEP" "$TOTAL_STEPS" "$1"
  echo "  ${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

_ok()   { printf "    ${GREEN}✓${NC} %s\n" "$1"; }
_warn() { printf "    ${YELLOW}⚠${NC}  %s\n" "$1"; }
_err()  { printf "    ${RED}✗${NC} %s\n" "$1"; }
_info() { printf "    ${DIM}→${NC} %s\n" "$1"; }

_check_cmd() {
  local cmd="$1" label="${2:-$1}"
  if command -v "$cmd" &>/dev/null; then
    local ver; ver=$("$cmd" --version 2>&1 | head -1 | tr '\n' ' ')
    _ok "$label ${DIM}(${ver:0:60})${NC}"
    return 0
  else
    _err "$label not found — please install it first"
    return 1
  fi
}

_prompt() {
  local var="$1" prompt="$2" default="${3:-}"
  local hint=""
  [[ -n "$default" ]] && hint=" [${DIM}${default}${NC}]"
  
  printf "    ${CYAN}?${NC} %s${hint}: " "$prompt"
  read -r value
  value="${value:-$default}"
  
  if [[ -n "$value" ]]; then
    export "$var"="$value"
    _ok "$var set"
  fi
}

_done_box() {
  echo ""
  echo "  ╔══════════════════════════════════════════════════════════╗"
  echo "  ║                    ${GREEN}${BOLD}LOCALDISTILL READY${NC}                   ║"
  echo "  ╠══════════════════════════════════════════════════════════╣"
  echo "  ║                                                          ║"
  printf "  ║  ${BOLD}Proxy:${NC}     http://localhost:8787${NC}                         ║\n"
  printf "  ║  ${BOLD}Dashboard:${NC} http://localhost:8000${NC}                         ║\n"
  echo "  ║                                                          ║"
  printf "  ║  ${BOLD}Train:${NC}         ./train.sh${NC}                                ║\n"
  printf "  ║  ${BOLD}Cloud train:${NC}   ./scripts/cloud_train.sh runpod${NC}          ║\n"
  printf "  ║  ${BOLD}View logs:${NC}     docker compose logs -f proxy${NC}             ║\n"
  printf "  ║  ${BOLD}Stop:${NC}          docker compose down${NC}                      ║\n"
  echo "  ║                                                          ║"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    printf "  ║  ${YELLOW}⚠  OPENAI_API_KEY missing — proxy won't route calls${NC}     ║\n"
    printf "  ║  ${DIM}Add to .env then: docker compose restart proxy${NC}           ║\n"
  fi
  echo "  ║                                                          ║"
  echo "  ╚══════════════════════════════════════════════════════════╝"
  echo ""
}

_spinner() {
  local pid=$1 msg="$2"
  local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r    ${DIM}%s${NC} %s" "${spin:$i:1}" "$msg"
    i=$(( (i + 1) % ${#spin} ))
    sleep 0.1
  done
  wait "$pid"
  printf "\r    ${GREEN}✓${NC} %s ${DIM}(done)${NC}\n" "$msg"
}

_cleanup() {
  echo ""
  echo "  ${RED}Setup interrupted.${NC}"
  echo "  ${DIM}No changes were made to running services.${NC}"
  exit 1
}
trap _cleanup INT TERM

# ═══════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════

clear 2>/dev/null || true
echo ""
echo "  ${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo "  ${BOLD}${CYAN}║${NC}                    ${BOLD}LOCALDISTILL${NC}                          ${BOLD}${CYAN}║${NC}"
echo "  ${BOLD}${CYAN}║${NC}         ${DIM}Capture → Curate → Train Your Own Model${NC}         ${BOLD}${CYAN}║${NC}"
echo "  ${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Parse flags ──

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true; shift ;;
    --no-mcp)     SKIP_MCP=true; shift ;;
    --help|-h)
      echo "  Usage: ./setup.sh [--skip-build] [--no-mcp]"
      echo ""
      echo "  Options:"
      echo "    --skip-build   Skip Docker image build (use if already built)"
      echo "    --no-mcp       Skip Hermes MCP registration"
      exit 0
      ;;
    *) echo "  Unknown flag: $1"; exit 1 ;;
  esac
done

# ═══════════════════════════════════════════════════════════════
# Step 1: Prerequisites
# ═══════════════════════════════════════════════════════════════

_step "Checking prerequisites"

_check_cmd docker || exit 1

if docker compose version &>/dev/null; then
  _ok "docker compose $(docker compose version --short 2>/dev/null || echo '')"
elif docker-compose --version &>/dev/null; then
  _ok "docker-compose $(docker-compose --version | grep -oP '\d+\.\d+\.\d+')"
else
  _err "docker compose not found"
  exit 1
fi

_check_cmd git

if nvidia-smi &>/dev/null; then
  HAS_GPU=true
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU")
  gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || echo "?")
  _ok "NVIDIA GPU detected: ${gpu_name} (${gpu_mem})"
else
  _warn "No NVIDIA GPU detected — training will need --cloud flag"
fi

# ═══════════════════════════════════════════════════════════════
# Step 2: Environment
# ═══════════════════════════════════════════════════════════════

_step "Configuring environment"

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  _ok ".env created from .env.example"
else
  _ok ".env already exists (skipped)"
fi

# Inline API key prompts
echo ""
echo "  ${BOLD}API Keys${NC}"
echo "  ${DIM}─────────────────────────────────────────────────${NC}"

# Load any existing keys
source .env 2>/dev/null || true
HAD_OPENAI="${OPENAI_API_KEY:-}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  printf "    ${CYAN}?${NC} OpenAI API key ${DIM}(sk-...) [skip]${NC}: "
  read -r key
  if [[ -n "$key" ]]; then
    sed -i "/^OPENAI_API_KEY=/d" .env 2>/dev/null || true
    echo "OPENAI_API_KEY=$key" >> .env
    export OPENAI_API_KEY="$key"
    _ok "OpenAI API key saved"
  else
    _warn "OPENAI_API_KEY not set — proxy won't route API calls"
  fi
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  _ok "OpenAI API key ${DIM}(already configured)${NC}"
fi

printf "    ${CYAN}?${NC} Anthropic API key ${DIM}(sk-ant-...) [skip]${NC}: "
read -r ant_key
if [[ -n "$ant_key" ]]; then
  sed -i "/^ANTHROPIC_API_KEY=/d" .env 2>/dev/null || true
  echo "ANTHROPIC_API_KEY=$ant_key" >> .env
  _ok "Anthropic API key saved"
else
  _info "Anthropic API key skipped"
fi

# ═══════════════════════════════════════════════════════════════
# Step 3: Build images
# ═══════════════════════════════════════════════════════════════

_step "Building Docker images"

if $SKIP_BUILD; then
  _info "Skipping build (--skip-build)"
else
  echo ""
  docker compose build proxy 2>&1 &
  _spinner $! "Building proxy image..."
  
  docker compose build api 2>&1 &
  _spinner $! "Building api image..."
  
  if $HAS_GPU; then
    docker compose --profile training build trainer 2>&1 &
    _spinner $! "Building trainer image (GPU)..."
  else
    _info "Skipping trainer build (no GPU)"
  fi
fi

# ═══════════════════════════════════════════════════════════════
# Step 4: Initialize database
# ═══════════════════════════════════════════════════════════════

_step "Initializing database"

mkdir -p data
if [[ ! -f "data/localdistill.db" ]]; then
  docker compose run --rm --entrypoint python proxy -c "
from db import init_db
init_db()
print('Database initialized')
" 2>&1 | tail -1
  _ok "Database created at ./data/localdistill.db"
else
  _ok "Database already exists at ./data/localdistill.db"
fi

# ═══════════════════════════════════════════════════════════════
# Step 5: Start services
# ═══════════════════════════════════════════════════════════════

_step "Starting services"

docker compose up -d proxy api 2>&1 &
_spinner $! "Starting proxy + api..."

# Health check
sleep 3
if curl -sf http://localhost:8787/health >/dev/null 2>&1; then
  _ok "proxy → http://localhost:8787 (healthy)"
else
  _warn "proxy health check failed — check: docker compose logs proxy"
fi

if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  _ok "api   → http://localhost:8000 (healthy)"
else
  _warn "api health check failed — check: docker compose logs api"
fi

# ═══════════════════════════════════════════════════════════════
# Step 6: Detect MCP clients
# ═══════════════════════════════════════════════════════════════

_step "Detecting MCP clients"

MCP_CMD="python $SCRIPT_DIR/mcp/mcp_server.py"
MCP_CONFIG='{
  "mcpServers": {
    "localdistill": {
      "command": "python",
      "args": ["'"$SCRIPT_DIR"'/mcp/mcp_server.py"]
    }
  }
}'

DETECTED_CLIENTS=()
command -v hermes  &>/dev/null && DETECTED_CLIENTS+=("hermes")
[[ -f "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ]] && DETECTED_CLIENTS+=("claude-desktop")
[[ -f "$HOME/.vscode/mcp.json" ]] && DETECTED_CLIENTS+=("vscode")
command -v cursor  &>/dev/null && DETECTED_CLIENTS+=("cursor")
command -v claude  &>/dev/null && DETECTED_CLIENTS+=("claude-code")
command -v codex   &>/dev/null && DETECTED_CLIENTS+=("codex")

if [[ ${#DETECTED_CLIENTS[@]} -gt 0 ]]; then
  echo ""
  echo "  ${BOLD}Detected MCP-compatible tools:${NC}"
  for i in "${!DETECTED_CLIENTS[@]}"; do
    printf "    ${GREEN}%d)${NC} %s\n" "$((i+1))" "${DETECTED_CLIENTS[$i]}"
  done
  printf "    ${DIM}%d)${NC} Other (print config for manual setup)\n" "$((${#DETECTED_CLIENTS[@]}+1))"
  printf "    ${DIM}%d)${NC} Skip MCP setup\n" "$((${#DETECTED_CLIENTS[@]}+2))"
  echo ""
  printf "    ${CYAN}?${NC} Select tool to configure ${DIM}[1]${NC}: "
  read -r choice
  choice="${choice:-1}"
else
  _info "No MCP-compatible tools detected"
  choice=""
fi

# ═══════════════════════════════════════════════════════════════
# Step 7: Configure MCP
# ═══════════════════════════════════════════════════════════════

_step "Configuring MCP server"

configured=false

_mcp_done() {
  configured=true
  _ok "$1 MCP configured — tools: signal_quality, get_last_conversation, search_knowledge, add_to_rag, get_training_status"
}

_print_manual() {
  echo ""
  echo "  ${YELLOW}Add this to your MCP client config:${NC}"
  echo ""
  echo "$MCP_CONFIG"
  echo ""
}

case "${DETECTED_CLIENTS[$((choice-1))]:-}" in
  hermes)
    if hermes mcp list 2>/dev/null | grep -q "localdistill"; then
      _ok "Already registered with Hermes"
    else
      hermes mcp add localdistill --command "$MCP_CMD" 2>&1 && _mcp_done "Hermes" || _print_manual
    fi
    ;;
  claude-desktop)
    # Merge into Claude Desktop config
    cf="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    if [[ -f "$cf" ]]; then
      python3 -c "
import json, sys
with open('$cf') as f: cfg = json.load(f)
cfg.setdefault('mcpServers', {})['localdistill'] = {'command': 'python', 'args': ['$SCRIPT_DIR/mcp/mcp_server.py']}
with open('$cf', 'w') as f: json.dump(cfg, f, indent=2)
print('ok')
" 2>/dev/null && _mcp_done "Claude Desktop" || _print_manual
    else
      _warn "Claude Desktop config not found at $cf"
      _print_manual
    fi
    ;;
  vscode)
    vf="$HOME/.vscode/mcp.json"
    python3 -c "
import json, os
cfg = {}
if os.path.exists('$vf'):
    with open('$vf') as f: cfg = json.load(f)
cfg.setdefault('mcpServers', {})['localdistill'] = {'command': 'python', 'args': ['$SCRIPT_DIR/mcp/mcp_server.py']}
os.makedirs(os.path.dirname('$vf'), exist_ok=True)
with open('$vf', 'w') as f: json.dump(cfg, f, indent=2)
print('ok')
" 2>/dev/null && _mcp_done "VS Code" || _print_manual
    ;;
  cursor)
    cf="$HOME/.cursor/mcp.json"
    python3 -c "
import json, os
cfg = {}
if os.path.exists('$cf'):
    with open('$cf') as f: cfg = json.load(f)
cfg.setdefault('mcpServers', {})['localdistill'] = {'command': 'python', 'args': ['$SCRIPT_DIR/mcp/mcp_server.py']}
os.makedirs(os.path.dirname('$cf'), exist_ok=True)
with open('$cf', 'w') as f: json.dump(cfg, f, indent=2)
print('ok')
" 2>/dev/null && _mcp_done "Cursor" || _print_manual
    ;;
  claude-code)
    cf="$HOME/.claude/mcp.json"
    python3 -c "
import json, os
cfg = {}
if os.path.exists('$cf'):
    with open('$cf') as f: cfg = json.load(f)
cfg.setdefault('mcpServers', {})['localdistill'] = {'command': 'python', 'args': ['$SCRIPT_DIR/mcp/mcp_server.py']}
os.makedirs(os.path.dirname('$cf'), exist_ok=True)
with open('$cf', 'w') as f: json.dump(cfg, f, indent=2)
print('ok')
" 2>/dev/null && _mcp_done "Claude Code" || _print_manual
    ;;
  codex)
    cf="$HOME/.codex/mcp.json"
    python3 -c "
import json, os
cfg = {}
if os.path.exists('$cf'):
    with open('$cf') as f: cfg = json.load(f)
cfg.setdefault('mcpServers', {})['localdistill'] = {'command': 'python', 'args': ['$SCRIPT_DIR/mcp/mcp_server.py']}
os.makedirs(os.path.dirname('$cf'), exist_ok=True)
with open('$cf', 'w') as f: json.dump(cfg, f, indent=2)
print('ok')
" 2>/dev/null && _mcp_done "Codex" || _print_manual
    ;;
  *)
    _info "MCP setup skipped"
    _print_manual
    ;;
esac

if ! $configured && [[ -n "${choice:-}" ]] && [[ "$choice" != "$((${#DETECTED_CLIENTS[@]}+2))" ]]; then
  _print_manual
fi

# ═══════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════

_done_box