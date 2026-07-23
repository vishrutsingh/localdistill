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
TOTAL_STEPS=6
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
# Step 6: MCP registration
# ═══════════════════════════════════════════════════════════════

_step "Registering MCP server"

if $SKIP_MCP; then
  _info "Skipping MCP registration (--no-mcp)"
elif command -v hermes &>/dev/null; then
  if hermes mcp list 2>/dev/null | grep -q "localdistill"; then
    _ok "MCP server already registered with Hermes"
  else
    hermes mcp add localdistill --command "python $SCRIPT_DIR/mcp/mcp_server.py" 2>&1 || true
    _ok "localdistill-curator registered with Hermes"
    _info "Tools: signal_quality, get_last_conversation, search_knowledge, add_to_rag, get_training_status"
  fi
else
  _warn "Hermes CLI not found — MCP not registered"
  _info "To add manually: hermes mcp add localdistill --command \"python $(pwd)/mcp/mcp_server.py\""
fi

# ═══════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════

_done_box