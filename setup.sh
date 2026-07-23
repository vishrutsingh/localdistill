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
  BOLD=$(tput bold 2>/dev/null || echo '')
  DIM=$(tput dim 2>/dev/null || echo '')
  GREEN=$(tput setaf 2 2>/dev/null || echo '')
  YELLOW=$(tput setaf 3 2>/dev/null || echo '')
  RED=$(tput setaf 1 2>/dev/null || echo '')
  BLUE=$(tput setaf 4 2>/dev/null || echo '')
  CYAN=$(tput setaf 6 2>/dev/null || echo '')
  NC=$(tput sgr0 2>/dev/null || echo '')
else
  BOLD='' DIM='' GREEN='' YELLOW='' RED='' BLUE='' CYAN='' NC=''
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STEP=0
TOTAL_STEPS=8
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
  source .env 2>/dev/null || true
  echo ""
  echo "  ╔══════════════════════════════════════════════════════════╗"
  echo "  ║                    ${GREEN}${BOLD}LOCALDISTILL READY${NC}                   ║"
  echo "  ╠══════════════════════════════════════════════════════════╣"
  echo "  ║                                                          ║"
  printf "  ║  ${BOLD}Proxy:${NC}     http://localhost:8787${NC}                         ║\n"
  printf "  ║  ${BOLD}Dashboard:${NC} http://localhost:8000${NC}                         ║\n"
  printf "  ║  ${BOLD}Model:${NC}      %-45s ${NC}║\n" "${LOCALDISTILL_MODEL:-unsloth/Llama-3.2-3B-Instruct}"
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

# Fast-path: everything already configured
source .env 2>/dev/null || true
if [[ -n "${OPENAI_API_KEY:-}${ANTHROPIC_API_KEY:-}${OPENROUTER_API_KEY:-}" ]] && \
   [[ -n "${LOCALDISTILL_API_MODEL:-}" ]] && [[ -n "${LOCALDISTILL_MODEL:-}" ]]; then
  echo ""
  echo "  ${GREEN}${BOLD}Everything configured.${NC}"
  echo "  ${DIM}─────────────────────────────────────────────────${NC}"
  echo "    API model:   ${LOCALDISTILL_API_MODEL}"
  echo "    Provider:    ${OPENAI_API_KEY:+OpenAI }${ANTHROPIC_API_KEY:+Anthropic }${OPENROUTER_API_KEY:+OpenRouter}"
  echo "    Training:    ${LOCALDISTILL_MODEL}"
  echo ""
  printf "    ${CYAN}?${NC} Start services? ${DIM}[Y/n]${NC}: "; read -r go
  [[ "$go" =~ ^[Nn] ]] && { echo "  Exiting."; exit 0; }
  docker compose build proxy api 2>&1 | tail -1
  docker compose up -d proxy api 2>&1 | tail -1
  echo "  ${GREEN}✓${NC} Proxy: http://localhost:8787  Dashboard: http://localhost:8000"
  exit 0
fi

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

source .env 2>/dev/null || true

echo ""
echo "  ${BOLD}Select providers to configure:${NC}"
echo ""

# Check existing keys
OPENAI_SET=false; ANTHROPIC_SET=false; OPENROUTER_SET=false
[[ -n "${OPENAI_API_KEY:-}"     ]] && OPENAI_SET=true
[[ -n "${ANTHROPIC_API_KEY:-}"   ]] && ANTHROPIC_SET=true
[[ -n "${OPENROUTER_API_KEY:-}"  ]] && OPENROUTER_SET=true

_read_key() { local var="$1" hint="$2"; local cur d=""; set +u; cur="${!var}"; set -u; [[ -n "$cur" ]] && d=" ${DIM}[Enter to keep ${cur:0:8}...]${NC}"; printf "    ${CYAN}?${NC} ${var} ${DIM}($hint)${NC}${d}: "; read -r k; k="${k:-$cur}"; [[ -n "$k" ]] && { sed -i "/^${var}=/d" .env 2>/dev/null; echo "${var}=$k" >> .env; export "$var"="$k"; return 0; }; return 1; }

_pick_model() { local prov="$1" model_var="$2"; shift 2; local models=("$@")
  echo ""; echo "  ${BOLD}Select $prov model:${NC}"
  for i in "${!models[@]}"; do printf "    ${GREEN}%d)${NC} %s\n" "$((i+1))" "${models[$i]}"; done
  printf "    ${DIM}%d)${NC} Custom ID\n" "$((${#models[@]}+1))"; echo ""
  printf "    ${CYAN}?${NC} Pick ${DIM}[1]${NC}: "; read -r c; c="${c:-1}"
  if [[ "$c" -le "${#models[@]}" ]] 2>/dev/null && [[ "$c" -ge 1 ]]; then
    local m="${models[$((c-1))]}"; m="${m%% *}"
    sed -i "/^${model_var}=/d" .env 2>/dev/null; echo "${model_var}=$m" >> .env; export "$model_var"="$m"
  elif [[ "$c" -eq "$((${#models[@]}+1))" ]] 2>/dev/null; then
    printf "    ${CYAN}?${NC} Model ID: "; read -r m
    sed -i "/^${model_var}=/d" .env 2>/dev/null; echo "${model_var}=$m" >> .env; export "$model_var"="$m"
  fi
  _ok "$prov model: $(set +u; printf '%s' "${!model_var}"; set -u)"
}

_do_openai()    { _read_key OPENAI_API_KEY "sk-..." && _pick_model OpenAI    LOCALDISTILL_API_MODEL "openai/gpt-4o  (flagship)" "openai/gpt-4o-mini  (fast, cheap)" "openai/o3-mini  (reasoning)"; }
_do_anthropic() { _read_key ANTHROPIC_API_KEY "sk-ant-..." && _pick_model Anthropic LOCALDISTILL_API_MODEL "anthropic/claude-sonnet-4  (best balance)" "anthropic/claude-haiku-4  (fast)" "anthropic/claude-opus-4  (strongest)"; }
_do_openrouter(){ _read_key OPENROUTER_API_KEY "sk-or-..." && _pick_model OpenRouter LOCALDISTILL_API_MODEL "deepseek/deepseek-v4-pro  (best all-around)" "anthropic/claude-sonnet-4" "openai/gpt-4o" "google/gemini-2.5-flash" "meta-llama/llama-4-maverick"; }
_pick_model_openai()    { _pick_model OpenAI    LOCALDISTILL_API_MODEL "openai/gpt-4o  (flagship)" "openai/gpt-4o-mini  (fast, cheap)" "openai/o3-mini  (reasoning)"; }
_pick_model_anthropic() { _pick_model Anthropic LOCALDISTILL_API_MODEL "anthropic/claude-sonnet-4  (best balance)" "anthropic/claude-haiku-4  (fast)" "anthropic/claude-opus-4  (strongest)"; }
_pick_model_openrouter(){ _pick_model OpenRouter LOCALDISTILL_API_MODEL "deepseek/deepseek-v4-pro  (best all-around)" "anthropic/claude-sonnet-4" "openai/gpt-4o" "google/gemini-2.5-flash" "meta-llama/llama-4-maverick"; }

COUNT=$($OPENAI_SET && echo 1 || echo 0)
COUNT=$((COUNT + $($ANTHROPIC_SET && echo 1 || echo 0)))
COUNT=$((COUNT + $($OPENROUTER_SET && echo 1 || echo 0)))

# Auto-select if exactly one provider configured
if [[ $COUNT -eq 1 ]]; then
  $OPENAI_SET     && { _info "Using OpenAI (key detected)";       _pick_model_openai; }
  $ANTHROPIC_SET  && { _info "Using Anthropic (key detected)";    _pick_model_anthropic; }
  $OPENROUTER_SET && { _info "Using OpenRouter (key detected)";   _pick_model_openrouter; }
else
  echo "    ${GREEN}1)${NC} OpenAI ${DIM}(gpt-4o, gpt-4o-mini)${NC}"
  echo "    ${GREEN}2)${NC} Anthropic ${DIM}(claude-sonnet-4, haiku)${NC}"
  echo "    ${GREEN}3)${NC} OpenRouter ${DIM}(any model, one key)${NC}"
  echo "    ${GREEN}4)${NC} All three"
  echo "    ${DIM}5)${NC} Skip"
  echo ""
  printf "    ${CYAN}?${NC} Pick ${DIM}[1]${NC}: "; read -r prov; prov="${prov:-1}"
  case "$prov" in
    1) _do_openai ;;
    2) _do_anthropic ;;
    3) _do_openrouter ;;
    4) _do_openai; _do_anthropic; _do_openrouter ;;
    *) _info "Skipping" ;;
  esac
fi
source .env 2>/dev/null || true
# Step 3: Select base model for training
# ═══════════════════════════════════════════════════════════════

_step "Selecting base model for fine-tuning"

load_model() {
  local name="$1" hf="$2" size="$3" tier="$4"
  MODELS+=("$name")
  MODEL_HF+=("$hf")
  MODEL_SIZE+=("$size")
  MODEL_TIER+=("$tier")
}

MODELS=(); MODEL_HF=(); MODEL_SIZE=(); MODEL_TIER=()

load_model "Llama 3.2 3B Instruct"     "unsloth/Llama-3.2-3B-Instruct"     "3B"  "fast"
load_model "Qwen 2.5 3B Instruct"       "unsloth/Qwen2.5-3B-Instruct"       "3B"  "fast"
load_model "Phi-3.5 Mini Instruct"      "unsloth/Phi-3.5-mini-instruct"      "4B"  "fast"
load_model "Mistral 7B Instruct v0.3"   "unsloth/Mistral-7B-Instruct-v0.3"   "7B"  "balanced"
load_model "Llama 3.1 8B Instruct"      "unsloth/Llama-3.1-8B-Instruct"      "8B"  "balanced"
load_model "Qwen 2.5 7B Instruct"       "unsloth/Qwen2.5-7B-Instruct"        "7B"  "balanced"
load_model "Llama 3.3 70B Instruct"     "unsloth/Llama-3.3-70B-Instruct"     "70B" "cloud"

# Load saved preference
source .env 2>/dev/null || true
CURRENT_MODEL="${LOCALDISTILL_MODEL:-}"

echo ""
echo "  ${BOLD}Recommended base models:${NC}"
echo ""

for i in "${!MODELS[@]}"; do
  tag=""
  case "${MODEL_TIER[$i]}" in
    fast)     tag="${GREEN}⚡ fast${NC}" ;;
    balanced) tag="${CYAN}⚖ balanced${NC}" ;;
    cloud)    tag="${YELLOW}☁ cloud${NC}" ;;
  esac
  marker=" "
  [[ "${MODEL_HF[$i]}" == "$CURRENT_MODEL" ]] && marker="${GREEN}▶${NC}"
  printf "    ${GREEN}%d)${NC} %-35s ${DIM}%-5s${NC} %s\n" \
    "$((i+1))" "${MODELS[$i]}" "${MODEL_SIZE[$i]}" "$tag"
  [[ "${MODEL_HF[$i]}" == "$CURRENT_MODEL" ]] && \
    printf "       ${DIM}↑ currently selected${NC}\n"
done

printf "    ${DIM}%d)${NC} Custom (enter HuggingFace model ID)\n" "$((${#MODELS[@]}+1))"
echo ""

# Determine default choice
default=1
for i in "${!MODEL_HF[@]}"; do
  [[ "${MODEL_HF[$i]}" == "$CURRENT_MODEL" ]] && default=$((i+1))
done

printf "    ${CYAN}?${NC} Select model ${DIM}[$default]${NC}: "
read -r model_choice
model_choice="${model_choice:-$default}"

if [[ "$model_choice" -le "${#MODELS[@]}" ]] 2>/dev/null && [[ "$model_choice" -ge 1 ]]; then
  SELECTED="${MODEL_HF[$((model_choice-1))]}"
  SELECTED_NAME="${MODELS[$((model_choice-1))]}"
  SELECTED_SIZE="${MODEL_SIZE[$((model_choice-1))]}"
  SELECTED_TIER="${MODEL_TIER[$((model_choice-1))]}"
elif [[ "$model_choice" -eq "$((${#MODELS[@]}+1))" ]] 2>/dev/null; then
  printf "    ${CYAN}?${NC} HuggingFace model ID: "
  read -r custom_model
  SELECTED="$custom_model"
  SELECTED_NAME="custom: $custom_model"
  SELECTED_SIZE="?"
  SELECTED_TIER="custom"
else
  SELECTED="${MODEL_HF[0]}"
  SELECTED_NAME="${MODELS[0]}"
  SELECTED_SIZE="${MODEL_SIZE[0]}"
  SELECTED_TIER="${MODEL_TIER[0]}"
  _warn "Invalid choice — defaulting to $SELECTED_NAME"
fi

# Save to .env
sed -i "/^LOCALDISTILL_MODEL=/d" .env 2>/dev/null || true
echo "LOCALDISTILL_MODEL=$SELECTED" >> .env

# Handle cloud-tier warning
if [[ "$SELECTED_TIER" == "cloud" ]]; then
  _ok "$SELECTED_NAME (${SELECTED_SIZE}) — cloud training only"
  _warn "70B model requires cloud GPU (A100 80GB). Use: ./scripts/cloud_train.sh"
elif [[ "$SELECTED_TIER" == "fast" ]] && ! $HAS_GPU; then
  _ok "$SELECTED_NAME (${SELECTED_SIZE})"
  _warn "No local GPU detected. Model can still train via cloud: ./scripts/cloud_train.sh"
else
  _ok "$SELECTED_NAME (${SELECTED_SIZE})${DIM} — will train on ${NC}$($HAS_GPU && echo "local GPU" || echo "cloud")"
fi

# Offer to download model immediately via Ollama
if command -v ollama &>/dev/null; then
  echo ""
  printf "    ${CYAN}?${NC} Download this model now with Ollama for local inference? ${DIM}[y/N]${NC}: "
  read -r dl
  if [[ "$dl" =~ ^[Yy]$ ]]; then
    ollama_name=$(echo "$SELECTED" | sed 's|unsloth/||; s|/|:|')
    echo ""
    ollama pull "$ollama_name" 2>&1 &
    _spinner $! "Downloading $ollama_name..."
    _ok "Model pulled — ready for inference via ollama run $ollama_name"
  fi
else
  _info "Ollama not installed — model will be downloaded during training"
fi

# ═══════════════════════════════════════════════════════════════
# Step 4: Build images
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