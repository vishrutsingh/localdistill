# localdistill

Capture your API model conversations → curate → train your own local model.

**Phase 1 (built):** Proxy captures every API call to SQLite, auto-detects quality signals.
**Phase 2 (built):** MCP server for inline curation during conversations.
**Phase 3 (next):** LoRA fine-tuning pipeline from curated conversations.

## Quick Start

```bash
cd ~/localdistill

# 1. Add your API keys
cp .env.example .env
# Edit .env with your keys

# 2. Start the proxy
./run_proxy.sh

# 3. Point your tools to http://localhost:8787/v1
# Every API call is now logged, scored, and stored.
```

## MCP Curation Server

Add to your MCP client (Hermes, Claude Desktop, etc.):

```bash
hermes mcp add localdistill --command "python ~/localdistill/mcp_server.py"
```

Available slash commands/tools:
- `/signal_quality` — mark conversation as good/bad for training
- `/get_last_conversation` — check recent quality score
- `/search_knowledge` — search the RAG knowledge base
- `/get_training_status` — training pipeline overview
- `/add_to_rag` — manually index a conversation

## Architecture

```
Your Tools → localdistill Proxy (:8787) → Real API (OpenAI/Claude)
                      ↓
                  SQLite DB
                      ↓
              ┌───────┴───────┐
              ↓               ↓
         MCP Server      Training Pipeline
         (curation)      (Phase 3: LoRA)
```

## Files

- `db.py` — SQLite schema + queries
- `quality.py` — Signal detection + scoring
- `proxy_server.py` — FastAPI proxy server
- `mcp_server.py` — MCP curation server
- `proxy_config.yaml` — LiteLLM config (alternative)
- `run_proxy.sh` — Startup script