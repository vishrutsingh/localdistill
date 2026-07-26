"""
LocalDistill Monitor Dashboard

A separate FastAPI service for monitoring pipeline runs.
Provides:
- Real-time run status
- Log streaming (WebSocket)
- Historical run data
- Metrics visualization
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn


# Configuration - works both in Docker and locally
SCRIPT_DIR = Path(__file__).parent.parent  # localdistill root
LOGS_DIR = Path(os.environ.get("LOCALDISTILL_LOGS_DIR", SCRIPT_DIR / "logs"))
ADAPTERS_DIR = Path(os.environ.get("LOCALDISTILL_ADAPTERS_DIR", SCRIPT_DIR / "adapters"))
REFRESH_INTERVAL = int(os.environ.get("MONITOR_REFRESH_INTERVAL", 5))


class RunTracker:
    """Track and cache run status."""
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.runs_dir = logs_dir / "runs"
        self._cache: Dict[str, Dict] = {}
        self._last_scan = None
    
    def get_runs(self, limit: int = 20) -> List[Dict]:
        """Get recent runs, sorted by date descending."""
        if not self.runs_dir.exists():
            return []
        
        runs = []
        for run_path in sorted(self.runs_dir.iterdir(), reverse=True)[:limit]:
            if not run_path.is_dir():
                continue
            
            run_id = run_path.name
            
            # Try to load status
            status = self._load_status(run_path)
            
            runs.append({
                "id": run_id,
                "path": str(run_path),
                **status,
            })
        
        return runs
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get details for a specific run."""
        # Find run directory (partial match on run_id)
        if not self.runs_dir.exists():
            return None
        
        for run_path in self.runs_dir.iterdir():
            if run_id in run_path.name:
                status = self._load_status(run_path)
                logs = self._load_logs(run_path)
                events = self._load_events(run_path)
                
                return {
                    "id": run_path.name,
                    "path": str(run_path),
                    "logs": logs,
                    "events": events,  # All events
                    **status,
                }
        
        return None
    
    def _load_status(self, run_path: Path) -> Dict:
        """Load status.json for a run."""
        status_file = run_path / "status.json"
        
        if status_file.exists():
            try:
                with open(status_file) as f:
                    return json.load(f)
            except:
                pass
        
        # Fallback: infer from files
        return {
            "stage": "unknown",
            "started_at": datetime.fromtimestamp(run_path.stat().st_mtime).isoformat(),
            "progress": 0,
            "metrics": {},
        }
    
    def _load_logs(self, run_path: Path, tail: int = 1000) -> List[str]:
        """Load last N lines from run.log."""
        log_file = run_path / "run.log"
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file) as f:
                lines = f.readlines()
                return [l.rstrip() for l in lines[-tail:]]
        except:
            return []
    
    def _load_events(self, run_path: Path) -> List[Dict]:
        """Load events from events.jsonl."""
        events_file = run_path / "events.jsonl"
        
        if not events_file.exists():
            return []
        
        events = []
        try:
            with open(events_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except:
            pass
        
        return events


class AdapterTracker:
    """Track trained adapters."""
    
    def __init__(self, adapters_dir: Path):
        self.adapters_dir = adapters_dir
    
    def get_adapters(self, limit: int = 20) -> List[Dict]:
        """Get list of adapters."""
        if not self.adapters_dir.exists():
            return []
        
        adapters = []
        for adapter_path in sorted(self.adapters_dir.iterdir(), reverse=True)[:limit]:
            if not adapter_path.is_dir():
                continue
            
            # Check for adapter files
            config_file = adapter_path / "adapter_config.json"
            if not config_file.exists():
                continue
            
            info = {
                "id": adapter_path.name,
                "path": str(adapter_path),
                "created_at": datetime.fromtimestamp(adapter_path.stat().st_mtime).isoformat(),
                "has_gguf": (adapter_path / "gguf").exists(),
            }
            
            # Load adapter config
            try:
                with open(config_file) as f:
                    info["config"] = json.load(f)
            except:
                pass
            
            adapters.append(info)
        
        return adapters


# Initialize trackers
run_tracker = RunTracker(LOGS_DIR)
adapter_tracker = AdapterTracker(ADAPTERS_DIR)

# WebSocket connections for live updates
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Startup
    print(f"[monitor] Starting dashboard on port {os.environ.get('MONITOR_PORT', 8080)}")
    print(f"[monitor] Logs dir: {LOGS_DIR}")
    print(f"[monitor] Adapters dir: {ADAPTERS_DIR}")
    yield
    # Shutdown
    for ws in active_connections:
        await ws.close()


app = FastAPI(
    title="LocalDistill Monitor",
    description="Pipeline monitoring dashboard",
    lifespan=lifespan,
)


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Get overall system status."""
    runs = run_tracker.get_runs(limit=5)
    adapters = adapter_tracker.get_adapters(limit=5)
    
    # Find active run (not complete, not failed, and no completed_at timestamp)
    active_run = None
    for run in runs:
        stage = run.get("stage", "")
        completed = run.get("completed_at")
        if stage not in ("complete", "failed", "unknown") and not completed:
            active_run = run
            break
    
    return {
        "active_run": active_run,
        "recent_runs": len(runs),
        "total_adapters": len(adapters),
        "latest_adapter": adapters[0] if adapters else None,
    }


@app.get("/api/runs")
async def get_runs(limit: int = 20):
    """Get list of pipeline runs."""
    return run_tracker.get_runs(limit=limit)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get details for a specific run."""
    run = run_tracker.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/api/runs/{run_id}/logs")
async def get_run_logs(run_id: str, tail: int = 100):
    """Get logs for a specific run."""
    run = run_tracker.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return {"logs": run.get("logs", [])}


@app.get("/api/adapters")
async def get_adapters(limit: int = 20):
    """Get list of trained adapters."""
    return adapter_tracker.get_adapters(limit=limit)


@app.websocket("/ws/logs/{run_id}")
async def websocket_logs(websocket: WebSocket, run_id: str):
    """WebSocket for streaming logs."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Find log file
        run = run_tracker.get_run(run_id)
        if not run:
            await websocket.send_json({"error": "Run not found"})
            return
        
        log_file = Path(run["path"]) / "run.log"
        events_file = Path(run["path"]) / "events.jsonl"
        
        # Send initial logs
        if log_file.exists():
            with open(log_file) as f:
                for line in f.readlines()[-50:]:
                    await websocket.send_json({"type": "log", "data": line.rstrip()})
        
        # Watch for new logs
        last_size = log_file.stat().st_size if log_file.exists() else 0
        last_events = events_file.stat().st_size if events_file.exists() else 0
        
        while True:
            await asyncio.sleep(1)
            
            # Check for new log lines
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size > last_size:
                    with open(log_file) as f:
                        f.seek(last_size)
                        new_content = f.read()
                        for line in new_content.splitlines():
                            if line.strip():
                                await websocket.send_json({"type": "log", "data": line})
                    last_size = current_size
            
            # Check for new events
            if events_file.exists():
                current_events = events_file.stat().st_size
                if current_events > last_events:
                    with open(events_file) as f:
                        f.seek(last_events)
                        new_content = f.read()
                        for line in new_content.splitlines():
                            if line.strip():
                                try:
                                    event = json.loads(line)
                                    await websocket.send_json({"type": "event", "data": event})
                                except:
                                    pass
                    last_events = current_events
            
            # Send heartbeat
            await websocket.send_json({"type": "heartbeat"})
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"[monitor] WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# ─── Dashboard HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LocalDistill Monitor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .log-line { font-family: 'Monaco', 'Menlo', 'Consolas', monospace; font-size: 11px; line-height: 1.4; }
        .log-DEBUG { color: #6b7280; }
        .log-INFO { color: #3b82f6; }
        .log-WARNING { color: #f59e0b; }
        .log-ERROR { color: #ef4444; }
        .log-SUCCESS { color: #22c55e; }
        .stage-init { color: #6b7280; background: #374151; }
        .stage-curate { color: #3b82f6; background: #1e3a5f; }
        .stage-train { color: #f59e0b; background: #78350f; }
        .stage-benchmark { color: #8b5cf6; background: #4c1d95; }
        .stage-deploy { color: #10b981; background: #064e3b; }
        .stage-complete { color: #22c55e; background: #14532d; }
        .stage-failed { color: #ef4444; background: #7f1d1d; }
        .tab-active { border-bottom: 2px solid #3b82f6; color: #3b82f6; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
    <div x-data="dashboard()" x-init="init()" class="container mx-auto px-4 py-8 max-w-7xl">
        <!-- Header -->
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-2xl font-bold text-blue-400">LocalDistill Monitor</h1>
            <div class="flex items-center gap-4">
                <span x-show="status.active_run" class="px-3 py-1 bg-green-600 rounded-full text-sm animate-pulse">
                    Training
                </span>
                <span x-show="!status.active_run" class="px-3 py-1 bg-gray-600 rounded-full text-sm">
                    Idle
                </span>
                <button @click="refresh()" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                    Refresh
                </button>
            </div>
        </div>
        
        <!-- Status Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-xs text-gray-500 uppercase">Active Run</div>
                <template x-if="status.active_run">
                    <div>
                        <div class="text-lg font-mono mt-1" x-text="status.active_run.id.slice(0,12)"></div>
                        <div class="mt-2 bg-gray-700 rounded-full h-1.5">
                            <div class="bg-blue-500 h-1.5 rounded-full transition-all" 
                                 :style="'width: ' + (status.active_run.progress || 0) + '%'"></div>
                        </div>
                    </div>
                </template>
                <template x-if="!status.active_run">
                    <div class="text-gray-500 mt-1">None</div>
                </template>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-xs text-gray-500 uppercase">Total Runs</div>
                <div class="text-2xl font-bold mt-1" x-text="runs.length || 0"></div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-xs text-gray-500 uppercase">Adapters</div>
                <div class="text-2xl font-bold mt-1" x-text="status.total_adapters || 0"></div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-xs text-gray-500 uppercase">Latest Adapter</div>
                <div class="text-lg font-mono mt-1" x-text="status.latest_adapter?.id?.slice(0,8) || '-'"></div>
            </div>
        </div>
        
        <!-- Main Content: Runs List + Detail Panel -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Runs List -->
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-sm font-semibold mb-3 text-gray-400 uppercase">Pipeline Runs</h2>
                <div class="space-y-2 max-h-[600px] overflow-y-auto">
                    <template x-for="run in runs" :key="run.id">
                        <div @click="selectRun(run.id)" 
                             class="p-3 rounded cursor-pointer transition-colors"
                             :class="selectedRunId === run.id ? 'bg-blue-900/50 border border-blue-500' : 'bg-gray-700/50 hover:bg-gray-700'">
                            <div class="flex justify-between items-start">
                                <div class="font-mono text-sm" x-text="run.id.slice(0,16)"></div>
                                <span class="px-2 py-0.5 rounded text-xs"
                                      :class="'stage-' + (run.stage || 'unknown')"
                                      x-text="run.stage || '?'"></span>
                            </div>
                            <div class="text-xs text-gray-500 mt-1" x-text="formatDate(run.started_at)"></div>
                            <template x-if="run.metrics?.loss">
                                <div class="text-xs text-yellow-400 mt-1">
                                    Loss: <span x-text="run.metrics.loss.toFixed(4)"></span>
                                </div>
                            </template>
                        </div>
                    </template>
                    <template x-if="runs.length === 0">
                        <div class="text-gray-500 text-sm p-4 text-center">No runs yet</div>
                    </template>
                </div>
            </div>
            
            <!-- Detail Panel -->
            <div class="lg:col-span-2 bg-gray-800 rounded-lg p-4">
                <template x-if="!selectedRunId">
                    <div class="text-gray-500 text-center py-20">
                        Select a run to view details
                    </div>
                </template>
                <template x-if="selectedRunId && runDetail">
                    <div>
                        <!-- Run Header -->
                        <div class="flex justify-between items-center mb-4">
                            <div>
                                <h2 class="text-lg font-mono" x-text="runDetail.id"></h2>
                                <div class="text-xs text-gray-500 mt-1">
                                    Started: <span x-text="formatDate(runDetail.started_at)"></span>
                                    <template x-if="runDetail.completed_at">
                                        <span> | Completed: <span x-text="formatDate(runDetail.completed_at)"></span></span>
                                    </template>
                                </div>
                            </div>
                            <span class="px-3 py-1 rounded text-sm"
                                  :class="'stage-' + (runDetail.stage || 'unknown')"
                                  x-text="runDetail.stage || 'unknown'"></span>
                        </div>
                        
                        <!-- Metrics -->
                        <template x-if="runDetail.metrics && Object.keys(runDetail.metrics).length > 0">
                            <div class="grid grid-cols-4 gap-3 mb-4">
                                <template x-for="[key, val] in Object.entries(runDetail.metrics)" :key="key">
                                    <div class="bg-gray-700/50 rounded p-2">
                                        <div class="text-xs text-gray-500 uppercase" x-text="key"></div>
                                        <div class="text-sm font-mono" x-text="typeof val === 'number' ? val.toFixed(4) : val"></div>
                                    </div>
                                </template>
                            </div>
                        </template>
                        
                        <!-- Loss Chart (fixed container, not in template) -->
                        <div class="mb-4 bg-gray-700/30 rounded p-3" x-show="lossData.length > 1">
                            <div class="text-xs text-gray-500 uppercase mb-2">Training Loss</div>
                            <div style="height: 120px;">
                                <canvas id="lossChart"></canvas>
                            </div>
                        </div>
                        
                        <!-- Tabs -->
                        <div class="border-b border-gray-700 mb-3">
                            <button @click="logTab = 'events'" 
                                    class="px-4 py-2 text-sm"
                                    :class="logTab === 'events' ? 'tab-active' : 'text-gray-500'">
                                Events (<span x-text="runDetail.events?.length || 0"></span>)
                            </button>
                            <button @click="logTab = 'logs'" 
                                    class="px-4 py-2 text-sm"
                                    :class="logTab === 'logs' ? 'tab-active' : 'text-gray-500'">
                                Raw Logs (<span x-text="runDetail.logs?.length || 0"></span>)
                            </button>
                        </div>
                        
                        <!-- Events Tab -->
                        <div x-show="logTab === 'events'" class="bg-gray-900 rounded p-3 max-h-[400px] overflow-y-auto">
                            <template x-for="(evt, i) in runDetail.events || []" :key="i">
                                <div class="log-line py-0.5 flex gap-2">
                                    <span class="text-gray-600 w-20 flex-shrink-0" x-text="evt.timestamp?.slice(11,19) || ''"></span>
                                    <span class="w-16 flex-shrink-0" :class="'log-' + evt.level" x-text="'[' + evt.stage + ']'"></span>
                                    <span :class="'log-' + evt.level" x-text="evt.message"></span>
                                </div>
                            </template>
                            <template x-if="!runDetail.events?.length">
                                <div class="text-gray-500 text-sm">No events</div>
                            </template>
                        </div>
                        
                        <!-- Logs Tab -->
                        <div x-show="logTab === 'logs'" class="bg-gray-900 rounded p-3 max-h-[400px] overflow-y-auto">
                            <template x-for="(line, i) in runDetail.logs || []" :key="i">
                                <div class="log-line py-0.5 text-gray-300" x-text="line"></div>
                            </template>
                            <template x-if="!runDetail.logs?.length">
                                <div class="text-gray-500 text-sm">No logs</div>
                            </template>
                        </div>
                    </div>
                </template>
            </div>
        </div>
    </div>
    
    <script>
        function dashboard() {
            return {
                status: {},
                runs: [],
                selectedRunId: null,
                runDetail: null,
                logTab: 'events',
                lossData: [],
                lossChart: null,
                
                async init() {
                    await this.refresh();
                    setInterval(() => this.refresh(), 5000);
                },
                
                async refresh() {
                    try {
                        const [statusRes, runsRes] = await Promise.all([
                            fetch('/api/status'),
                            fetch('/api/runs?limit=50')
                        ]);
                        this.status = await statusRes.json();
                        this.runs = await runsRes.json();
                        
                        // Auto-select active run or first run
                        if (!this.selectedRunId && this.runs.length > 0) {
                            this.selectRun(this.status.active_run?.id || this.runs[0].id);
                        } else if (this.selectedRunId) {
                            // Refresh current selection
                            await this.loadRunDetail(this.selectedRunId);
                        }
                    } catch (e) {
                        console.error('Refresh failed:', e);
                    }
                },
                
                async selectRun(runId) {
                    this.selectedRunId = runId;
                    await this.loadRunDetail(runId);
                },
                
                async loadRunDetail(runId) {
                    try {
                        const res = await fetch(`/api/runs/${runId}`);
                        this.runDetail = await res.json();
                        
                        // Extract loss data for chart
                        this.lossData = (this.runDetail.events || [])
                            .filter(e => e.data?.loss !== undefined)
                            .map(e => ({ step: e.data.step, loss: e.data.loss }));
                        
                        this.$nextTick(() => this.renderLossChart());
                    } catch (e) {
                        console.error('Load run failed:', e);
                    }
                },
                
                renderLossChart() {
                    const ctx = document.getElementById('lossChart');
                    if (!ctx) return;
                    
                    // Always destroy previous chart
                    if (this.lossChart) {
                        this.lossChart.destroy();
                        this.lossChart = null;
                    }
                    
                    if (this.lossData.length < 2) return;
                    
                    this.lossChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: this.lossData.map(d => d.step),
                            datasets: [{
                                label: 'Loss',
                                data: this.lossData.map(d => d.loss),
                                borderColor: '#f59e0b',
                                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                fill: true,
                                tension: 0.3,
                                pointRadius: 0,
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            animation: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: { display: true, grid: { color: '#374151' }, ticks: { color: '#6b7280', maxTicksLimit: 10 } },
                                y: { display: true, grid: { color: '#374151' }, ticks: { color: '#6b7280' } }
                            }
                        }
                    });
                },
                
                formatDate(iso) {
                    if (!iso) return '-';
                    const d = new Date(iso);
                    return d.toLocaleString();
                }
            };
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard."""
    return DASHBOARD_HTML


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """Run the monitor server."""
    port = int(os.environ.get("MONITOR_PORT", 8080))
    host = os.environ.get("MONITOR_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
