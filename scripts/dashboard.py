"""
Lightweight Python server for the IndiQuant Quantitative Research Dashboard.
Provides APIs for configuration management, execution of backtests, 
and reading generated reports and plots.
"""

from __future__ import annotations

import http.server
import socketserver
import json
import os
import sys
import subprocess
import threading
import urllib.parse
from pathlib import Path
import traceback

PORT = 8000
WORKSPACE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = WORKSPACE_DIR / "config"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
DASHBOARD_DIR = WORKSPACE_DIR / "src" / "dashboard"

# Global state for background runner
runner_state = {
    "status": "IDLE",  # IDLE, RUNNING, SUCCESS, FAILED
    "process": None,
    "log_file": OUTPUTS_DIR / "live_run.log"
}
runner_lock = threading.Lock()


def run_backtest_thread(period: str, force_refresh: bool):
    global runner_state
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = runner_state["log_file"]
    
    # Empty existing log
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"--- Launching Backtest Simulation (Period: {period}, Force Refresh: {force_refresh}) ---\n")
        lf.flush()
    
    cmd = [sys.executable, str(WORKSPACE_DIR / "scripts" / "run_backtest.py"), "--period", period]
    if force_refresh:
        cmd.append("--force-refresh")
        
    try:
        # Start subprocess redirecting stdout & stderr to the log file
        with open(log_path, "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(WORKSPACE_DIR),
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            with runner_lock:
                runner_state["process"] = proc
                runner_state["status"] = "RUNNING"
                
            proc.wait()
            
            with runner_lock:
                runner_state["process"] = None
                if proc.returncode == 0:
                    runner_state["status"] = "SUCCESS"
                    lf.write("\n--- Backtest Run Completed Successfully ---\n")
                else:
                    runner_state["status"] = "FAILED"
                    lf.write(f"\n--- Backtest Run Failed with Exit Code {proc.returncode} ---\n")
    except Exception as exc:
        with runner_lock:
            runner_state["status"] = "FAILED"
            runner_state["process"] = None
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\nERROR: Failed to launch subprocess: {exc}\n")
            lf.write(traceback.format_exc())


class DashboardHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Suppress standard logging to console for cleaner CLI output
        pass

    def end_headers(self):
        # Prevent browser caching of API responses and HTML
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # 1. API Endpoints
        if path == "/api/configs":
            self.handle_get_configs()
        elif path == "/api/metrics":
            self.handle_get_metrics()
        elif path == "/api/logs":
            self.handle_get_logs()
            
        # 2. Static outputs (for charts/CSV downloads)
        elif path.startswith("/outputs/"):
            self.handle_serve_static(OUTPUTS_DIR, path[9:])
            
        # 3. Frontend files
        elif path == "/" or path == "/index.html":
            self.handle_serve_static(DASHBOARD_DIR, "index.html")
        else:
            # Check if requested file exists in dashboard dir
            rel_file = path.lstrip("/")
            if (DASHBOARD_DIR / rel_file).exists() and (DASHBOARD_DIR / rel_file).is_file():
                self.handle_serve_static(DASHBOARD_DIR, rel_file)
            else:
                self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == "/api/configs":
            self.handle_post_configs()
        elif path == "/api/run-backtest":
            self.handle_post_run_backtest()
        else:
            self.send_error(404, "Endpoint Not Found")

    # ─── GET Handlers ─────────────────────────────────────────────────────────

    def handle_get_configs(self):
        configs = {}
        for name in ["universe", "strategy", "backtest"]:
            file_path = CONFIG_DIR / f"{name}.yaml"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        configs[name] = f.read()
                except Exception as exc:
                    configs[name] = f"Error reading config: {exc}"
            else:
                configs[name] = f"Config file {name}.yaml not found."
                
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(configs).encode("utf-8"))

    def handle_get_metrics(self):
        metrics_file = OUTPUTS_DIR / "metrics.json"
        metrics = {}
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception as exc:
                metrics = {"error": f"Failed to parse metrics: {exc}"}
        else:
            metrics = {"error": "Metrics file not found. Please run the backtest first."}
            
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(metrics).encode("utf-8"))

    def handle_get_logs(self):
        log_file = runner_state["log_file"]
        log_content = ""
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    log_content = f.read()
            except Exception as exc:
                log_content = f"Error reading logs: {exc}"
        else:
            log_content = "No backtest has been run yet in this session."
            
        response = {
            "status": runner_state["status"],
            "logs": log_content
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    # ─── POST Handlers ────────────────────────────────────────────────────────

    def handle_post_configs(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        
        try:
            data = json.loads(post_data)
            for name in ["universe", "strategy", "backtest"]:
                if name in data:
                    yaml_content = data[name]
                    # Simple validation: parse it as YAML
                    import yaml
                    try:
                        yaml.safe_load(yaml_content)
                    except Exception as parse_err:
                        raise ValueError(f"YAML Syntax Error in {name}.yaml: {parse_err}")
                        
                    file_path = CONFIG_DIR / f"{name}.yaml"
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(yaml_content)
                        
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "SUCCESS", "message": "Configs saved successfully."}).encode("utf-8"))
            
        except Exception as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "FAILED", "message": str(exc)}).encode("utf-8"))

    def handle_post_run_backtest(self):
        global runner_state
        
        # Check if already running
        with runner_lock:
            if runner_state["status"] == "RUNNING":
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "FAILED", "message": "Backtest is already running."}).encode("utf-8"))
                return
                
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        
        period = "full"
        force_refresh = False
        
        try:
            if post_data:
                data = json.loads(post_data)
                period = data.get("period", "full")
                force_refresh = data.get("force_refresh", False)
        except Exception:
            pass
            
        # Start background thread
        thread = threading.Thread(target=run_backtest_thread, args=(period, force_refresh))
        thread.daemon = True
        thread.start()
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "RUNNING", "message": "Backtest launched in background."}).encode("utf-8"))

    # ─── Static Serve Helper ──────────────────────────────────────────────────

    def handle_serve_static(self, base_dir: Path, rel_path: str):
        rel_path = urllib.parse.unquote(rel_path)
        file_path = (base_dir / rel_path).resolve()
        
        # Security check: ensure file is inside base_dir
        if not str(file_path).startswith(str(base_dir.resolve())):
            self.send_error(403, "Access Denied")
            return
            
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File Not Found")
            return
            
        # Determine mime-type
        content_type = "application/octet-stream"
        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".json":
            content_type = "application/json; charset=utf-8"
        elif file_path.suffix == ".csv":
            content_type = "text/csv; charset=utf-8"
        elif file_path.suffix in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            content_type = f"image/{file_path.suffix[1:]}"
            
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        
        try:
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        except Exception as exc:
            print(f"Error serving {file_path}: {exc}")


def main():
    # Make sure outputs dir exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    
    handler = DashboardHTTPRequestHandler
    
    # Try different ports if 8000 is taken
    port = PORT
    for _ in range(10):
        try:
            with socketserver.TCPServer(("", port), handler) as httpd:
                print("==========================================================")
                print(f"IndiQuant Backtest Dashboard is running at:")
                print(f"   http://localhost:{port}/")
                print("==========================================================")
                print("Press Ctrl+C to stop the dashboard server.")
                httpd.serve_forever()
        except OSError:
            print(f"Port {port} is busy. Trying next port...")
            port += 1
            
    print("Could not find a free port to bind the server.")
    sys.exit(1)


if __name__ == "__main__":
    main()
