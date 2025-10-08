"""
Telemetry and Monitoring System for LW-FedSSL
Provides real-time monitoring, logging, and visualization
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
from collections import defaultdict
import numpy as np

class TelemetrySystem:
    """Central telemetry and monitoring system"""
    
    def __init__(self, log_dir: str = "./logs", enable_dashboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.enable_dashboard = enable_dashboard
        self.start_time = time.time()
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.client_metrics = defaultdict(lambda: defaultdict(list))
        self.round_metrics = defaultdict(list)
        
        # Current state
        self.current_stage = 0
        self.current_round = 0
        self.active_clients = set()
        self.client_status = {}
        
        # Locks for thread safety
        self.lock = threading.Lock()
        
        # Initialize log files
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging infrastructure"""
        self.session_dir = self.log_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log files
        self.main_log = self.session_dir / "main.log"
        self.metrics_log = self.session_dir / "metrics.jsonl"
        self.client_log = self.session_dir / "client_activity.jsonl"
        self.system_log = self.session_dir / "system.jsonl"
        
        self._log_event("system", {
            "event": "session_start",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_round_start(self, stage: int, round_num: int, selected_clients: List[int]):
        """Log the start of a training round"""
        with self.lock:
            self.current_stage = stage
            self.current_round = round_num
            self.active_clients = set(selected_clients)
            
            event = {
                "event": "round_start",
                "stage": stage,
                "round": round_num,
                "selected_clients": selected_clients,
                "num_clients": len(selected_clients),
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_event("system", event)
            print(f"\n{'='*70}")
            print(f"🚀 Stage {stage} - Round {round_num} Started")
            print(f"📊 Selected Clients: {len(selected_clients)} ({selected_clients[:5]}{'...' if len(selected_clients) > 5 else ''})")
            print(f"{'='*70}\n")
    
    def log_round_end(self, stage: int, round_num: int, metrics: Dict[str, float]):
        """Log the end of a training round"""
        with self.lock:
            event = {
                "event": "round_end",
                "stage": stage,
                "round": round_num,
                "metrics": metrics,
                "duration": time.time() - self.start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store metrics
            for key, value in metrics.items():
                self.round_metrics[key].append(value)
            
            self._log_event("metrics", event)
            
            print(f"\n✅ Round {round_num} Complete")
            print(f"📈 Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])}")
            print(f"⏱️  Duration: {metrics.get('round_time', 0):.2f}s\n")
    
    def log_client_update(self, client_id: int, stage: int, round_num: int, 
                          metrics: Dict[str, float], status: str = "success"):
        """Log client training update"""
        with self.lock:
            event = {
                "event": "client_update",
                "client_id": client_id,
                "stage": stage,
                "round": round_num,
                "status": status,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store client metrics
            for key, value in metrics.items():
                self.client_metrics[client_id][key].append(value)
            
            self.client_status[client_id] = status
            self._log_event("client", event)
    
    def log_communication(self, client_id: int, bytes_sent: float, bytes_received: float):
        """Log communication costs"""
        with self.lock:
            event = {
                "event": "communication",
                "client_id": client_id,
                "bytes_sent_mb": bytes_sent / (1024**2),
                "bytes_received_mb": bytes_received / (1024**2),
                "total_mb": (bytes_sent + bytes_received) / (1024**2),
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_event("metrics", event)
            self.metrics["communication_mb"].append(event["total_mb"])
    
    def log_computation(self, client_id: int, flops: float, time_seconds: float):
        """Log computational costs"""
        with self.lock:
            event = {
                "event": "computation",
                "client_id": client_id,
                "flops": flops,
                "gflops": flops / 1e9,
                "time_seconds": time_seconds,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_event("metrics", event)
            self.metrics["computation_gflops"].append(event["gflops"])
    
    def log_aggregation(self, stage: int, round_num: int, aggregation_metrics: Dict[str, float]):
        """Log aggregation metrics"""
        with self.lock:
            event = {
                "event": "aggregation",
                "stage": stage,
                "round": round_num,
                "metrics": aggregation_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_event("metrics", event)
    
    def log_client_selection(self, stage: int, round_num: int, 
                            available_clients: List[int], 
                            selected_clients: List[int],
                            selection_strategy: str,
                            selection_scores: Optional[Dict[int, float]] = None):
        """Log client selection process"""
        with self.lock:
            event = {
                "event": "client_selection",
                "stage": stage,
                "round": round_num,
                "strategy": selection_strategy,
                "available": len(available_clients),
                "selected": len(selected_clients),
                "selected_clients": selected_clients,
                "selection_scores": selection_scores,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_event("system", event)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        with self.lock:
            return {
                "session_id": self.session_id,
                "uptime": time.time() - self.start_time,
                "current_stage": self.current_stage,
                "current_round": self.current_round,
                "active_clients": list(self.active_clients),
                "client_status": dict(self.client_status),
                "metrics": {
                    key: {
                        "current": values[-1] if values else 0,
                        "history": values[-20:],  # Last 20 values
                        "average": np.mean(values) if values else 0,
                        "std": np.std(values) if values else 0
                    }
                    for key, values in self.metrics.items()
                },
                "round_metrics": dict(self.round_metrics)
            }
    
    def _log_event(self, log_type: str, event: Dict[str, Any]):
        """Write event to appropriate log file"""
        if log_type == "system":
            log_file = self.system_log
        elif log_type == "client":
            log_file = self.client_log
        elif log_type == "metrics":
            log_file = self.metrics_log
        else:
            log_file = self.main_log
        
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the training session"""
        with self.lock:
            total_time = time.time() - self.start_time
            
            report = f"""
{'='*80}
LW-FedSSL Training Session Summary
{'='*80}
Session ID: {self.session_id}
Total Duration: {total_time:.2f} seconds ({total_time/60:.2f} minutes)

Training Progress:
  - Current Stage: {self.current_stage}
  - Current Round: {self.current_round}
  - Total Clients Participated: {len(self.client_status)}

Communication Metrics:
  - Total Communication: {sum(self.metrics.get('communication_mb', [])):.2f} MB
  - Average per Round: {np.mean(self.metrics.get('communication_mb', [0])):.2f} MB

Computation Metrics:
  - Total Computation: {sum(self.metrics.get('computation_gflops', [])):.2f} GFLOPs
  - Average per Round: {np.mean(self.metrics.get('computation_gflops', [0])):.2f} GFLOPs

Client Performance:
"""
            for client_id, metrics in list(self.client_metrics.items())[:10]:
                avg_loss = np.mean(metrics.get('loss', [0]))
                report += f"  - Client {client_id}: Avg Loss = {avg_loss:.4f}\n"
            
            report += f"\n{'='*80}\n"
            report += f"Logs saved to: {self.session_dir}\n"
            report += f"{'='*80}\n"
            
            return report
    
    def save_summary(self):
        """Save summary report to file"""
        summary = self.generate_summary_report()
        summary_file = self.session_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(summary)
        print(summary)
        return summary

class ProgressMonitor:
    """Real-time progress monitoring"""
    
    def __init__(self, telemetry: TelemetrySystem):
        self.telemetry = telemetry
        self.update_interval = 5  # seconds
        self.running = False
        self.monitor_thread = None
    
    def start(self):
        """Start monitoring thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.running:
            dashboard_data = self.telemetry.get_dashboard_data()
            self._display_dashboard(dashboard_data)
            time.sleep(self.update_interval)
    
    def _display_dashboard(self, data: Dict[str, Any]):
        """Display dashboard in console"""
        # This would be replaced with a proper dashboard UI
        # For now, just update the console periodically
        pass

# Singleton instance
_telemetry_instance = None

def get_telemetry(log_dir: str = "./logs") -> TelemetrySystem:
    """Get or create telemetry system instance"""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetrySystem(log_dir=log_dir)
    return _telemetry_instance
