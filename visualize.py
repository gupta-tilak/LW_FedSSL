"""
Visualization and Analysis Tools for LW-FedSSL
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import seaborn as sns

class ResultsVisualizer:
    """Visualize and compare training results"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.figures_dir = self.log_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_metrics(self, session_id: str) -> Dict:
        """Load metrics from a session"""
        session_dir = self.log_dir / session_id
        metrics_file = session_dir / "metrics.jsonl"
        
        metrics = []
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
        
        return self._organize_metrics(metrics)
    
    def _organize_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Organize metrics by type"""
        organized = {
            'round_end': [],
            'communication': [],
            'computation': [],
            'aggregation': []
        }
        
        for metric in metrics_list:
            event_type = metric.get('event')
            if event_type in organized:
                organized[event_type].append(metric)
        
        return organized
    
    def plot_loss_curves(self, lwfedssl_metrics: Dict, baseline_metrics: Dict = None):
        """Plot loss curves comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # LW-FedSSL loss
        lw_rounds = [m.get('round', 0) for m in lwfedssl_metrics.get('round_end', [])]
        lw_losses = [m.get('metrics', {}).get('loss', 0) for m in lwfedssl_metrics.get('round_end', [])]
        
        ax.plot(lw_rounds, lw_losses, 'o-', linewidth=2, markersize=6, 
               label='LW-FedSSL', color='#2E86AB')
        
        # Baseline loss
        if baseline_metrics:
            bl_rounds = [m.get('round', 0) for m in baseline_metrics.get('round_end', [])]
            bl_losses = [m.get('metrics', {}).get('loss', 0) for m in baseline_metrics.get('round_end', [])]
            ax.plot(bl_rounds, bl_losses, 's-', linewidth=2, markersize=6,
                   label='Baseline FedSSL', color='#A23B72')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Loss curve saved to {self.figures_dir / 'loss_comparison.png'}")
    
    def plot_communication_costs(self, lwfedssl_metrics: Dict, baseline_metrics: Dict = None):
        """Plot communication costs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Per-round communication
        lw_rounds = [m.get('round', 0) for m in lwfedssl_metrics.get('round_end', [])]
        lw_comm = [m.get('metrics', {}).get('communication_mb', 0) 
                  for m in lwfedssl_metrics.get('round_end', [])]
        
        ax1.bar(lw_rounds, lw_comm, alpha=0.7, color='#2E86AB', label='LW-FedSSL')
        
        if baseline_metrics:
            bl_rounds = [m.get('round', 0) for m in baseline_metrics.get('round_end', [])]
            bl_comm = [m.get('metrics', {}).get('communication_mb', 0) 
                      for m in baseline_metrics.get('round_end', [])]
            ax1.bar(bl_rounds, bl_comm, alpha=0.7, color='#A23B72', label='Baseline')
        
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Communication (MB)', fontsize=12)
        ax1.set_title('Per-Round Communication', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative communication
        lw_cumsum = np.cumsum(lw_comm)
        ax2.plot(lw_rounds, lw_cumsum, 'o-', linewidth=2, markersize=6,
                color='#2E86AB', label='LW-FedSSL')
        
        if baseline_metrics and bl_comm:
            bl_cumsum = np.cumsum(bl_comm)
            ax2.plot(bl_rounds, bl_cumsum, 's-', linewidth=2, markersize=6,
                    color='#A23B72', label='Baseline')
        
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Cumulative Communication (MB)', fontsize=12)
        ax2.set_title('Cumulative Communication Cost', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'communication_costs.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Communication plot saved to {self.figures_dir / 'communication_costs.png'}")
    
    def plot_client_participation(self, session_id: str):
        """Plot client participation heatmap"""
        session_dir = self.log_dir / session_id
        client_log = session_dir / "client_activity.jsonl"
        
        if not client_log.exists():
            print("⚠️  No client activity log found")
            return
        
        # Load client activity
        client_rounds = {}
        with open(client_log, 'r') as f:
            for line in f:
                event = json.loads(line)
                if event.get('event') == 'client_update':
                    cid = event.get('client_id')
                    round_num = event.get('round')
                    if cid not in client_rounds:
                        client_rounds[cid] = []
                    client_rounds[cid].append(round_num)
        
        # Create participation matrix
        max_round = max([max(rounds) for rounds in client_rounds.values()])
        clients = sorted(client_rounds.keys())
        
        participation = np.zeros((len(clients), max_round))
        for i, cid in enumerate(clients):
            for round_num in client_rounds[cid]:
                participation[i, round_num - 1] = 1
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        im = ax.imshow(participation, aspect='auto', cmap='YlGnBu', interpolation='nearest')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Client ID', fontsize=12)
        ax.set_title('Client Participation Heatmap', fontsize=14, fontweight='bold')
        
        # Set ticks
        ax.set_yticks(range(0, len(clients), max(1, len(clients)//20)))
        ax.set_yticklabels([clients[i] for i in range(0, len(clients), max(1, len(clients)//20))])
        
        plt.colorbar(im, ax=ax, label='Participated')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'client_participation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Participation heatmap saved to {self.figures_dir / 'client_participation.png'}")
    
    def plot_convergence_analysis(self, lwfedssl_metrics: Dict):
        """Plot convergence rate analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss convergence
        lw_rounds = [m.get('round', 0) for m in lwfedssl_metrics.get('round_end', [])]
        lw_losses = [m.get('metrics', {}).get('loss', 0) for m in lwfedssl_metrics.get('round_end', [])]
        
        # Compute convergence rate (loss reduction per round)
        conv_rates = []
        window = 5
        for i in range(window, len(lw_losses)):
            rate = (lw_losses[i-window] - lw_losses[i]) / window
            conv_rates.append(rate)
        
        ax1.plot(lw_rounds[window:], conv_rates, 'o-', linewidth=2, markersize=6, color='#F18F01')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Convergence Rate', fontsize=12)
        ax1.set_title('Loss Convergence Rate', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Loss variance across rounds
        variances = []
        for i in range(window, len(lw_losses)):
            var = np.var(lw_losses[i-window:i])
            variances.append(var)
        
        ax2.plot(lw_rounds[window:], variances, 'o-', linewidth=2, markersize=6, color='#C73E1D')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Loss Variance', fontsize=12)
        ax2.set_title('Loss Stability (Lower is Better)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Convergence analysis saved to {self.figures_dir / 'convergence_analysis.png'}")
    
    def generate_comparison_report(self, lwfedssl_session: str, baseline_session: str = None):
        """Generate comprehensive comparison report"""
        lw_metrics = self.load_metrics(lwfedssl_session)
        
        print(f"\n{'='*80}")
        print("📊 Generating Visualization Report")
        print(f"{'='*80}\n")
        
        # Load baseline if available
        bl_metrics = None
        if baseline_session:
            bl_metrics = self.load_metrics(baseline_session)
        
        # Generate all plots
        self.plot_loss_curves(lw_metrics, bl_metrics)
        self.plot_communication_costs(lw_metrics, bl_metrics)
        self.plot_client_participation(lwfedssl_session)
        self.plot_convergence_analysis(lw_metrics)
        
        # Compute final statistics
        lw_losses = [m.get('metrics', {}).get('loss', 0) for m in lw_metrics.get('round_end', [])]
        lw_comm = [m.get('metrics', {}).get('communication_mb', 0) for m in lw_metrics.get('round_end', [])]
        
        report = f"""
{'='*80}
LW-FedSSL Performance Report
{'='*80}

Session: {lwfedssl_session}

Training Performance:
  - Final Loss: {lw_losses[-1] if lw_losses else 'N/A'}
  - Loss Reduction: {lw_losses[0] - lw_losses[-1] if len(lw_losses) > 1 else 'N/A'}
  - Average Loss: {np.mean(lw_losses) if lw_losses else 'N/A'}

Communication:
  - Total Communication: {sum(lw_comm):.2f} MB
  - Average per Round: {np.mean(lw_comm):.2f} MB
  - Peak per Round: {max(lw_comm):.2f} MB

"""
        
        if bl_metrics:
            bl_losses = [m.get('metrics', {}).get('loss', 0) for m in bl_metrics.get('round_end', [])]
            bl_comm = [m.get('metrics', {}).get('communication_mb', 0) for m in bl_metrics.get('round_end', [])]
            
            comm_reduction = ((sum(bl_comm) - sum(lw_comm)) / sum(bl_comm) * 100) if sum(bl_comm) > 0 else 0
            
            report += f"""
Comparison with Baseline FedSSL:
  - Communication Reduction: {comm_reduction:.1f}%
  - LW-FedSSL Communication: {sum(lw_comm):.2f} MB
  - Baseline Communication: {sum(bl_comm):.2f} MB
  
"""
        
        report += f"""
{'='*80}
All visualizations saved to: {self.figures_dir}
{'='*80}
"""
        
        print(report)
        
        # Save report
        report_file = self.figures_dir / "performance_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to {report_file}\n")

def visualize_results(lwfedssl_session: str, baseline_session: str = None):
    """Main visualization function"""
    visualizer = ResultsVisualizer()
    visualizer.generate_comparison_report(lwfedssl_session, baseline_session)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize LW-FedSSL Results")
    parser.add_argument("--lwfedssl-session", type=str, required=True,
                       help="LW-FedSSL session ID")
    parser.add_argument("--baseline-session", type=str, default=None,
                       help="Baseline session ID (optional)")
    
    args = parser.parse_args()
    
    visualize_results(args.lwfedssl_session, args.baseline_session)
