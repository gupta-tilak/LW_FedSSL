"""
Quick Start Demo for Enhanced LW-FedSSL
Demonstrates the key features with a minimal example
"""
import torch
from pathlib import Path
import json

from config import CONFIG
from telemetry import get_telemetry
from metrics import MetricsTracker, ComparisonMetrics
from client_selector import get_selector
from data_utils import get_cifar10_partitioned
from task import TinyCNN

def demo_data_partitioning():
    """Demonstrate data partitioning strategies"""
    print("\n" + "="*80)
    print("📊 Demo 1: Data Partitioning Strategies")
    print("="*80 + "\n")
    
    # IID partitioning
    print("1. IID Partitioning:")
    partitioner_iid, _ = get_cifar10_partitioned(
        num_clients=10,
        distribution='iid',
        batch_size=128
    )
    partitioner_iid.print_statistics()
    
    # Non-IID Label partitioning
    print("\n2. Non-IID Label Partitioning:")
    partitioner_non_iid, _ = get_cifar10_partitioned(
        num_clients=10,
        distribution='non_iid_label',
        batch_size=128
    )
    partitioner_non_iid.print_statistics()
    
    # Non-IID Dirichlet partitioning
    print("\n3. Non-IID Dirichlet Partitioning (α=0.5):")
    partitioner_dirichlet, _ = get_cifar10_partitioned(
        num_clients=10,
        distribution='non_iid_dirichlet',
        alpha=0.5,
        batch_size=128
    )
    partitioner_dirichlet.print_statistics()

def demo_client_selection():
    """Demonstrate client selection strategies"""
    print("\n" + "="*80)
    print("🎯 Demo 2: Client Selection Strategies")
    print("="*80 + "\n")
    
    strategies = ['random', 'performance_based', 'diversity_based', 'hybrid', 'adaptive']
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 60)
        
        selector = get_selector(
            strategy=strategy,
            num_clients=40,
            min_clients=10,
            max_clients=20
        )
        
        # Simulate client selection for 3 rounds
        available_clients = list(range(40))
        
        for round_num in range(1, 4):
            result = selector.select(available_clients, round_num)
            print(f"  Round {round_num}: Selected {len(result.selected_clients)} clients")
            print(f"    Clients: {result.selected_clients[:5]}...")
            
            # Update with dummy performance data
            for cid in result.selected_clients:
                selector.update_client_performance(
                    cid, 
                    loss=1.5 + 0.1 * (round_num - 1),
                    training_time=10.0,
                    success=True
                )

def demo_telemetry():
    """Demonstrate telemetry system"""
    print("\n" + "="*80)
    print("📡 Demo 3: Telemetry System")
    print("="*80 + "\n")
    
    telemetry = get_telemetry("./demo_logs")
    
    # Simulate training rounds
    for stage in range(1, 3):
        for round_num in range(1, 4):
            # Log round start
            selected_clients = [0, 1, 2, 5, 8]
            telemetry.log_round_start(stage, round_num, selected_clients)
            
            # Simulate client updates
            for cid in selected_clients:
                metrics = {
                    "loss": 2.0 - (stage * 0.3 + round_num * 0.1),
                    "training_time": 15.0 + cid * 0.5
                }
                telemetry.log_client_update(cid, stage, round_num, metrics)
            
            # Log round end
            round_metrics = {
                "loss": 1.8 - (stage * 0.3 + round_num * 0.1),
                "round_time": 18.0,
                "communication_mb": 5.2,
                "num_clients": len(selected_clients)
            }
            telemetry.log_round_end(stage, round_num, round_metrics)
    
    # Generate summary
    summary = telemetry.generate_summary_report()
    print(summary)
    
    # Get dashboard data
    dashboard = telemetry.get_dashboard_data()
    print("\n📊 Dashboard Data Sample:")
    print(f"  Session: {dashboard['session_id']}")
    print(f"  Uptime: {dashboard['uptime']:.2f}s")
    print(f"  Current Stage: {dashboard['current_stage']}")
    print(f"  Current Round: {dashboard['current_round']}")

def demo_metrics():
    """Demonstrate metrics tracking"""
    print("\n" + "="*80)
    print("📈 Demo 4: Advanced Metrics")
    print("="*80 + "\n")
    
    tracker = MetricsTracker()
    
    # Simulate training progression
    for i in range(10):
        loss = 2.5 - i * 0.15
        accuracy = 0.5 + i * 0.03
        comm_bytes = 1024 * 1024 * 5  # 5 MB
        
        tracker.update_basic_metrics(loss, accuracy, comm_bytes)
    
    # Compute advanced metrics
    conv_rate = tracker.compute_convergence_rate()
    comm_efficiency = tracker.compute_communication_efficiency()
    
    print(f"Convergence Rate: {conv_rate:.4f}")
    print(f"Communication Efficiency: {comm_efficiency['communication_efficiency']:.6f}")
    print(f"Total Communication: {comm_efficiency['total_communication_mb']:.2f} MB")
    
    # Get full summary
    summary = tracker.get_summary()
    print("\n📊 Metrics Summary:")
    for metric_name, metric_data in summary['basic_metrics'].items():
        print(f"\n  {metric_name.upper()}:")
        print(f"    Mean: {metric_data['mean']:.4f}")
        print(f"    Std: {metric_data['std']:.4f}")
        print(f"    Final: {metric_data['final']:.4f}")

def demo_model():
    """Demonstrate model architecture"""
    print("\n" + "="*80)
    print("🧠 Demo 5: Model Architecture")
    print("="*80 + "\n")
    
    model = TinyCNN()
    
    print("Model Structure:")
    print(model)
    
    print("\n\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Layer-wise parameters
    print("\n  Layer-wise Breakdown:")
    for i, layer in enumerate(model.layers):
        params = sum(p.numel() for p in layer.parameters())
        print(f"    Layer {i+1}: {params:,} parameters")
    
    # Test forward pass at different depths
    print("\n\nForward Pass Test:")
    x = torch.randn(4, 3, 32, 32)
    
    for depth in range(1, 4):
        output = model(x, depth=depth)
        print(f"  Depth {depth}: Input {tuple(x.shape)} → Output {tuple(output.shape)}")

def demo_comparison():
    """Demonstrate comparison metrics"""
    print("\n" + "="*80)
    print("⚖️  Demo 6: LW-FedSSL vs Baseline Comparison")
    print("="*80 + "\n")
    
    comparison = ComparisonMetrics()
    
    # Simulate LW-FedSSL training
    print("Simulating LW-FedSSL training...")
    for i in range(30):
        loss = 2.0 - i * 0.05
        accuracy = 0.6 + i * 0.01
        comm_bytes = 1024 * 1024 * 3  # 3 MB per round (layer only)
        
        comparison.update_lwfedssl(loss=loss, accuracy=accuracy, comm_bytes=comm_bytes)
    
    # Simulate Baseline training
    print("Simulating Baseline FedSSL training...")
    for i in range(30):
        loss = 2.0 - i * 0.048
        accuracy = 0.6 + i * 0.009
        comm_bytes = 1024 * 1024 * 12  # 12 MB per round (full model)
        
        comparison.update_baseline(loss=loss, accuracy=accuracy, comm_bytes=comm_bytes)
    
    # Get comparison
    comp_results = comparison.get_comparison()
    
    print("\n📊 Comparison Results:")
    print("\nLW-FedSSL:")
    print(f"  Final Loss: {comp_results['lwfedssl']['basic_metrics']['loss']['final']:.4f}")
    print(f"  Total Communication: {comp_results['lwfedssl']['basic_metrics']['communication_mb']['mean'] * 30:.2f} MB")
    
    print("\nBaseline FedSSL:")
    print(f"  Final Loss: {comp_results['baseline']['basic_metrics']['loss']['final']:.4f}")
    print(f"  Total Communication: {comp_results['baseline']['basic_metrics']['communication_mb']['mean'] * 30:.2f} MB")
    
    print("\n🎯 Improvements:")
    for metric, improvement in comp_results['improvements'].items():
        print(f"  {metric}: {improvement}")

def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("🚀 Enhanced LW-FedSSL Feature Demonstration")
    print("="*80)
    
    demos = [
        ("Data Partitioning", demo_data_partitioning),
        ("Client Selection", demo_client_selection),
        ("Telemetry System", demo_telemetry),
        ("Advanced Metrics", demo_metrics),
        ("Model Architecture", demo_model),
        ("Performance Comparison", demo_comparison)
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ Demo Complete!")
    print("="*80)
    print("\n📚 Next Steps:")
    print("  1. Run local simulation: python simulate_clients.py --num-clients 10")
    print("  2. Start distributed training: python enhanced_server.py --mode lwfedssl")
    print("  3. Visualize results: python visualize.py --lwfedssl-session <session_id>")
    print("\n")

if __name__ == "__main__":
    main()
