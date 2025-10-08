"""
Enhanced LW-FedSSL Server with Advanced Features
- Dynamic client selection (40 clients)
- Comprehensive telemetry
- Advanced metrics tracking
- Baseline comparison
"""
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, Scalar, FitIns
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path

from config import CONFIG
from telemetry import get_telemetry
from metrics import MetricsTracker, ComparisonMetrics
from client_selector import get_selector
from task import TinyCNN

class EnhancedLWFedSSLStrategy(FedAvg):
    """Enhanced strategy with client selection and comprehensive tracking"""
    
    def __init__(self, model: nn.Module, current_stage: int, 
                 client_selector, metrics_tracker: MetricsTracker, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.current_stage = current_stage
        self.client_selector = client_selector
        self.metrics_tracker = metrics_tracker
        self.telemetry = get_telemetry(CONFIG.LOG_DIR)
        
        # Track round-specific data
        self.round_start_time = None
        self.total_communication = 0
        self.round_number = 0
        
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configure clients for training with dynamic selection"""
        self.round_number = server_round
        self.round_start_time = time.time()
        
        # Get available clients
        available_clients_dict = client_manager.all()
        available_clients = list(available_clients_dict.keys())
        
        # Check if clients have numeric IDs or UUID-style IDs
        try:
            # Try to parse first client ID as int
            int(available_clients[0])
            use_numeric_ids = True
        except (ValueError, IndexError):
            use_numeric_ids = False
        
        if use_numeric_ids:
            # Use our selection strategy for numeric client IDs
            available_client_ids = [int(cid) for cid in available_clients]
            selection_result = self.client_selector.select(available_client_ids, server_round)
            selected_client_ids = [str(cid) for cid in selection_result.selected_clients]
        else:
            # For UUID-style IDs, select subset based on fraction_fit
            import random
            num_to_select = max(int(len(available_clients) * self.fraction_fit), self.min_fit_clients)
            num_to_select = min(num_to_select, len(available_clients))
            selected_client_ids = random.sample(available_clients, num_to_select)
            available_client_ids = list(range(len(available_clients)))  # Just for logging
            
            # Create a simple selection result
            from types import SimpleNamespace
            selection_result = SimpleNamespace(
                selected_clients=list(range(len(selected_client_ids))),
                scores=[1.0] * len(selected_client_ids),
                strategy_used="random_uuid"
            )
        
        # Log selection
        self.telemetry.log_client_selection(
            stage=self.current_stage,
            round_num=server_round,
            available_clients=available_client_ids,
            selected_clients=selection_result.selected_clients,
            selection_strategy=selection_result.strategy_used,
            selection_scores=selection_result.scores
        )
        
        # Log round start
        self.telemetry.log_round_start(
            stage=self.current_stage,
            round_num=server_round,
            selected_clients=selection_result.selected_clients
        )
        
        # Create fit config
        config = {
            "stage": self.current_stage,
            "round": server_round,
            "local_epochs": CONFIG.LOCAL_EPOCHS,
            "batch_size": CONFIG.BATCH_SIZE,
            "learning_rate": CONFIG.LEARNING_RATE
        }
        
        # Select client proxies - use the actual Flower client IDs
        if use_numeric_ids:
            # Map selected numeric IDs to client proxies
            selected_proxies = []
            for cid in selection_result.selected_clients:
                cid_str = str(cid)
                if cid_str in available_clients_dict:
                    selected_proxies.append(available_clients_dict[cid_str])
        else:
            # Use the selected UUID clients directly
            selected_proxies = [available_clients_dict[cid] for cid in selected_client_ids]
        
        print(f"🔍 DEBUG: use_numeric_ids={use_numeric_ids}, selected {len(selected_proxies)} proxies out of {len(selected_client_ids) if not use_numeric_ids else len(selection_result.selected_clients)} requested")
        
        # Create FitIns objects with parameters and config
        fit_ins = FitIns(parameters=parameters, config=config)
        
        # Return list of (client_proxy, fit_ins) tuples
        return [(proxy, fit_ins) for proxy in selected_proxies]
    
    def aggregate_fit(self, server_round: int, 
                     results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], 
                     failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates with comprehensive tracking"""
        
        if not results:
            return None, {}
        
        # Extract weights and metadata
        weights_results = []
        client_losses = []
        client_ids = []
        
        for client_proxy, fit_res in results:
            # Extract client ID - handle both numeric and UUID formats
            try:
                if '_' in client_proxy.cid:
                    cid = int(client_proxy.cid.split('_')[1])
                else:
                    try:
                        cid = int(client_proxy.cid)
                    except ValueError:
                        # UUID format - use hash as numeric ID for tracking
                        cid = abs(hash(client_proxy.cid)) % 10000
            except:
                cid = 0  # Fallback
            
            client_ids.append(cid)
            
            # Get parameters
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((params, fit_res.num_examples))
            
            # Track metrics
            loss = fit_res.metrics.get("loss", 0)
            client_losses.append(loss)
            training_time = fit_res.metrics.get("training_time", 0)
            
            # Update client selector
            self.client_selector.update_client_performance(
                cid, loss, training_time, success=True
            )
            
            # Log client update
            self.telemetry.log_client_update(
                client_id=cid,
                stage=self.current_stage,
                round_num=server_round,
                metrics=fit_res.metrics,
                status="success"
            )
            
            # Compute client contribution
            if hasattr(self.metrics_tracker, 'compute_client_contribution'):
                global_params = self.get_layer_parameters()
                contribution = self.metrics_tracker.compute_client_contribution(
                    cid, params, global_params, fit_res.num_examples
                )
        
        # Aggregate weights
        aggregated_weights = self.aggregate_weights(weights_results)
        
        # Update global model
        layer_idx = self.current_stage - 1
        layer_state = {
            f"layers.{layer_idx}.weight": torch.tensor(aggregated_weights[0]),
            f"layers.{layer_idx}.bias": torch.tensor(aggregated_weights[1])
        }
        self.model.load_state_dict(layer_state, strict=False)
        
        # Compute metrics
        round_time = time.time() - self.round_start_time
        avg_loss = np.mean(client_losses)
        
        # Communication costs
        params_size = sum([w.nbytes for w in aggregated_weights])
        comm_bytes = params_size * len(results) * 2  # Upload + Download
        self.total_communication += comm_bytes
        
        # Compute aggregation variance
        if hasattr(self.metrics_tracker, 'compute_aggregation_variance'):
            client_params = [params for params, _ in weights_results]
            agg_variance = self.metrics_tracker.compute_aggregation_variance(client_params)
        else:
            agg_variance = {}
        
        # Compile metrics
        metrics = {
            "loss": float(avg_loss),
            "round_time": float(round_time),
            "communication_mb": float(comm_bytes / (1024**2)),
            "num_clients": len(results),
            "total_communication_mb": float(self.total_communication / (1024**2)),
            **agg_variance
        }
        
        # Update metrics tracker
        self.metrics_tracker.update_basic_metrics(
            loss=avg_loss,
            accuracy=0.0,  # Will be computed during evaluation
            comm_bytes=comm_bytes
        )
        
        # Log round end
        self.telemetry.log_round_end(
            stage=self.current_stage,
            round_num=server_round,
            metrics=metrics
        )
        
        # Log aggregation
        self.telemetry.log_aggregation(
            stage=self.current_stage,
            round_num=server_round,
            aggregation_metrics=agg_variance
        )
        
        return fl.common.ndarrays_to_parameters(aggregated_weights), metrics
    
    def aggregate_weights(self, results):
        """Weighted averaging of model parameters"""
        total_examples = sum([num_examples for _, num_examples in results])
        
        # Initialize aggregated weights
        aggregated = [np.zeros_like(w) for w in results[0][0]]
        
        for weights, num_examples in results:
            for i, w in enumerate(weights):
                aggregated[i] += w * (num_examples / total_examples)
        
        return aggregated
    
    def get_layer_parameters(self) -> List[np.ndarray]:
        """Get current layer parameters"""
        layer_idx = self.current_stage - 1
        layer = self.model.layers[layer_idx]
        return [
            layer.weight.data.cpu().numpy(),
            layer.bias.data.cpu().numpy()
        ]

class BaselineFedSSLStrategy(FedAvg):
    """Baseline FedSSL strategy (full model aggregation)"""
    
    def __init__(self, model: nn.Module, metrics_tracker: MetricsTracker, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.metrics_tracker = metrics_tracker
        self.telemetry = get_telemetry(CONFIG.LOG_DIR)
        self.round_start_time = None
        self.total_communication = 0
    
    def aggregate_fit(self, server_round: int, 
                     results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], 
                     failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate full model parameters"""
        
        if not results:
            return None, {}
        
        self.round_start_time = time.time()
        
        # Extract all parameters (full model)
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate
        aggregated_weights = self.aggregate_weights(weights_results)
        
        # Update model
        params_dict = self.model.state_dict()
        keys = list(params_dict.keys())
        for i, key in enumerate(keys):
            if i < len(aggregated_weights):
                params_dict[key] = torch.tensor(aggregated_weights[i])
        self.model.load_state_dict(params_dict)
        
        # Metrics
        round_time = time.time() - self.round_start_time
        losses = [fit_res.metrics.get("loss", 0) for _, fit_res in results]
        avg_loss = np.mean(losses)
        
        # Communication (full model)
        params_size = sum([w.nbytes for w in aggregated_weights])
        comm_bytes = params_size * len(results) * 2
        self.total_communication += comm_bytes
        
        metrics = {
            "loss": float(avg_loss),
            "round_time": float(round_time),
            "communication_mb": float(comm_bytes / (1024**2)),
            "total_communication_mb": float(self.total_communication / (1024**2))
        }
        
        self.metrics_tracker.update_basic_metrics(avg_loss, 0.0, comm_bytes)
        
        return fl.common.ndarrays_to_parameters(aggregated_weights), metrics
    
    def aggregate_weights(self, results):
        """Weighted averaging"""
        total_examples = sum([num_examples for _, num_examples in results])
        aggregated = [np.zeros_like(w) for w in results[0][0]]
        
        for weights, num_examples in results:
            for i, w in enumerate(weights):
                aggregated[i] += w * (num_examples / total_examples)
        
        return aggregated

def run_lwfedssl_server():
    """Run LW-FedSSL server with all enhancements"""
    
    print(f"\n{'='*80}")
    print(f"🚀 Enhanced LW-FedSSL Server Starting")
    print(f"{'='*80}\n")
    
    # Initialize telemetry
    telemetry = get_telemetry(CONFIG.LOG_DIR)
    
    # Initialize model
    model = TinyCNN()
    
    # Initialize client selector
    client_selector = get_selector(
        strategy=CONFIG.CLIENT_SELECTION_STRATEGY,
        num_clients=CONFIG.NUM_CLIENTS,
        min_clients=CONFIG.MIN_CLIENTS_PER_ROUND,
        max_clients=CONFIG.MAX_CLIENTS_PER_ROUND
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Train each stage
    for stage in range(1, CONFIG.NUM_STAGES + 1):
        print(f"\n{'='*80}")
        print(f"📚 Starting Stage {stage}/{CONFIG.NUM_STAGES}")
        print(f"{'='*80}\n")
        
        # Create strategy with more flexible settings for testing
        strategy = EnhancedLWFedSSLStrategy(
            model=model,
            current_stage=stage,
            client_selector=client_selector,
            metrics_tracker=metrics_tracker,
            fraction_fit=0.5,  # Only require 50% of clients
            fraction_evaluate=0.0,
            min_fit_clients=min(5, CONFIG.MIN_CLIENTS_PER_ROUND),  # Lower minimum for testing
            min_available_clients=min(5, CONFIG.MIN_CLIENTS_PER_ROUND)  # Lower minimum
        )
        
        # Server config with extended timeout
        config = ServerConfig(
            num_rounds=CONFIG.ROUNDS_PER_STAGE,
            round_timeout=300.0  # 5 minutes timeout per round
        )
        
        # Start server
        fl.server.start_server(
            server_address=f"{CONFIG.SERVER_ADDRESS}:{CONFIG.SERVER_PORT}",
            config=config,
            strategy=strategy
        )
        
        print(f"\n✅ Stage {stage} Complete!")
        print(f"Total Communication: {strategy.total_communication / (1024**2):.2f} MB\n")

    
    # Save final model
    checkpoint_dir = Path(CONFIG.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), checkpoint_dir / "lwfedssl_final.pt")
    
    # Generate summary
    print(f"\n{'='*80}")
    print(f"🎉 LW-FedSSL Training Complete!")
    print(f"{'='*80}\n")
    
    summary = telemetry.save_summary()
    metrics_summary = metrics_tracker.get_summary()
    
    # Save metrics
    import json
    with open(checkpoint_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n📊 Metrics saved to: {checkpoint_dir / 'metrics_summary.json'}")
    print(f"💾 Model saved to: {checkpoint_dir / 'lwfedssl_final.pt'}\n")

def run_baseline_server():
    """Run baseline FedSSL for comparison"""
    
    print(f"\n{'='*80}")
    print(f"🔄 Running Baseline FedSSL for Comparison")
    print(f"{'='*80}\n")
    
    telemetry = get_telemetry(CONFIG.LOG_DIR)
    model = TinyCNN()
    metrics_tracker = MetricsTracker()
    
    strategy = BaselineFedSSLStrategy(
        model=model,
        metrics_tracker=metrics_tracker,
        fraction_fit=0.5,  # Only require 50% of clients
        min_fit_clients=min(5, CONFIG.MIN_CLIENTS_PER_ROUND)  # Lower minimum
    )
    
    config = ServerConfig(
        num_rounds=CONFIG.BASELINE_ROUNDS,
        round_timeout=300.0  # 5 minutes timeout per round
    )
    
    fl.server.start_server(
        server_address=f"{CONFIG.SERVER_ADDRESS}:{CONFIG.SERVER_PORT + 1}",
        config=config,
        strategy=strategy
    )
    
    # Save baseline results
    checkpoint_dir = Path(CONFIG.CHECKPOINT_DIR)
    torch.save(model.state_dict(), checkpoint_dir / "baseline_final.pt")
    
    import json
    with open(checkpoint_dir / "baseline_metrics.json", "w") as f:
        json.dump(metrics_tracker.get_summary(), f, indent=2)
    
    print(f"\n✅ Baseline FedSSL Complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LW-FedSSL Server")
    parser.add_argument("--mode", type=str, default="lwfedssl", 
                       choices=["lwfedssl", "baseline", "both"],
                       help="Run mode: lwfedssl, baseline, or both")
    args = parser.parse_args()
    
    if args.mode == "lwfedssl":
        run_lwfedssl_server()
    elif args.mode == "baseline":
        run_baseline_server()
    else:
        # Run both for comparison
        run_lwfedssl_server()
        print("\n\n")
        run_baseline_server()
