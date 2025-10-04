"""
LW-FedSSL Server Application
Orchestrates federated learning rounds and aggregates layer updates
Runs on local MacBook Air - No GPU required
"""
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

class LWFedSSLStrategy(FedAvg):
    """Custom strategy for Layer-wise Federated SSL"""
    
    def __init__(self, model, current_stage: int, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.current_stage = current_stage
        self.comm_bytes = []
        self.round_flops = []
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate layer parameters from clients"""
        
        if not results:
            return None, {}
        
        # Extract layer weights from client results
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate using weighted average
        aggregated_weights = self.aggregate_weights(weights_results)
        
        # Update only the trained layer in global model
        layer_idx = self.current_stage - 1
        layer_state = {f"layers.{layer_idx}.weight": torch.tensor(aggregated_weights[0]),
                       f"layers.{layer_idx}.bias": torch.tensor(aggregated_weights[1])}
        
        self.model.load_state_dict(layer_state, strict=False)
        
        # Track communication costs
        params_size = sum([w.nbytes for w in aggregated_weights]) / (1024**2)  # MB
        self.comm_bytes.append(params_size * len(results))  # Upload + Download
        
        # Aggregate metrics
        metrics = {}
        if results:
            metrics["comm_mb"] = params_size * len(results)
            losses = [fit_res.metrics.get("loss", 0) for _, fit_res in results]
            metrics["avg_loss"] = sum(losses) / len(losses) if losses else 0
            
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


def start_server(model, stage: int, num_rounds: int = 8):
    """Start Flower server for given training stage"""
    
    # Configure strategy
    strategy = LWFedSSLStrategy(
        model=model,
        current_stage=stage,
        fraction_fit=1.0,  # Select all available clients
        fraction_evaluate=0.0,  # No evaluation during training
        min_fit_clients=1,  # Wait for at least 2 clients
        min_available_clients=1,
    )
    
    # Configure server
    config = ServerConfig(num_rounds=num_rounds)
    
    print(f"\n{'='*60}")
    print(f"Starting LW-FedSSL Server - Stage {stage}")
    print(f"Waiting for clients to connect...")
    print(f"{'='*60}\n")
    
    # Start server (will wait for clients to connect)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
    
    return strategy.comm_bytes, strategy.round_flops


if __name__ == "__main__":
    from task import TinyCNN
    
    # Initialize model
    model = TinyCNN()
    
    # Train each stage sequentially
    total_comm = []
    total_flops = []
    
    for stage in range(1, 4):  # 3 layers
        comm, flops = start_server(model, stage, num_rounds=8)
        total_comm.extend(comm)
        total_flops.extend(flops)
        
        print(f"\nStage {stage} completed!")
        print(f"Communication: {sum(comm):.2f} MB")
    
    print(f"\n{'='*60}")
    print(f"LW-FedSSL Training Complete!")
    print(f"Total Communication: {sum(total_comm):.2f} MB")
    print(f"{'='*60}\n")
