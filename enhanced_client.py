"""
Enhanced Client Application for LW-FedSSL
Supports 40+ clients with advanced features
"""
import flwr as fl
from flwr.client import NumPyClient
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np
import time
import argparse

from task import TinyCNN
from data_utils import DataPartitioner, get_ssl_augmentation
from config import CONFIG

class EnhancedLWFedSSLClient(NumPyClient):
    """Enhanced Flower client with advanced features"""
    
    def __init__(self, client_id: int, model: nn.Module, train_loader, 
                 device: str = "cpu", telemetry_enabled: bool = True):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.telemetry_enabled = telemetry_enabled
        
        # Initialize model weights
        self._initialize_model()
        
        # SSL augmentation
        self.ssl_aug = get_ssl_augmentation()
        
        print(f"✅ Client {client_id} initialized on {device}")
    
    def _initialize_model(self):
        """Initialize model weights"""
        for layer in self.model.layers:
            if hasattr(layer, 'weight'):
                nn.init.kaiming_normal_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        for proj_head in self.model.proj_heads.values():
            for module in proj_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return layer parameters"""
        stage = config.get("stage", 1)
        layer_idx = stage - 1
        layer = self.model.layers[layer_idx]
        
        return [
            layer.weight.data.cpu().numpy(),
            layer.bias.data.cpu().numpy()
        ]
    
    def set_parameters(self, parameters: List[np.ndarray], config: Dict):
        """Update layer parameters from server"""
        stage = config.get("stage", 1)
        layer_idx = stage - 1
        layer = self.model.layers[layer_idx]
        
        layer.weight.data = torch.tensor(parameters[0], dtype=torch.float32).to(self.device)
        layer.bias.data = torch.tensor(parameters[1], dtype=torch.float32).to(self.device)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        try:
            print(f"\n🔵 Client {self.client_id} - fit() called with config: {config}")
            
            start_time = time.time()
            
            stage = config.get("stage", 1)
            local_epochs = config.get("local_epochs", CONFIG.LOCAL_EPOCHS)
            learning_rate = config.get("learning_rate", CONFIG.LEARNING_RATE)
            
            print(f"\n{'='*60}")
            print(f"🔄 Client {self.client_id} - Stage {stage} Training")
            print(f"{'='*60}")
            
            # Update parameters
            self.set_parameters(parameters, config)
            
            # Freeze old layers
            self._freeze_layers(stage - 1)
            
            # Setup optimizer
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.Adam(trainable_params, lr=learning_rate)
            
            # Training loop
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(local_epochs):
                epoch_loss = 0.0
                
                for batch_idx, (xb, _) in enumerate(self.train_loader):
                    # SSL augmentation
                    x1, x2 = self._ssl_augment(xb)
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    
                    # Forward pass
                    z1 = self.model(x1, depth=stage)
                    z2 = self.model(x2, depth=stage)
                    
                    # Contrastive loss
                    loss = self._simclr_loss(z1, z2)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                print(f"  Epoch {epoch+1}/{local_epochs}: Loss = {avg_epoch_loss:.4f}")
            
            training_time = time.time() - start_time
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            print(f"\n✅ Training complete!")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Time: {training_time:.2f}s")
            print(f"{'='*60}\n")
            
            # Prepare metrics
            metrics = {
                "loss": float(avg_loss),
                "training_time": float(training_time),
                "num_batches": num_batches,
                "client_id": self.client_id
            }
            
            # Return updated parameters
            return (
                self.get_parameters(config),
                len(self.train_loader.dataset),
                metrics
            )
        except Exception as e:
            print(f"\n❌ Client {self.client_id} - fit() FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model (optional)"""
        return 0.0, 0, {}
    
    def _freeze_layers(self, active_layer_idx: int):
        """Freeze all layers except active one"""
        for i, layer in enumerate(self.model.layers):
            for p in layer.parameters():
                p.requires_grad = (i == active_layer_idx)
        
        # Keep projection heads trainable
        for proj_head in self.model.proj_heads.values():
            for p in proj_head.parameters():
                p.requires_grad = True
    
    def _ssl_augment(self, batch_images):
        """Apply SSL augmentations"""
        return (
            torch.stack([self.ssl_aug(img) for img in batch_images]),
            torch.stack([self.ssl_aug(img) for img in batch_images])
        )
    
    def _simclr_loss(self, z1, z2, tau=0.5):
        """SimCLR NT-Xent loss"""
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        batch_size = z1.size(0)
        logits = torch.mm(z1, z2.t()) / tau
        labels = torch.arange(batch_size, device=z1.device)
        
        return nn.functional.cross_entropy(logits, labels)

class BaselineClient(NumPyClient):
    """Baseline FedSSL client (full model training)"""
    
    def __init__(self, client_id: int, model: nn.Module, train_loader, device: str = "cpu"):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.ssl_aug = get_ssl_augmentation()
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model"""
        for layer in self.model.layers:
            if hasattr(layer, 'weight'):
                nn.init.kaiming_normal_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return all model parameters"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set all model parameters"""
        params_dict = self.model.state_dict()
        keys = list(params_dict.keys())
        for i, key in enumerate(keys):
            if i < len(parameters):
                params_dict[key] = torch.tensor(parameters[i])
        self.model.load_state_dict(params_dict)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train full model"""
        start_time = time.time()
        
        self.set_parameters(parameters)
        
        # Train full model at depth 3
        optimizer = optim.Adam(self.model.parameters(), lr=CONFIG.LEARNING_RATE)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(CONFIG.LOCAL_EPOCHS):
            for xb, _ in self.train_loader:
                x1 = torch.stack([self.ssl_aug(img) for img in xb]).to(self.device)
                x2 = torch.stack([self.ssl_aug(img) for img in xb]).to(self.device)
                
                z1 = self.model(x1, depth=3)
                z2 = self.model(x2, depth=3)
                
                # SimCLR loss
                z1 = nn.functional.normalize(z1, dim=1)
                z2 = nn.functional.normalize(z2, dim=1)
                logits = torch.mm(z1, z2.t()) / 0.5
                labels = torch.arange(z1.size(0), device=self.device)
                loss = nn.functional.cross_entropy(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        training_time = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"loss": float(avg_loss), "training_time": float(training_time)}
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        return 0.0, 0, {}

def start_client(server_address: str, client_id: int, 
                partitioner: DataPartitioner, mode: str = "lwfedssl"):
    """Start Flower client"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"🚀 Client {client_id} Starting")
    print(f"{'='*70}")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Server: {server_address}")
    
    # Get data
    train_loader = partitioner.get_client_loader(client_id, batch_size=CONFIG.BATCH_SIZE)
    client_info = partitioner.get_client_dataset_info(client_id)
    
    print(f"Dataset: {client_info['num_samples']} samples, {client_info['num_classes']} classes")
    print(f"{'='*70}\n")
    
    # Initialize model
    model = TinyCNN()
    
    # Create client
    if mode == "lwfedssl":
        client = EnhancedLWFedSSLClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            device=device
        )
    else:
        client = BaselineClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            device=device
        )
    
    # Connect to server
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        print(f"\n✅ Client {client_id} completed successfully!")
    except Exception as e:
        print(f"\n❌ Client {client_id} failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced LW-FedSSL Client")
    parser.add_argument("--server", type=str, required=True, 
                       help="Server address (e.g., 'localhost:8080')")
    parser.add_argument("--client-id", type=int, required=True, 
                       help="Client ID (0-39)")
    parser.add_argument("--mode", type=str, default="lwfedssl",
                       choices=["lwfedssl", "baseline"],
                       help="Training mode")
    parser.add_argument("--data-distribution", type=str, default="iid",
                       choices=["iid", "non_iid_label", "non_iid_dirichlet"],
                       help="Data distribution type")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Dirichlet alpha for non-IID distribution")
    
    args = parser.parse_args()
    
    # Load and partition data
    from data_utils import get_cifar10_partitioned
    
    partitioner, _ = get_cifar10_partitioned(
        num_clients=CONFIG.NUM_CLIENTS,
        distribution=args.data_distribution,
        alpha=args.alpha,
        batch_size=CONFIG.BATCH_SIZE
    )
    
    # Start client
    start_client(args.server, args.client_id, partitioner, args.mode)
