"""
LW-FedSSL Client Application - Fixed for Initial Parameters
Deploy this on Google Colab or Kaggle notebooks
"""
import flwr as fl
from flwr.client import NumPyClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import numpy as np
import time

class LWFedSSLClient(NumPyClient):
    """Flower client for layer-wise federated SSL training"""
    
    def __init__(self, model, train_loader, stage: int, local_epochs: int = 3, device: str = "cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.stage = stage
        self.local_epochs = local_epochs
        self.device = device
        
        # IMPORTANT: Initialize model weights if not already done
        self._initialize_model()
        print(f"Client initialized on device: {device}")
        
    def _initialize_model(self):
        """Initialize model weights properly"""
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
        """Return current layer parameters - CRITICAL for server initialization"""
        layer_idx = self.stage - 1
        layer = self.model.layers[layer_idx]
        
        params = [
            layer.weight.data.cpu().numpy(),
            layer.bias.data.cpu().numpy()
        ]
        
        print(f"Sending parameters for layer {layer_idx} (shapes: {[p.shape for p in params]})")
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Update layer parameters from server"""
        layer_idx = self.stage - 1
        layer = self.model.layers[layer_idx]
        layer.weight.data = torch.tensor(parameters[0], dtype=torch.float32).to(self.device)
        layer.bias.data = torch.tensor(parameters[1], dtype=torch.float32).to(self.device)
        print(f"Received parameters for layer {layer_idx}")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        
        print(f"\n{'='*50}")
        print(f"Starting training for Stage {self.stage}")
        print(f"{'='*50}")
        
        # Update model with server parameters
        self.set_parameters(parameters)
        
        # Freeze all layers except current stage
        self.freeze_old_layers(self.stage - 1)
        
        # Setup optimizer for trainable parameters only
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
        optimizer = optim.Adam(trainable_params, lr=1e-3)
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (xb, _) in enumerate(self.train_loader):
                # SSL augmentation
                x1, x2 = self.ssl_batch_augment(xb)
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                # Forward pass through current depth
                z1 = self.model(x1, depth=self.stage)
                z2 = self.model(x2, depth=self.stage)
                
                # Contrastive loss
                loss = self.simclr_nt_xent(z1, z2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                epoch_batches += 1
                
                # Print progress every 20 batches
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}/{self.local_epochs}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            print(f"Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"\n{'='*50}")
        print(f"Stage {self.stage} Training Complete!")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Total Batches: {num_batches}")
        print(f"{'='*50}\n")
        
        # Return updated parameters
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": float(avg_loss)}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model (optional - not used in SSL)"""
        return 0.0, 0, {}
    
    def freeze_old_layers(self, active_layer_idx: int):
        """Freeze all layers except the active one"""
        for i, layer in enumerate(self.model.layers):
            for p in layer.parameters():
                p.requires_grad = (i == active_layer_idx)
        
        # Keep projection heads trainable
        for proj_head in self.model.proj_heads.values():
            for p in proj_head.parameters():
                p.requires_grad = True
    
    def ssl_batch_augment(self, batch_images):
        """Apply SSL augmentations to create two views"""
        from torchvision import transforms
        
        ssl_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
        ])
        
        return (
            torch.stack([ssl_aug(img) for img in batch_images]),
            torch.stack([ssl_aug(img) for img in batch_images])
        )
    
    def simclr_nt_xent(self, z1, z2, tau=0.5):
        """SimCLR contrastive loss (NT-Xent)"""
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        batch_size = z1.size(0)
        logits = torch.mm(z1, z2.t()) / tau
        labels = torch.arange(batch_size, device=z1.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


def start_client(server_address: str, client_id: int, stage: int):
    """Start Flower client and connect to server"""
    from task import TinyCNN, get_client_dataloader
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Client {client_id} Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Server: {server_address}")
    print(f"Training Stage: {stage}")
    print(f"{'='*60}\n")
    
    # Initialize model and data
    print("Initializing model...")
    model = TinyCNN()
    
    print("Loading client dataset...")
    train_loader = get_client_dataloader(client_id, batch_size=128)
    print(f"Dataset size: {len(train_loader.dataset)} samples")
    
    # Create client
    client = LWFedSSLClient(
        model=model,
        train_loader=train_loader,
        stage=stage,
        local_epochs=3,
        device=device
    )
    
    print(f"\nConnecting to server at {server_address}...")
    
    # Connect to server
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        print("\n✅ Client training completed successfully!")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting checklist:")
        print("1. ✓ Server is running on MacBook (check terminal)")
        print("2. ✓ ngrok tunnel is active (check ngrok terminal)")
        print("3. ✓ Server address is correct (copy from ngrok)")
        print("4. ✓ Port 8080 is not blocked by firewall")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LW-FedSSL Flower Client")
    parser.add_argument("--server", type=str, required=True, help="Server address (e.g., '0.tcp.in.ngrok.io:17027')")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (1 or 2)")
    parser.add_argument("--stage", type=int, required=True, help="Training stage (1, 2, or 3)")
    args = parser.parse_args()
    
    start_client(args.server, args.client_id, args.stage)
