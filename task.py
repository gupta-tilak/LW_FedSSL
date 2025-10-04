"""
Shared model and data loading utilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class TinyCNN(nn.Module):
    """3-layer CNN with projection heads for different depths"""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 32, 3, padding=1),  # Layer 1
            nn.Conv2d(32, 64, 3, padding=1),  # Layer 2
            nn.Conv2d(64, 128, 3, padding=1)  # Layer 3
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection heads for different depths
        self.proj_heads = nn.ModuleDict({
            '1': nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            ),
            '2': nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            ),
            '3': nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ),
        })
    
    def forward(self, x, depth):
        """Forward pass up to specified depth"""
        for i in range(depth):
            x = F.relu(self.layers[i](x))
        x = self.pool(x)
        return self.proj_heads[str(depth)](x)


def get_client_dataloader(client_id: int, batch_size: int = 128):
    """Load CIFAR-10 data for specific client"""
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load full CIFAR-10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Split into 2 clients
    n = len(trainset)
    c1_size = n // 2
    c2_size = n - c1_size
    client1_ds, client2_ds = random_split(trainset, [c1_size, c2_size], 
                                          generator=torch.Generator().manual_seed(42))
    
    # Return appropriate client data
    if client_id == 1:
        return DataLoader(client1_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        return DataLoader(client2_ds, batch_size=batch_size, shuffle=True, num_workers=2)
