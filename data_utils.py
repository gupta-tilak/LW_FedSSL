"""
Data Partitioning Utilities for Federated Learning
Supports IID and Non-IID data distributions across clients
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Dict, Tuple
from collections import defaultdict

class DataPartitioner:
    """Partition data across clients with various distribution strategies"""
    
    def __init__(self, dataset: Dataset, num_clients: int, 
                 distribution: str = 'iid', alpha: float = 0.5, seed: int = 42):
        """
        Args:
            dataset: PyTorch dataset to partition
            num_clients: Number of clients
            distribution: 'iid', 'non_iid_label', or 'non_iid_dirichlet'
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.distribution = distribution
        self.alpha = alpha
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.client_indices = self._partition_data()
    
    def _partition_data(self) -> Dict[int, List[int]]:
        """Partition dataset according to specified distribution"""
        if self.distribution == 'iid':
            return self._iid_partition()
        elif self.distribution == 'non_iid_label':
            return self._non_iid_label_partition()
        elif self.distribution == 'non_iid_dirichlet':
            return self._non_iid_dirichlet_partition()
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def _iid_partition(self) -> Dict[int, List[int]]:
        """IID partition: randomly distribute data equally"""
        n = len(self.dataset)
        indices = np.random.permutation(n)
        
        # Split into equal chunks
        splits = np.array_split(indices, self.num_clients)
        
        return {i: splits[i].tolist() for i in range(self.num_clients)}
    
    def _non_iid_label_partition(self) -> Dict[int, List[int]]:
        """
        Non-IID label partition: each client gets samples from only 2-3 classes
        """
        # Get labels
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        else:
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        
        num_classes = len(np.unique(labels))
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # Assign 2-3 classes to each client
        client_indices = {i: [] for i in range(self.num_clients)}
        classes_per_client = 2
        
        for i in range(self.num_clients):
            # Randomly select classes for this client
            selected_classes = np.random.choice(num_classes, classes_per_client, replace=False)
            
            for cls in selected_classes:
                # Get a portion of this class's data
                cls_data = class_indices[cls]
                n_samples = len(cls_data) // (self.num_clients // classes_per_client + 1)
                
                # Remove assigned samples from the pool
                assigned = cls_data[:n_samples]
                class_indices[cls] = cls_data[n_samples:]
                
                client_indices[i].extend(assigned)
        
        # Shuffle each client's data
        for i in range(self.num_clients):
            np.random.shuffle(client_indices[i])
        
        return client_indices
    
    def _non_iid_dirichlet_partition(self) -> Dict[int, List[int]]:
        """
        Non-IID Dirichlet partition: use Dirichlet distribution to create heterogeneous data
        Lower alpha = more non-IID
        """
        # Get labels
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        else:
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        
        num_classes = len(np.unique(labels))
        
        # Group indices by class
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        
        # Initialize client indices
        client_indices = {i: [] for i in range(self.num_clients)}
        
        # For each class, distribute samples to clients using Dirichlet
        for cls_idx, indices in enumerate(class_indices):
            np.random.shuffle(indices)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Distribute indices according to proportions
            splits = np.split(indices, 
                            (np.cumsum(proportions)[:-1] * len(indices)).astype(int))
            
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())
        
        # Shuffle each client's data
        for i in range(self.num_clients):
            np.random.shuffle(client_indices[i])
        
        return client_indices
    
    def get_client_loader(self, client_id: int, batch_size: int = 128, 
                         shuffle: bool = True) -> DataLoader:
        """Get DataLoader for a specific client"""
        indices = self.client_indices[client_id]
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    
    def get_client_dataset_info(self, client_id: int) -> Dict:
        """Get information about a client's dataset"""
        indices = self.client_indices[client_id]
        
        # Get labels for this client
        if hasattr(self.dataset, 'targets'):
            labels = np.array([self.dataset.targets[i] for i in indices])
        else:
            labels = np.array([self.dataset[i][1] for i in indices])
        
        # Count label distribution
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            'client_id': client_id,
            'num_samples': len(indices),
            'label_distribution': label_dist,
            'num_classes': len(unique)
        }
    
    def print_statistics(self):
        """Print partitioning statistics"""
        print(f"\n{'='*70}")
        print(f"Data Partitioning Statistics ({self.distribution})")
        print(f"{'='*70}")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Number of clients: {self.num_clients}")
        
        all_stats = []
        for i in range(self.num_clients):
            info = self.get_client_dataset_info(i)
            all_stats.append(info)
        
        # Summary statistics
        sample_counts = [s['num_samples'] for s in all_stats]
        print(f"\nSample distribution:")
        print(f"  Mean: {np.mean(sample_counts):.1f}")
        print(f"  Std: {np.std(sample_counts):.1f}")
        print(f"  Min: {np.min(sample_counts)}")
        print(f"  Max: {np.max(sample_counts)}")
        
        # Show first few clients
        print(f"\nFirst 5 clients:")
        for i in range(min(5, self.num_clients)):
            info = all_stats[i]
            print(f"\n  Client {i}:")
            print(f"    Samples: {info['num_samples']}")
            print(f"    Classes: {info['num_classes']}")
            print(f"    Distribution: {info['label_distribution']}")
        
        print(f"\n{'='*70}\n")

def get_cifar10_partitioned(num_clients: int = 40, 
                            distribution: str = 'iid',
                            alpha: float = 0.5,
                            batch_size: int = 128) -> Tuple[DataPartitioner, DataLoader]:
    """
    Get partitioned CIFAR-10 dataset and test loader
    
    Returns:
        partitioner: DataPartitioner object
        test_loader: Test data loader
    """
    # Base transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Partition training data
    partitioner = DataPartitioner(
        dataset=train_dataset,
        num_clients=num_clients,
        distribution=distribution,
        alpha=alpha
    )
    
    # Test loader (same for all)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return partitioner, test_loader

def get_ssl_augmentation():
    """Get SSL augmentation transform for SimCLR"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
