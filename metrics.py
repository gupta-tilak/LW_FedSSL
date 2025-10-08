"""
Advanced Metrics Tracker for LW-FedSSL
Implements comprehensive metrics beyond basic loss and communication
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score
import time

class MetricsTracker:
    """Track comprehensive metrics for federated learning"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = defaultdict(list)
        self.client_contributions = defaultdict(float)
        self.gradient_history = []
        self.representation_history = []
        
    def update_basic_metrics(self, loss: float, accuracy: float, comm_bytes: float):
        """Update basic metrics"""
        self.metrics['loss'].append(loss)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['communication_mb'].append(comm_bytes / (1024**2))
    
    def compute_convergence_rate(self, window_size: int = 5) -> float:
        """Compute convergence rate based on loss reduction"""
        if len(self.metrics['loss']) < window_size:
            return 0.0
        
        recent_losses = self.metrics['loss'][-window_size:]
        if len(recent_losses) < 2:
            return 0.0
        
        # Compute rate of loss decrease
        convergence_rate = -(recent_losses[-1] - recent_losses[0]) / window_size
        return float(convergence_rate)
    
    def compute_client_contribution(self, client_id: int, 
                                    client_params: List[np.ndarray],
                                    global_params: List[np.ndarray],
                                    num_samples: int) -> float:
        """
        Compute client contribution score based on:
        1. Parameter update magnitude
        2. Direction alignment with global update
        3. Data quantity
        """
        # Parameter update magnitude
        param_diff = sum([np.linalg.norm(cp - gp) for cp, gp in zip(client_params, global_params)])
        
        # Normalize by number of parameters
        total_params = sum([p.size for p in client_params])
        normalized_diff = param_diff / total_params if total_params > 0 else 0
        
        # Weight by data quantity
        contribution = normalized_diff * np.log1p(num_samples)
        
        self.client_contributions[client_id] += contribution
        return float(contribution)
    
    def compute_gradient_diversity(self, client_gradients: List[List[np.ndarray]]) -> float:
        """
        Compute gradient diversity across clients
        Higher diversity indicates more varied local data distributions
        """
        if len(client_gradients) < 2:
            return 0.0
        
        # Flatten gradients for each client
        flat_grads = []
        for grads in client_gradients:
            flat_grad = np.concatenate([g.flatten() for g in grads])
            flat_grads.append(flat_grad)
        
        # Compute pairwise cosine distances
        distances = []
        for i in range(len(flat_grads)):
            for j in range(i + 1, len(flat_grads)):
                dist = cosine(flat_grads[i], flat_grads[j])
                if not np.isnan(dist):
                    distances.append(dist)
        
        diversity = np.mean(distances) if distances else 0.0
        self.metrics['gradient_diversity'].append(float(diversity))
        return float(diversity)
    
    def compute_representation_quality(self, model: nn.Module, 
                                       data_loader, 
                                       depth: int,
                                       device: str = 'cpu') -> Dict[str, float]:
        """
        Compute representation quality metrics:
        1. Silhouette score (cluster quality)
        2. Representation variance
        3. Dimension utilization
        """
        model.eval()
        representations = []
        labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                z = model(batch_x, depth=depth)
                representations.append(z.cpu().numpy())
                labels.append(batch_y.numpy())
        
        representations = np.concatenate(representations, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # Compute metrics
        metrics = {}
        
        # 1. Silhouette score (if we have enough samples)
        if len(representations) >= 100 and len(np.unique(labels)) > 1:
            try:
                # Sample for efficiency
                indices = np.random.choice(len(representations), min(1000, len(representations)), replace=False)
                sil_score = silhouette_score(representations[indices], labels[indices], metric='cosine')
                metrics['silhouette_score'] = float(sil_score)
            except:
                metrics['silhouette_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0
        
        # 2. Representation variance (how spread out are the representations)
        metrics['representation_variance'] = float(np.var(representations))
        
        # 3. Dimension utilization (how many dimensions are actively used)
        dim_variances = np.var(representations, axis=0)
        active_dims = np.sum(dim_variances > 0.01)  # Threshold for "active"
        metrics['dimension_utilization'] = float(active_dims / len(dim_variances))
        
        # 4. Mean norm of representations
        metrics['mean_representation_norm'] = float(np.mean(np.linalg.norm(representations, axis=1)))
        
        self.representation_history.append(metrics)
        return metrics
    
    def compute_layer_importance(self, model: nn.Module, 
                                 layer_idx: int,
                                 data_loader,
                                 device: str = 'cpu') -> float:
        """
        Compute importance of a specific layer using gradient-based sensitivity
        """
        model.eval()
        layer = model.layers[layer_idx]
        
        # Enable gradients for this layer
        for param in layer.parameters():
            param.requires_grad = True
        
        total_gradient_norm = 0.0
        num_batches = 0
        
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            batch_x.requires_grad = True
            
            # Forward pass through all layers up to current depth
            x = batch_x
            for i in range(layer_idx + 1):
                x = torch.relu(model.layers[i](x))
            
            # Compute gradient of output w.r.t. layer parameters
            output_norm = torch.norm(x)
            output_norm.backward()
            
            # Aggregate gradient norms
            for param in layer.parameters():
                if param.grad is not None:
                    total_gradient_norm += torch.norm(param.grad).item()
            
            num_batches += 1
            if num_batches >= 10:  # Sample only a few batches for efficiency
                break
        
        importance = total_gradient_norm / num_batches if num_batches > 0 else 0.0
        return float(importance)
    
    def compute_aggregation_variance(self, client_params_list: List[List[np.ndarray]]) -> Dict[str, float]:
        """
        Compute variance in client parameters before aggregation
        High variance indicates heterogeneous updates
        """
        if len(client_params_list) < 2:
            return {'param_variance': 0.0, 'param_std': 0.0}
        
        # Stack parameters from all clients
        variances = []
        for param_idx in range(len(client_params_list[0])):
            param_stack = np.stack([client_params[param_idx] for client_params in client_params_list])
            param_var = np.var(param_stack)
            variances.append(param_var)
        
        metrics = {
            'param_variance': float(np.mean(variances)),
            'param_std': float(np.sqrt(np.mean(variances))),
            'max_param_variance': float(np.max(variances)),
            'min_param_variance': float(np.min(variances))
        }
        
        return metrics
    
    def compute_communication_efficiency(self) -> Dict[str, float]:
        """Compute communication efficiency metrics"""
        if not self.metrics['communication_mb']:
            return {'efficiency': 0.0}
        
        total_comm = sum(self.metrics['communication_mb'])
        num_rounds = len(self.metrics['communication_mb'])
        
        # Loss reduction per MB communicated
        if len(self.metrics['loss']) >= 2:
            initial_loss = self.metrics['loss'][0]
            final_loss = self.metrics['loss'][-1]
            loss_reduction = initial_loss - final_loss
            efficiency = loss_reduction / total_comm if total_comm > 0 else 0
        else:
            efficiency = 0
        
        return {
            'communication_efficiency': float(efficiency),
            'avg_comm_per_round': float(total_comm / num_rounds) if num_rounds > 0 else 0,
            'total_communication_mb': float(total_comm)
        }
    
    def compute_flops(self, model: nn.Module, 
                     input_shape: Tuple[int, ...],
                     depth: int) -> float:
        """
        Compute FLOPs for forward pass through model up to specified depth
        """
        # Simplified FLOP calculation
        flops = 0
        current_channels = input_shape[0]
        spatial_size = input_shape[1] * input_shape[2]
        
        for i in range(depth):
            layer = model.layers[i]
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
                # FLOPs = 2 * kernel_size * in_channels * out_channels * output_spatial_size
                flops += 2 * kernel_size * current_channels * out_channels * spatial_size
                current_channels = out_channels
        
        return float(flops)
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of all tracked metrics"""
        summary = {
            'basic_metrics': {
                key: {
                    'mean': float(np.mean(values)) if values else 0,
                    'std': float(np.std(values)) if values else 0,
                    'final': float(values[-1]) if values else 0,
                    'history': [float(v) for v in values]
                }
                for key, values in self.metrics.items()
            },
            'client_contributions': dict(self.client_contributions),
            'convergence_rate': self.compute_convergence_rate(),
            'communication_efficiency': self.compute_communication_efficiency()
        }
        
        if self.representation_history:
            summary['representation_quality'] = {
                key: float(np.mean([r[key] for r in self.representation_history]))
                for key in self.representation_history[0].keys()
            }
        
        return summary

class ComparisonMetrics:
    """Compare LW-FedSSL with baseline FedSSL"""
    
    def __init__(self):
        self.lwfedssl_metrics = MetricsTracker()
        self.baseline_metrics = MetricsTracker()
    
    def update_lwfedssl(self, **kwargs):
        """Update LW-FedSSL metrics"""
        if 'loss' in kwargs and 'accuracy' in kwargs and 'comm_bytes' in kwargs:
            self.lwfedssl_metrics.update_basic_metrics(
                kwargs['loss'], kwargs['accuracy'], kwargs['comm_bytes']
            )
    
    def update_baseline(self, **kwargs):
        """Update baseline metrics"""
        if 'loss' in kwargs and 'accuracy' in kwargs and 'comm_bytes' in kwargs:
            self.baseline_metrics.update_basic_metrics(
                kwargs['loss'], kwargs['accuracy'], kwargs['comm_bytes']
            )
    
    def get_comparison(self) -> Dict[str, any]:
        """Get comparative analysis"""
        lw_summary = self.lwfedssl_metrics.get_summary()
        baseline_summary = self.baseline_metrics.get_summary()
        
        comparison = {
            'lwfedssl': lw_summary,
            'baseline': baseline_summary,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in ['loss', 'communication_mb']:
            if metric in lw_summary['basic_metrics'] and metric in baseline_summary['basic_metrics']:
                lw_val = lw_summary['basic_metrics'][metric]['final']
                baseline_val = baseline_summary['basic_metrics'][metric]['final']
                if baseline_val != 0:
                    improvement = ((baseline_val - lw_val) / baseline_val) * 100
                    comparison['improvements'][metric] = f"{improvement:.2f}%"
        
        return comparison
