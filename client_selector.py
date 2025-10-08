"""
Client Selection Strategies for LW-FedSSL
Implements various strategies for dynamic client selection
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import ClientConfig
import random

@dataclass
class ClientSelectionResult:
    """Result of client selection"""
    selected_clients: List[int]
    selection_scores: Dict[int, float]
    strategy_used: str

class ClientSelector:
    """Base class for client selection strategies"""
    
    def __init__(self, num_clients: int, min_clients: int, max_clients: int):
        self.num_clients = num_clients
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.client_history = {i: [] for i in range(num_clients)}
        self.client_configs = {}
    
    def register_client(self, client_id: int, config: ClientConfig):
        """Register a client with its configuration"""
        self.client_configs[client_id] = config
    
    def update_client_performance(self, client_id: int, loss: float, 
                                  training_time: float, success: bool):
        """Update client performance history"""
        # Initialize history for new clients (handles UUID-based IDs)
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        
        self.client_history[client_id].append({
            'loss': loss,
            'training_time': training_time,
            'success': success,
            'timestamp': len(self.client_history[client_id])
        })
    
    def select(self, available_clients: List[int], round_num: int) -> ClientSelectionResult:
        """Select clients for the current round"""
        raise NotImplementedError

class RandomSelector(ClientSelector):
    """Random client selection"""
    
    def select(self, available_clients: List[int], round_num: int) -> ClientSelectionResult:
        num_select = random.randint(self.min_clients, 
                                   min(self.max_clients, len(available_clients)))
        selected = random.sample(available_clients, num_select)
        scores = {cid: 1.0 for cid in selected}
        
        return ClientSelectionResult(
            selected_clients=selected,
            selection_scores=scores,
            strategy_used="random"
        )

class PerformanceBasedSelector(ClientSelector):
    """Select clients based on historical performance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 0.7  # Weight for loss vs. speed
    
    def select(self, available_clients: List[int], round_num: int) -> ClientSelectionResult:
        scores = {}
        
        for cid in available_clients:
            if cid in self.client_history and self.client_history[cid]:
                history = self.client_history[cid]
                recent = history[-5:]  # Last 5 rounds
                
                # Average loss (lower is better)
                avg_loss = np.mean([h['loss'] for h in recent if 'loss' in h])
                
                # Average training time (lower is better)
                avg_time = np.mean([h['training_time'] for h in recent if 'training_time' in h])
                
                # Success rate
                success_rate = np.mean([h['success'] for h in recent if 'success' in h])
                
                # Combined score (higher is better)
                # Normalize and invert loss and time
                loss_score = 1.0 / (1.0 + avg_loss) if avg_loss > 0 else 1.0
                time_score = 1.0 / (1.0 + avg_time) if avg_time > 0 else 1.0
                
                score = self.alpha * loss_score + (1 - self.alpha) * time_score
                score *= success_rate
                
                scores[cid] = score
            else:
                # New client - give average score
                scores[cid] = 0.5
        
        # Select top performers
        sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        num_select = random.randint(self.min_clients, 
                                   min(self.max_clients, len(available_clients)))
        selected = [cid for cid, _ in sorted_clients[:num_select]]
        
        return ClientSelectionResult(
            selected_clients=selected,
            selection_scores=scores,
            strategy_used="performance_based"
        )

class DiversityBasedSelector(ClientSelector):
    """Select diverse set of clients to improve generalization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_embeddings = {}
    
    def update_client_embedding(self, client_id: int, gradient: np.ndarray):
        """Update client embedding based on gradient"""
        if client_id not in self.client_embeddings:
            self.client_embeddings[client_id] = []
        self.client_embeddings[client_id].append(gradient)
        
        # Keep only recent embeddings
        if len(self.client_embeddings[client_id]) > 10:
            self.client_embeddings[client_id] = self.client_embeddings[client_id][-10:]
    
    def compute_diversity_score(self, client_id: int, selected_clients: List[int]) -> float:
        """Compute diversity score for a client given already selected clients"""
        if client_id not in self.client_embeddings or not self.client_embeddings[client_id]:
            return 1.0  # New client gets high diversity
        
        if not selected_clients:
            return 1.0
        
        # Get average embedding for client
        client_emb = np.mean(self.client_embeddings[client_id], axis=0)
        
        # Compute distance to selected clients
        distances = []
        for sel_id in selected_clients:
            if sel_id in self.client_embeddings and self.client_embeddings[sel_id]:
                sel_emb = np.mean(self.client_embeddings[sel_id], axis=0)
                # Cosine distance
                dist = 1 - np.dot(client_emb, sel_emb) / (
                    np.linalg.norm(client_emb) * np.linalg.norm(sel_emb) + 1e-8
                )
                distances.append(dist)
        
        # Higher average distance = more diverse
        return np.mean(distances) if distances else 1.0
    
    def select(self, available_clients: List[int], round_num: int) -> ClientSelectionResult:
        num_select = random.randint(self.min_clients, 
                                   min(self.max_clients, len(available_clients)))
        
        selected = []
        scores = {}
        
        # Greedy selection for diversity
        remaining = available_clients.copy()
        
        while len(selected) < num_select and remaining:
            # Compute diversity scores for remaining clients
            div_scores = {}
            for cid in remaining:
                div_scores[cid] = self.compute_diversity_score(cid, selected)
            
            # Select most diverse client
            best_client = max(div_scores.items(), key=lambda x: x[1])[0]
            selected.append(best_client)
            scores[best_client] = div_scores[best_client]
            remaining.remove(best_client)
        
        return ClientSelectionResult(
            selected_clients=selected,
            selection_scores=scores,
            strategy_used="diversity_based"
        )

class HybridSelector(ClientSelector):
    """Hybrid strategy combining performance and diversity"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_selector = PerformanceBasedSelector(*args, **kwargs)
        self.diversity_selector = DiversityBasedSelector(*args, **kwargs)
        self.performance_weight = 0.6
    
    def update_client_performance(self, client_id: int, loss: float, 
                                  training_time: float, success: bool):
        super().update_client_performance(client_id, loss, training_time, success)
        self.performance_selector.update_client_performance(client_id, loss, training_time, success)
    
    def update_client_embedding(self, client_id: int, gradient: np.ndarray):
        self.diversity_selector.update_client_embedding(client_id, gradient)
    
    def select(self, available_clients: List[int], round_num: int) -> ClientSelectionResult:
        # Get scores from both strategies
        perf_result = self.performance_selector.select(available_clients, round_num)
        div_result = self.diversity_selector.select(available_clients, round_num)
        
        # Combine scores
        combined_scores = {}
        for cid in available_clients:
            perf_score = perf_result.selection_scores.get(cid, 0.5)
            div_score = div_result.selection_scores.get(cid, 0.5)
            combined_scores[cid] = (
                self.performance_weight * perf_score + 
                (1 - self.performance_weight) * div_score
            )
        
        # Select top clients
        sorted_clients = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        num_select = random.randint(self.min_clients, 
                                   min(self.max_clients, len(available_clients)))
        selected = [cid for cid, _ in sorted_clients[:num_select]]
        
        return ClientSelectionResult(
            selected_clients=selected,
            selection_scores=combined_scores,
            strategy_used="hybrid"
        )

class AdaptiveSelector(ClientSelector):
    """Adaptive selection that changes strategy based on training progress"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selectors = {
            'random': RandomSelector(*args, **kwargs),
            'performance': PerformanceBasedSelector(*args, **kwargs),
            'diversity': DiversityBasedSelector(*args, **kwargs),
            'hybrid': HybridSelector(*args, **kwargs)
        }
        self.current_strategy = 'random'
        self.strategy_schedule = {
            range(0, 5): 'random',      # Initial rounds: random
            range(5, 15): 'diversity',   # Early training: diversity
            range(15, 100): 'hybrid'     # Later: hybrid
        }
    
    def update_client_performance(self, client_id: int, loss: float, 
                                  training_time: float, success: bool):
        super().update_client_performance(client_id, loss, training_time, success)
        for selector in self.selectors.values():
            if hasattr(selector, 'update_client_performance'):
                selector.update_client_performance(client_id, loss, training_time, success)
    
    def update_client_embedding(self, client_id: int, gradient: np.ndarray):
        for selector in self.selectors.values():
            if hasattr(selector, 'update_client_embedding'):
                selector.update_client_embedding(client_id, gradient)
    
    def select(self, available_clients: List[int], round_num: int) -> ClientSelectionResult:
        # Determine strategy based on round number
        for round_range, strategy in self.strategy_schedule.items():
            if round_num in round_range:
                self.current_strategy = strategy
                break
        
        # Use selected strategy
        result = self.selectors[self.current_strategy].select(available_clients, round_num)
        result.strategy_used = f"adaptive_{self.current_strategy}"
        return result

def get_selector(strategy: str, num_clients: int, 
                min_clients: int, max_clients: int) -> ClientSelector:
    """Factory function to get appropriate selector"""
    selectors = {
        'random': RandomSelector,
        'performance_based': PerformanceBasedSelector,
        'diversity_based': DiversityBasedSelector,
        'hybrid': HybridSelector,
        'adaptive': AdaptiveSelector
    }
    
    if strategy not in selectors:
        print(f"Unknown strategy '{strategy}', using 'random'")
        strategy = 'random'
    
    return selectors[strategy](num_clients, min_clients, max_clients)
