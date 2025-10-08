"""
Configuration file for LW-FedSSL Enhanced System
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SystemConfig:
    """Global system configuration"""
    # Client settings
    NUM_CLIENTS: int = 40
    MIN_CLIENTS_PER_ROUND: int = 10
    MAX_CLIENTS_PER_ROUND: int = 20
    CLIENT_SELECTION_STRATEGY: str = "random"  # Options: random, performance_based, diversity_based
    
    # Training settings
    NUM_STAGES: int = 3
    ROUNDS_PER_STAGE: int = 10
    LOCAL_EPOCHS: int = 3
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1e-3
    
    # Model settings
    MODEL_CHANNELS: List[int] = None
    PROJECTION_DIM: int = 128
    
    # SSL settings
    TEMPERATURE: float = 0.5
    AUGMENTATION_STRENGTH: float = 0.4
    
    # Data settings
    DATASET: str = "CIFAR10"
    DATA_DISTRIBUTION: str = "iid"  # Options: iid, non_iid_label, non_iid_dirichlet
    ALPHA: float = 0.5  # For Dirichlet distribution
    
    # Communication settings
    SERVER_ADDRESS: str = "0.0.0.0"
    SERVER_PORT: int = 8080
    
    # Telemetry settings
    TELEMETRY_ENABLED: bool = True
    TELEMETRY_UPDATE_INTERVAL: int = 5  # seconds
    LOG_DIR: str = "./logs"
    CHECKPOINT_DIR: str = "./checkpoints"
    
    # Metrics settings
    COMPUTE_FLOPS: bool = True
    COMPUTE_REPRESENTATION_QUALITY: bool = True
    LINEAR_EVAL_INTERVAL: int = 5  # Run linear evaluation every N rounds
    
    # Baseline comparison
    RUN_BASELINE: bool = True
    BASELINE_ROUNDS: int = 30  # Total rounds for baseline FedSSL
    
    def __post_init__(self):
        if self.MODEL_CHANNELS is None:
            self.MODEL_CHANNELS = [32, 64, 128]

@dataclass
class ClientConfig:
    """Individual client configuration"""
    client_id: int
    data_size: int
    device: str = "cpu"
    compute_power: float = 1.0  # Relative compute power (0-1)
    bandwidth: float = 1.0  # Relative bandwidth (0-1)
    reliability: float = 1.0  # Historical reliability score (0-1)
    
@dataclass
class MetricsConfig:
    """Metrics tracking configuration"""
    # Basic metrics
    track_loss: bool = True
    track_communication: bool = True
    track_time: bool = True
    
    # Advanced metrics
    track_convergence_rate: bool = True
    track_client_contribution: bool = True
    track_gradient_diversity: bool = True
    track_representation_quality: bool = True
    
    # Comparison metrics
    track_baseline_comparison: bool = True
    track_layer_importance: bool = True
    track_aggregation_variance: bool = True

# Global configuration instance
CONFIG = SystemConfig()
METRICS_CONFIG = MetricsConfig()
