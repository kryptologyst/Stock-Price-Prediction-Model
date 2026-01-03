"""Utility functions for reproducible experiments and device management."""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
    """
    from omegaconf import OmegaConf
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    from omegaconf import OmegaConf
    
    OmegaConf.save(config, save_path)


def create_directories(paths: Dict[str, str]) -> None:
    """Create necessary directories for the project.
    
    Args:
        paths: Dictionary of path names and their corresponding directories
    """
    for path_name, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
        print(f"Created directory: {path_value}")


def log_experiment_info(config: DictConfig, results: Dict[str, Any]) -> None:
    """Log experiment information for reproducibility.
    
    Args:
        config: Experiment configuration
        results: Experiment results
    """
    experiment_info = {
        "config": config,
        "results": results,
        "device": str(get_device()),
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
    }
    
    # Save to logs directory
    logs_dir = config.get("paths", {}).get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"experiment_{timestamp}.json")
    
    with open(log_file, "w") as f:
        json.dump(experiment_info, f, indent=2, default=str)
    
    print(f"Experiment logged to: {log_file}")


class EarlyStopping:
    """Early stopping utility for model training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[Any] = None) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
            
        return False
