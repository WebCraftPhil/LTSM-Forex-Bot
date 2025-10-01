"""
Utilities for reproducible random seeds across the trading bot.
"""
import random
import numpy as np
import torch
import os
from typing import Optional

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for reproducible results across all libraries.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms in PyTorch
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for subprocesses
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Also set for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_rng_state() -> dict:
    """Get current random number generator states for saving."""
    return {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

def set_rng_state(state: dict) -> None:
    """Restore random number generator states."""
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'])

class SeedManager:
    """Context manager for temporary seed changes."""

    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.previous_state = None

    def __enter__(self):
        self.previous_state = get_rng_state()
        set_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_state is not None:
            set_rng_state(self.previous_state)
