import torch
import random
import math
from typing import Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

set_seed = lambda seed: random.seed(seed)

@dataclass
class DataConfig:
    d: int = 1000
    sparsity: int = 2
    noise_variance: float = 0.03
    n_train: int = 100_000
    n_val: int = 10_000
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TwoHotStream:
    """
    Generates B samples with s active coordinates (set to 1) in a d-dimensional vector,
    plus iid Gaussian noise with variance eta2.
    Returns (x, x_star, mask_on) where x_star is the clean two-hot and mask_on is its mask.
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        random.seed(torch.initial_seed() & 0xFFFFFFFF)

    @torch.no_grad()
    def sample_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d, sparsity, eta2 = self.cfg.d, self.cfg.sparsity, self.cfg.noise_variance
        idx = torch.stack([torch.randperm(d, device=self.device)[:sparsity] for _ in range(B)], dim=0)
        x_star = torch.zeros(B, d, device=self.device)
        x_star.scatter_(1, idx, 1.0)
        noise = torch.randn(B, d, device=self.device) * math.sqrt(eta2)
        x = x_star + noise
        return x, x_star


class TwoHotDataset(Dataset):
    """
    PyTorch Dataset wrapper for TwoHotStream.
    Generates samples on-the-fly for each index access.
    """
    def __init__(self, stream: TwoHotStream, size: int):
        """
        Args:
            stream: TwoHotStream instance to generate samples from
            size: Number of samples in the dataset (epoch length)
        """
        self.stream = stream
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """
        Generate a single sample.
        Note: idx is ignored since we generate fresh samples each time.
        """
        # Generate a single sample (batch size = 1)
        x, x_star = self.stream.sample_batch(1)
        # Remove batch dimension
        return x.squeeze(0), x_star.squeeze(0)


def create_dataloaders(cfg: DataConfig, batch_size: int = 128, num_workers: int = 0):
    """
    Create train and validation dataloaders.
    
    Args:
        cfg: DataConfig with dataset parameters
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader
    """
    # Create separate streams for train and val (different seeds)
    train_cfg = DataConfig(
        d=cfg.d,
        sparsity=cfg.sparsity,
        noise_variance=cfg.noise_variance,
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        seed=cfg.seed,
        device=cfg.device
    )
    
    val_cfg = DataConfig(
        d=cfg.d,
        sparsity=cfg.sparsity,
        noise_variance=cfg.noise_variance,
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        seed=cfg.seed + 1,  # Different seed for validation
        device=cfg.device
    )
    
    train_stream = TwoHotStream(train_cfg)
    val_stream = TwoHotStream(val_cfg)
    
    # Create datasets
    train_dataset = TwoHotDataset(train_stream, cfg.n_train)
    val_dataset = TwoHotDataset(val_stream, cfg.n_val)
    
    # Create dataloaders
    # Samples are generated directly on cfg.device, so pin_memory must be False
    # (CUDA tensors cannot be pinned; pinning is for CPU→GPU staging).
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Configuration
    cfg = DataConfig(
        d=1000,
        sparsity=2,
        noise_variance=0.03,
        n_train=1000,  # Smaller for demo
        n_val=200,
        seed=42
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(cfg, batch_size=32)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of val batches: {len(val_loader)}")
    
    # Test iteration
    print("\nTesting train loader:")
    for batch_idx, (x, x_star) in enumerate(train_loader):
        print(f"Batch {batch_idx}: x.shape={x.shape}, x_star.shape={x_star.shape}")
        print(f"x: {x}")
        print(f"x_star: {x_star}")
        print(f"noise: {(x - x_star).sum()}")
        if batch_idx >= 10:  # Show only first 3 batches
            break
    

    # Tests
    # x_star is two_hot vector
    # there are no repeats