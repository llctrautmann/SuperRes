from dataclasses import dataclass
import torch


@dataclass
class HyperParams:
    # Misc
    dim_test: bool = True

    # Training
    epochs: int = 100
    lr: float = 0.001
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    save_model: bool = True
    save_path: str = "./params/"

    # Data
    data_dir: str = "./data/cryptopunks/"
    batch_size: int = 16
    num_workers: int = 1
    pin_memory: bool = True

hyperparams = HyperParams()
