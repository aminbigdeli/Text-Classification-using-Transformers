import torch
from torch.utils.data import DataLoader


class DataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle: str = False,
        num_workers: int = 0
        ):
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )

