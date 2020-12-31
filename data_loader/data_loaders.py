from base import BaseDataLoader
from data_loader.dataset import TestDataset, TrainDataset, EvalDataset


class InputDataloader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, patch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = TrainDataset(data_dir)

        super().__init__(self.dataset, batch_size, patch_size, shuffle, validation_split, num_workers)


class TestDataloader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, patch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = TestDataset(data_dir)

        super().__init__(self.dataset, batch_size, patch_size, shuffle, validation_split, num_workers)


class EvalDataloader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, patch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = EvalDataset(data_dir)

        super().__init__(self.dataset, batch_size, patch_size, shuffle, validation_split, num_workers)
