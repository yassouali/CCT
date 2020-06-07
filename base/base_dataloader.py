import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split = 0.0):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        if val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else:
            self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        
        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)
        
        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)
