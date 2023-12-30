"""python dataset from the data genearted by data.py"""

import torch
from torch.utils.data import Dataset

from llm_non_identifiability.data import pad


class GrammarDataset(Dataset):
    def __init__(self, data):
        # pad the data
        self.data = torch.from_numpy(pad(data)).long()
        self.labels = self.data.clone()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
