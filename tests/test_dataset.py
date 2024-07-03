from rule_extrapolation.dataset import GrammarDataset
import torch


def test_data_is_long():
    data = GrammarDataset([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    assert data.data.dtype == torch.long
