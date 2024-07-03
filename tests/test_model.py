from rule_extrapolation.model import get_tgt_mask, TransformerDecoder
import torch


def test_get_tgt_mask():
    size = 5
    mask = get_tgt_mask(size, device="cpu")

    assert mask.shape == (size, size)
    assert torch.all(torch.tril(mask) == 0) == True
    assert torch.all((mask == float("-inf")).sum(0) == torch.arange(size)) == True


def test_relu_rescale():
    model = TransformerDecoder(relu_rescale=1.0)

    assert model.relu_rescale.requires_grad is False
