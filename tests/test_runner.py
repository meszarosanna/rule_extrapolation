import pytest
import torch
from pytorch_lightning.trainer import Trainer

from llm_non_identifiability.data import SOS_token
from llm_non_identifiability.datamodule import GrammarDataModule
from llm_non_identifiability.runner import LightningGrammarModule


def test_fit_and_predict(num_train, num_val, num_test):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule()
    dm = GrammarDataModule(num_train=num_train, num_val=num_val, num_test=num_test)
    trainer.fit(runner, datamodule=dm)

    trainer.predict(runner, datamodule=dm)


@pytest.mark.parametrize("next_token_pick_mode", ["sample", "max"])
def test_predict_inner(max_length, device, next_token_pick_mode):
    runner = LightningGrammarModule(next_token_pick_mode=next_token_pick_mode)

    # Here we test some examples to observe how the model predicts
    examples = [
        # torch.tensor(
        #     [[SOS_token.item(), 0, 0, 0, 0, 1, 1, 1, 1, ]],
        #     dtype=torch.long,
        #     device=device,
        # ),
        torch.tensor(
            [
                [SOS_token.item(), 0, 0, 0, 1, 1, 1],
                [SOS_token.item(), 0, 0, 1, 0, 1, 1],
            ],
            dtype=torch.long,
            device=device,
        ),
        # torch.tensor(
        #     [[SOS_token.item(), 0, 1, ]],
        #     dtype=torch.long,
        #     device=device,
        # ),
    ]

    for example in examples:
        runner._predict(prompt=example, max_length=max_length)
