import pytest
import torch
from pytorch_lightning.trainer import Trainer

from rule_extrapolation.data import SOS_token
from rule_extrapolation.datamodule import GrammarDataModule
from rule_extrapolation.runner import LightningGrammarModule


@pytest.mark.parametrize("adversarial_training", [True, False])
def test_fit_adversarial(num_train, num_val, num_test, adversarial_training):
    trainer = Trainer(fast_dev_run=True)
    grammar = "aNbN"
    runner = LightningGrammarModule(
        grammar=grammar, adversarial_training=adversarial_training
    )
    dm = GrammarDataModule(
        num_train=num_train, num_val=num_val, num_test=num_test, grammar=grammar
    )
    trainer.fit(runner, datamodule=dm)


@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_fit_optimizer(num_train, num_val, num_test, optimizer):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule(optimizer=optimizer)
    dm = GrammarDataModule(num_train=num_train, num_val=num_val, num_test=num_test)
    trainer.fit(runner, datamodule=dm)


@pytest.mark.parametrize("model", ["transformer", "linear", "lstm", "mamba", "xlstm"])
def test_fit_model(num_train, num_val, num_test, model, max_length):
    if model == "xlstm" and torch.cuda.is_available() is False:
        pytest.skip("xLSTM requires a GPU")
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule(max_data_length=max_length, model=model)
    dm = GrammarDataModule(
        num_train=num_train, num_val=num_val, num_test=num_test, max_length=max_length
    )
    trainer.fit(runner, datamodule=dm)


@pytest.mark.parametrize("model", ["transformer", "linear", "lstm", "mamba", "xlstm"])
def test_fit_sampling(num_train, num_val, num_test, model, max_length):
    if model == "xlstm" and torch.cuda.is_available() is False:
        pytest.skip("xLSTM requires a GPU")
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule(
        max_data_length=max_length, model=model, next_token_pick_mode="sample"
    )
    dm = GrammarDataModule(
        num_train=num_train, num_val=num_val, num_test=num_test, max_length=max_length
    )
    trainer.fit(runner, datamodule=dm)


@pytest.mark.parametrize(
    "grammar",
    [
        "baN",
        "bbaN",
        "aNbN",
        "aNbNcN",
        "parentheses_and_brackets",
        "not_nested_parentheses_and_brackets",
    ],
)
def test_fit_grammars(num_train, num_val, num_test, max_length, grammar):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule(grammar=grammar, max_data_length=max_length)
    dm = GrammarDataModule(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        max_length=max_length,
        grammar=grammar,
    )
    trainer.fit(runner, datamodule=dm)


def test_fit_extrapolation(
    num_train,
    num_val,
    num_test,
):
    trainer = Trainer(fast_dev_run=True)
    grammar = "aNbN"
    runner = LightningGrammarModule(grammar=grammar, extrapolation_training=True)
    dm = GrammarDataModule(
        num_train=num_train, num_val=num_val, num_test=num_test, grammar=grammar
    )
    trainer.fit(runner, datamodule=dm)


@pytest.mark.parametrize("next_token_pick_mode", ["sample", "max"])
def test_predict_inner(max_length, device, next_token_pick_mode):
    runner = LightningGrammarModule(next_token_pick_mode=next_token_pick_mode)

    # Here we test some examples to observe how the model predicts
    examples = [
        torch.tensor(
            [
                [
                    SOS_token.item(),
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [
                [SOS_token.item(), 0, 0, 0, 1, 1, 1],
                [SOS_token.item(), 0, 0, 1, 0, 1, 1],
            ],
            dtype=torch.long,
            device=device,
        ),
    ]

    for example in examples:
        runner._predict(prompt=example, max_length=max_length)
