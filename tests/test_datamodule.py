from llm_non_identifiability.datamodule import GrammarDataModule

from llm_non_identifiability.data import (
    generate_aNbN_grammar_data,
    generate_abN_grammar_data,
    generate_aNbM_grammar_data,
    pad,
)

import pytest

import torch


@pytest.mark.parametrize("grammar", ["aNbN", "abN", "aNbM"])
def test_generate_data_correctly(num_train, num_val, num_test, max_length, grammar):
    data_module = GrammarDataModule(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        max_length=max_length,
        grammar=grammar,
    )
    data_module.prepare_data()

    assert len(data_module.train_dataset) == num_train
    assert len(data_module.val_dataset) == num_val
    assert len(data_module.test_dataset) == num_test

    assert data_module.train_dataset.data.shape[1] == max_length
    assert data_module.val_dataset.data.shape[1] == max_length
    assert data_module.test_dataset.data.shape[1] == max_length


@pytest.mark.parametrize("grammar", ["aNbN", "abN", "aNbM"])
def test_grammar_rules(num_train, num_val, num_test, max_length, grammar, num_samples):
    data_module = GrammarDataModule(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        max_length=max_length,
        grammar=grammar,
    )
    data_module.prepare_data()

    aNbN_data = torch.from_numpy(
        pad(generate_aNbN_grammar_data(num_samples=num_samples, max_length=max_length))
    ).long()
    abN_data = torch.from_numpy(
        pad(generate_abN_grammar_data(num_samples=num_samples, max_length=max_length))
    ).long()
    aNbM_data = torch.from_numpy(
        pad(generate_aNbM_grammar_data(num_samples=num_samples, max_length=max_length))
    ).long()

    if grammar == "aNbN":
        assert torch.all(
            torch.tensor([data_module.grammar_rules(d) for d in aNbN_data])
        )
    elif grammar == "abN":
        assert torch.all(torch.tensor([data_module.grammar_rules(d) for d in abN_data]))
    elif grammar == "aNbM":
        assert torch.all(
            torch.tensor([data_module.grammar_rules(d) for d in aNbM_data])
        )
