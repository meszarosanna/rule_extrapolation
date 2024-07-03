import pytest

from rule_extrapolation.datamodule import GrammarDataModule


@pytest.mark.parametrize(
    "grammar", ["aNbN", "abN", "baN", "bbaN", "aNbM", "aNbNaN", "aNbNcN"]
)
def test_generate_data_correctly(num_train, num_val, num_test, max_length, grammar):
    data_module = GrammarDataModule(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        max_length=max_length,
        grammar=grammar,
    )
    data_module.prepare_data()

    if grammar in "aNbN":
        num_train = num_val = num_test = max_length // 2
    elif grammar == "aNbNaN":
        num_train = num_val = num_test = max_length // 3
    elif grammar == "aNbNcN":
        num_train = num_val = num_test = max_length // 3

    assert len(data_module.train_dataset) == num_train
    assert len(data_module.val_dataset) == num_val
    assert len(data_module.test_dataset) == num_test

    assert (
        data_module.train_dataset.data.shape[1] == max_length + 2
    )  # +2 for SOS and EOS tokens
    assert (
        data_module.val_dataset.data.shape[1] == max_length + 2
    )  # +2 for SOS and EOS tokens
    assert (
        data_module.test_dataset.data.shape[1] == max_length + 2
    )  # +2 for SOS and EOS tokens
