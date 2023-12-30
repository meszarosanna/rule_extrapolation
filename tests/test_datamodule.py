from llm_non_identifiability.datamodule import GrammarDataModule


def test_generate_data_correctly(max_length):
    data_module = GrammarDataModule(
        n_train=100, n_val=50, n_test=50, max_length=max_length
    )
    data_module.prepare_data()

    assert len(data_module.train_dataset) == 100
    assert len(data_module.val_dataset) == 50
    assert len(data_module.test_dataset) == 50

    assert data_module.train_dataset.data.shape[1] == max_length
    assert data_module.val_dataset.data.shape[1] == max_length
    assert data_module.test_dataset.data.shape[1] == max_length
