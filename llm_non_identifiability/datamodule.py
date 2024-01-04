"""pytorch lightning datamodule for the LLM non-identifiability experiment."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from llm_non_identifiability.data import (
    generate_aNbN_grammar_data,
)
from llm_non_identifiability.dataset import GrammarDataset


class GrammarDataModule(pl.LightningDataModule):
    def __init__(
        self, n_train=9000, n_val=3000, n_test=1024, max_length=32, batch_size=64
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        """
        This method is called only once to prepare the data.
        """
        train_data = generate_aNbN_grammar_data(
            self.hparams.n_train, self.hparams.max_length
        )
        val_data = generate_aNbN_grammar_data(
            self.hparams.n_val, self.hparams.max_length
        )
        test_data = generate_aNbN_grammar_data(
            self.hparams.n_test, self.hparams.max_length
        )

        self.test_dataset = GrammarDataset(test_data)
        self.val_dataset = GrammarDataset(val_data)
        self.train_dataset = GrammarDataset(train_data)

    def setup(self, stage=None):
        """
        This method is called once per GPU.
        """
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )
