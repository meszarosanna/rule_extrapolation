"""pytorch lightning datamodule for the LLM non-identifiability experiment."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os

from llm_non_identifiability.data import (
    generate_aNbN_grammar_data,
    generate_abN_grammar_data,
    generate_aNbM_grammar_data,
    generate_aNbNaN_grammar_data,
    generate_aNbNcN_grammar_data,
)
from llm_non_identifiability.dataset import GrammarDataset


class GrammarDataModule(pl.LightningDataModule):
    """
    DataModule for sequence data coming from an underlying PCFG grammar.
    """

    def __init__(
        self,
        num_train: int = 9000,
        num_val: int = 3000,
        num_test: int = 1024,
        max_length: int = 32,
        batch_size: int = 64,
        grammar: str = "aNbNcN",
        max_num_workers: int = 4,
    ):
        """

        :param max_num_workers:
        :param num_train:
        :param num_val:
        :param num_test:
        :param max_length:
        :param batch_size:
        """
        super().__init__()
        self.save_hyperparameters()

    def _select_grammar(self):
        """
        Selects the grammar to use.
        """
        if self.hparams.grammar == "aNbN":
            return generate_aNbN_grammar_data
        elif self.hparams.grammar == "abN":
            return generate_abN_grammar_data
        elif self.hparams.grammar == "aNbM":
            return generate_aNbM_grammar_data
        elif self.hparams.grammar == "aNbNaN":
            return generate_aNbNaN_grammar_data
        elif self.hparams.grammar == "aNbNcN":
            return generate_aNbNcN_grammar_data
        else:
            raise ValueError(f"Unknown grammar {self.hparams.grammar}")

    def prepare_data(self):
        """
        This method is called only once to prepare the data.
        """

        grammar_generator = self._select_grammar()

        if self.hparams.grammar in ["aNbN", "aNbNaN", "aNbNcN"]:
            # include all samples only once
            self.hparams.num_train = (
                self.hparams.num_val
            ) = self.hparams.num_test = self.hparams.max_length

        train_data = grammar_generator(self.hparams.num_train, self.hparams.max_length)
        val_data = grammar_generator(self.hparams.num_val, self.hparams.max_length)
        test_data = grammar_generator(self.hparams.num_test, self.hparams.max_length)

        self.test_dataset = GrammarDataset(
            test_data,
            max_length=self.hparams.max_length + 2,  # +2 for SOS and EOS tokens
        )
        self.val_dataset = GrammarDataset(
            val_data,
            max_length=self.hparams.max_length + 2,  # +2 for SOS and EOS tokens
        )
        self.train_dataset = GrammarDataset(
            train_data,
            max_length=self.hparams.max_length + 2,  # +2 for SOS and EOS tokens
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), self.hparams.max_num_workers),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), self.hparams.max_num_workers),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), self.hparams.max_num_workers),
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), self.hparams.max_num_workers),
            persistent_workers=True,
        )
