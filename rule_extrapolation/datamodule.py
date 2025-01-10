"""pytorch lightning datamodule for the LLM non-identifiability experiment."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os

from rule_extrapolation.data import (
    generate_aNbN_grammar_data,
    generate_abN_grammar_data,
    generate_baN_grammar_data,
    generate_bbaN_grammar_data,
    generate_aNbM_grammar_data,
    generate_aNbNaN_grammar_data,
    generate_matched_parentheses_and_brackets_data,
    generate_not_nested_matched_parentheses_and_brackets_data,
    generate_matched_parentheses_and_matched_brackets_data,
    generate_matched_brackets_data,
    generate_matched_parentheses_data,
    generate_aNbNcN_grammar_data,
)
from rule_extrapolation.dataset import GrammarDataset


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
        all_sequences: bool = True,
        batch_size: int = 64,
        grammar: str = "aNbN",
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
        elif self.hparams.grammar == "baN":
            return generate_baN_grammar_data
        elif self.hparams.grammar == "bbaN":
            return generate_bbaN_grammar_data
        elif self.hparams.grammar == "aNbM":
            return generate_aNbM_grammar_data
        elif self.hparams.grammar == "aNbNaN":
            return generate_aNbNaN_grammar_data
        elif self.hparams.grammar == "brackets":
            return generate_matched_brackets_data
        elif self.hparams.grammar == "parentheses":
            return generate_matched_parentheses_data
        elif self.hparams.grammar == "parentheses_and_brackets":
            return generate_matched_parentheses_and_brackets_data
        elif self.hparams.grammar == "separated_parentheses_and_brackets":
            return generate_matched_parentheses_and_matched_brackets_data
        elif self.hparams.grammar == "not_nested_parentheses_and_brackets":
            return generate_not_nested_matched_parentheses_and_brackets_data
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
            # self.hparams.num_val = self.hparams.num_test = self.hparams.num_train = self.hparams.max_length
            train_data = grammar_generator(
                self.hparams.num_train,
                self.hparams.max_length,
                self.hparams.all_sequences,
            )
            val_data = grammar_generator(
                self.hparams.num_val,
                self.hparams.max_length,
                self.hparams.all_sequences,
            )
            test_data = grammar_generator(
                self.hparams.num_test,
                self.hparams.max_length,
                self.hparams.all_sequences,
            )
        else:
            train_data = grammar_generator(
                self.hparams.num_train, self.hparams.max_length
            )
            val_data = grammar_generator(self.hparams.num_val, self.hparams.max_length)
            test_data = grammar_generator(
                self.hparams.num_test, self.hparams.max_length
            )

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

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count(), self.hparams.max_num_workers),
            persistent_workers=True,
        )

        self.val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), self.hparams.max_num_workers),
            persistent_workers=True,
        )

        self.test_dl = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), self.hparams.max_num_workers),
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    def predict_dataloader(self):
        return self.test_dl
