# Importing necessary libraries
import subprocess
from os.path import dirname
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from llm_non_identifiability.data import (
    check_same_number_as_bs,
    check_as_before_bs,
    EOS_token,
    PAD_token,
    check_sequence_finished,
    generate_test_prompts,
)
from llm_non_identifiability.model import Transformer


class LightningGrammarModule(pl.LightningModule):
    """
    LightningModule for training a Transformer on sequence data coming from a PCFG grammar.
    """

    def __init__(
        self,
        num_tokens: int = 4,
        dim_model: int = 8,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        max_pred_length: int = 64,
        test_prompt_length: int = 6,
        dropout_p: float = 0.1,
        lr: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offline: bool = False,
    ):
        """

        :param test_prompt_length:
        :param max_pred_length:
        :param offline:
        :param lr: learning rate
        :param device:
        """
        super().__init__()
        self.save_hyperparameters()

        self.hparams["loss_fn"] = nn.CrossEntropyLoss()
        self.model = Transformer(
            num_tokens=self.hparams.num_tokens,
            dim_model=self.hparams.dim_model,
            num_heads=self.hparams.num_heads,
            num_encoder_layers=self.hparams.num_encoder_layers,
            num_decoder_layers=self.hparams.num_decoder_layers,
            dropout_p=self.hparams.dropout_p,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def on_fit_start(self) -> None:
        test_prompts = generate_test_prompts(length=self.hparams.test_prompt_length).to(
            self.hparams.device
        )

        rules = self.trainer.datamodule.grammar_rules

        rules_met = [rules(t).item() for t in test_prompts]

        self.test_prompts_in_distribution = test_prompts[rules_met]
        self.test_prompts_out_of_distribution = test_prompts[[not r for r in rules_met]]

        self.hparams.test_prompts_in_distribution_len = len(
            self.test_prompts_in_distribution
        )
        self.hparams.test_prompts_ood_len = len(self.test_prompts_out_of_distribution)

        assert (
            len(test_prompts)
            == self.hparams.test_prompts_in_distribution_len
            + self.hparams.test_prompts_ood_len
        )

    def training_step(self, batch, batch_idx):
        panel_name = "Train"
        _, _, _, _, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        panel_name = "Val"
        X, y, y_expected, pred, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        # pick most likely token and calculate and log accuracy
        pred_tokens = self._pick_most_likely_tokens(pred)
        accuracy = torch.sum(pred_tokens == y_expected) / y_expected.numel()
        self.log(f"{panel_name}/accuracy", accuracy)

        (
            as_before_bs_accuracy,
            same_number_as_bs_accuracy,
            finished_accuracy,
            ood_as_before_bs_accuracy,
            ood_same_number_as_bs_accuracy,
            ood_finished_accuracy,
        ) = self._eval_prompt_prediction()
        self.log(f"{panel_name}/as_before_bs_accuracy", as_before_bs_accuracy)
        self.log(f"{panel_name}/same_number_as_bs_accuracy", same_number_as_bs_accuracy)
        self.log(f"{panel_name}/finished_accuracy", finished_accuracy)

        self.log(f"{panel_name}/ood_as_before_bs_accuracy", ood_as_before_bs_accuracy)
        self.log(
            f"{panel_name}/ood_same_number_as_bs_accuracy",
            ood_same_number_as_bs_accuracy,
        )
        self.log(f"{panel_name}/ood_finished_accuracy", ood_finished_accuracy)

        return loss

    @property
    def test_prompts_src(self):
        ds = self.trainer.datamodule.test_dataset.data.view(-1)
        return ds[ds != PAD_token.item()].long().to(self.hparams.device)

    def _eval_prompt_prediction(self, max_length: Optional[int] = None):
        if max_length is None:
            max_length = self.hparams.max_pred_length

        (
            as_before_bs_accuracy,
            finished_accuracy,
            same_number_as_bs_accuracy,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_in_distribution, max_length
        )

        (
            ood_as_before_bs_accuracy,
            ood_finished_accuracy,
            ood_same_number_as_bs_accuracy,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_out_of_distribution, max_length
        )

        return (
            as_before_bs_accuracy,
            same_number_as_bs_accuracy,
            finished_accuracy,
            ood_as_before_bs_accuracy,
            ood_same_number_as_bs_accuracy,
            ood_finished_accuracy,
        )

    def _calc_prompt_pred_metrics(self, prompts, max_length):
        as_before_bs = []
        same_number_as_bs = []
        finished = []
        for idx, prompt in enumerate(prompts):
            prompt_pred = self._predict(
                max_length=max_length,
                src=self.test_prompts_src.unsqueeze(0),
                prompt=prompt.unsqueeze(0),
            )
            prompt_pred = torch.tensor(
                prompt_pred, device=self.hparams.device, dtype=torch.long
            )
            as_before_bs.append(check_as_before_bs(prompt_pred))
            same_number_as_bs.append(check_same_number_as_bs(prompt_pred))

            finished.append(check_sequence_finished(prompt_pred))
        as_before_bs_accuracy = sum(as_before_bs) / len(as_before_bs)
        same_number_as_bs_accuracy = sum(same_number_as_bs) / len(same_number_as_bs)
        finished_accuracy = sum(finished) / len(finished)
        return as_before_bs_accuracy, finished_accuracy, same_number_as_bs_accuracy

    def _forward(self, batch):
        """
        Forward pass for calculating the model predictions and the loss.
        :param batch:
        :return:
        """
        X, y = batch

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = self.model.get_tgt_mask(sequence_length).to(self.hparams.device)

        # Standard training except we pass in y_input and tgt_mask
        pred = self.model(
            X,
            y_input,
            tgt_mask,
            self.model.create_pad_mask(X, 4).to(self.hparams.device),
            self.model.create_pad_mask(y_input, 4).to(self.hparams.device),
        )

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)

        loss = self.hparams.loss_fn(pred, y_expected)

        return X, y, y_expected, pred, loss

    def predict_step(  # type: ignore
        self,
        batch,
        batch_idx=None,
        prompt: Optional[torch.Tensor] = None,
        max_length: int = 32,
    ):
        """
        Overriding the default method of the LightningModule for being called with Trainer.predict().
        :param batch:
        :param batch_idx:
        :param prompt: optional prompt to start the prediction
        :param max_length: maximum sequence length for the prediction
        :return:
        """
        X, y = batch

        return self._predict(X[0].view(1, -1), max_length, prompt)

    def _predict(
        self,
        src: torch.Tensor,
        max_length: int = 32,
        prompt: Optional[torch.Tensor] = None,
    ):
        """
        Inner method for predicting a sequence.
        :param src: tensor of sequence(s) to "condition" the prediction on
        :param max_length: maximum sequence length for the prediction
        :param prompt: optional prompt to start the prediction
        :return:
        """
        if prompt is None:
            prompt = torch.tensor(
                [[0, 0, 0, 1]],
                dtype=torch.long,
                device=self.hparams.device,  # type: ignore
            )
        for _ in range(max_length):
            # Get mask to mask out the next words
            sequence_length = prompt.size(1)
            tgt_mask = self.model.get_tgt_mask(sequence_length).to(self.hparams.device)  # type: ignore

            # forward pass
            pred = self.model(
                src,
                prompt,
                tgt_mask,
                self.model.create_pad_mask(src, 4),
                self.model.create_pad_mask(prompt, 4),
            )

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)

            _, next_item = torch.max(pred[0, :, -1].view(-1), dim=-1)
            next_item = torch.tensor([[next_item]], device=self.hparams.device)  # type: ignore

            # Concatenate previous input with predicted best word
            prompt = torch.cat((prompt, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token.item():
                break
        return prompt.view(-1).tolist()

    def _pick_most_likely_tokens(self, pred: torch.Tensor) -> torch.Tensor:
        _, next_items = torch.max(pred, dim=1)
        return next_items.to(self.hparams.device)  # type: ignore

    def on_fit_end(self) -> None:
        self._sync_wandb()

    def _sync_wandb(self):
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            logger: pl.loggers.wandb.WandbLogger = self.logger  # type: ignore
            if self.hparams.offline is True:
                # Syncing W&B at the end
                # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
                sync_dir = dirname(logger.experiment.dir)
                # 2. mark run complete
                wandb.finish()  # type: ignore
                # 3. call the sync command for the run directory
                subprocess.check_call(["wandb", "sync", sync_dir])
