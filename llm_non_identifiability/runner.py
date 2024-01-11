# Importing necessary libraries
import subprocess
from os.path import dirname
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

import llm_non_identifiability.data
from llm_non_identifiability.model import Transformer


class LightningGrammarModule(pl.LightningModule):
    """
    LightningModule for training a Transformer on sequence data coming from a PCFG grammar.
    """

    def __init__(
        self,
        num_tokens: int = 5,  # SOS, EOS, 0, 1, PAD
        dim_model: int = 8,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout_p: float = 0.1,
        lr: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """

        :param lr: learning rate
        :param device:
        """
        super().__init__()
        self.save_hyperparameters()

        self.hparams["loss_fn"] = nn.CrossEntropyLoss()
        self.model = Transformer(
            num_tokens=self.hparams.num_tokens,  # type: ignore [union-attr,attr-defined]
            dim_model=self.hparams.dim_model,  # type: ignore [union-attr,attr-defined]
            num_heads=self.hparams.num_heads,  # type: ignore [union-attr,attr-defined]
            num_encoder_layers=self.hparams.num_encoder_layers,  # type: ignore [union-attr,attr-defined]
            num_decoder_layers=self.hparams.num_decoder_layers,  # type: ignore [union-attr,attr-defined]
            dropout_p=self.hparams.dropout_p,  # type: ignore [union-attr,attr-defined]
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        panel_name = "Train"
        _, _, _, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        panel_name = "Val"
        X, y, pred, loss = self._forward(batch)

        self.log(f"{panel_name}/loss", loss)

        return loss

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
            self.model.create_pad_mask(X, 4),
            self.model.create_pad_mask(y_input, 4),
        )

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)

        loss = self.hparams.loss_fn(pred, y_expected)

        return X, y, pred, loss

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
                [[llm_non_identifiability.data.SOS_token.item(), 0, 0, 0, 1]],
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

            # pick the highest probability token
            next_item = self._pick_most_likely_token(pred)

            # Concatenate previous input with predicted best word
            prompt = torch.cat((prompt, next_item), dim=1)

            # Stop if model predicts end of sentence
            if (
                next_item.view(-1).item()
                == llm_non_identifiability.data.EOS_token.item()
            ):
                break
        return prompt.view(-1).tolist()

    def _pick_most_likely_token(self, pred: torch.Tensor) -> torch.Tensor:
        _, next_item = torch.max(pred[-1].view(-1), dim=-1)
        next_item = torch.tensor([[next_item]], device=self.hparams.device)  # type: ignore
        return next_item

    def on_fit_end(self) -> None:
        self._sync_wandb()

    def _sync_wandb(self):
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            logger: pl.loggers.wandb.WandbLogger = self.logger  # type: ignore
            if self.hparams.offline is True:  # type: ignore [union-attr,attr-defined]
                # Syncing W&B at the end
                # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
                sync_dir = dirname(logger.experiment.dir)
                # 2. mark run complete
                wandb.finish()  # type: ignore
                # 3. call the sync command for the run directory
                subprocess.check_call(["wandb", "sync", sync_dir])
