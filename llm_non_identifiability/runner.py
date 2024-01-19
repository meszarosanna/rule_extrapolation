# Importing necessary libraries
import subprocess
from os.path import dirname
from typing import Optional, Dict, Any
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn

from llm_non_identifiability.data import (
    check_same_number_as_bs,
    check_as_before_bs,
    SOS_token,
    EOS_token,
    PAD_token,
    check_sequence_finished,
    generate_test_prompts,
)
from llm_non_identifiability.model import (
    TransformerDecoder,
    create_pad_mask,
    get_tgt_mask,
)


class LightningGrammarModule(pl.LightningModule):
    """
    LightningModule for training a Transformer on sequence data coming from a PCFG grammar.
    """

    def __init__(
        self,
        num_tokens: int = 5,
        dim_model: int = 8,
        dim_feedforward: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        max_pred_length: int = 64,
        test_prompt_length: int = 6,
        dropout_p: float = 0.1,
        lr: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offline: bool = False,
        next_token_pick_mode: str = "max",
        layer_norm_eps=2e-4,
    ):
        """

        :param layer_norm_eps:
        :param next_token_pick_mode:
        :param dim_feedforward:
        :param test_prompt_length:
        :param max_pred_length:
        :param offline:
        :param lr: learning rate
        :param device:
        """
        super().__init__()
        self.save_hyperparameters()

        self.hparams["loss_fn"] = nn.CrossEntropyLoss()
        self.model = TransformerDecoder(
            num_tokens=self.hparams.num_tokens,
            dim_model=self.hparams.dim_model,
            num_heads=self.hparams.num_heads,
            num_decoder_layers=self.hparams.num_decoder_layers,
            dropout_p=self.hparams.dropout_p,
            dim_feedforward=self.hparams.dim_feedforward,
            layer_norm_eps=self.hparams.layer_norm_eps,
        )

    @property
    def data_entropy(self):
        return math.log(self.trainer.datamodule.hparams.max_length // 2, math.e)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def on_fit_start(self) -> None:
        test_prompts = generate_test_prompts(length=self.hparams.test_prompt_length).to(
            self.hparams.device
        )

        rules = self.trainer.datamodule.prompt_grammar_rules

        rules_met = [rules(t) for t in test_prompts]

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

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            # log entropy of the test prompts = entropy of the distribution of the prompt lengths
            # log as a summary item
            self.logger.experiment.summary["data_entropy"] = self.data_entropy

    def training_step(self, batch, batch_idx):
        panel_name = "Train"
        _, _, _, _, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        panel_name = "Val"
        X, y, y_expected, pred, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        self.log(f"{panel_name}/kl", loss - self.data_entropy)

        # pick most likely token and calculate and log accuracy
        pred_tokens = self._pick_next_tokens(pred)
        accuracy = torch.sum(pred_tokens == y_expected) / y_expected.numel()
        self.log(f"{panel_name}/accuracy", accuracy)

        (
            prompts,
            as_before_bs_accuracy,
            same_number_as_bs_accuracy,
            finished_accuracy,
            grammatical_accuracy,
            ood_prompts,
            ood_as_before_bs_accuracy,
            ood_same_number_as_bs_accuracy,
            ood_finished_accuracy,
            ood_grammatical_accuracy,
            sos_prompts,
            sos_as_before_bs_accuracy,
            sos_same_number_as_bs_accuracy,
            sos_finished_accuracy,
            sos_grammatical_accuracy,
        ) = self._eval_prompt_prediction()
        self.log(f"{panel_name}/as_before_bs_accuracy", as_before_bs_accuracy)
        self.log(f"{panel_name}/same_number_as_bs_accuracy", same_number_as_bs_accuracy)
        self.log(f"{panel_name}/finished_accuracy", finished_accuracy)
        self.log(f"{panel_name}/grammatical_accuracy", grammatical_accuracy)

        self.log(f"{panel_name}/ood_as_before_bs_accuracy", ood_as_before_bs_accuracy)
        self.log(
            f"{panel_name}/ood_same_number_as_bs_accuracy",
            ood_same_number_as_bs_accuracy,
        )
        self.log(f"{panel_name}/ood_finished_accuracy", ood_finished_accuracy)
        self.log(f"{panel_name}/ood_grammatical_accuracy", ood_grammatical_accuracy)

        self.log(f"{panel_name}/sos_as_before_bs_accuracy", sos_as_before_bs_accuracy)
        self.log(
            f"{panel_name}/sos_same_number_as_bs_accuracy",
            sos_same_number_as_bs_accuracy,
        )
        self.log(f"{panel_name}/sos_finished_accuracy", sos_finished_accuracy)
        self.log(f"{panel_name}/sos_grammatical_accuracy", sos_grammatical_accuracy)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        (
            prompts,
            as_before_bs_accuracy,
            same_number_as_bs_accuracy,
            finished_accuracy,
            grammatical_accuracy,
            ood_prompts,
            ood_as_before_bs_accuracy,
            ood_same_number_as_bs_accuracy,
            ood_finished_accuracy,
            ood_grammatical_accuracy,
            sos_prompts,
            sos_as_before_bs_accuracy,
            sos_same_number_as_bs_accuracy,
            sos_finished_accuracy,
            sos_grammatical_accuracy,
        ) = self._eval_prompt_prediction()

        checkpoint["prompts"] = prompts.cpu().numpy()
        checkpoint["ood_prompts"] = ood_prompts.cpu().numpy()
        checkpoint["sos_prompts"] = sos_prompts.cpu().numpy()

    @property
    def test_prompts_src(self):
        ds = self.trainer.datamodule.test_dataset.data.view(-1)
        return ds[ds != PAD_token.item()].long().to(self.hparams.device)

    def _eval_prompt_prediction(self, max_length: Optional[int] = None):
        if max_length is None:
            max_length = self.hparams.max_pred_length

        (
            prompts,
            as_before_bs_accuracy,
            finished_accuracy,
            same_number_as_bs_accuracy,
            grammatical_accuracy,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_in_distribution, max_length
        )

        (
            ood_prompts,
            ood_as_before_bs_accuracy,
            ood_finished_accuracy,
            ood_same_number_as_bs_accuracy,
            ood_grammatical_accuracy,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_out_of_distribution, max_length
        )

        # prompt prediction for a batch of SOS tokens
        sos_prompts = (
            torch.ones(
                (self.trainer.datamodule.hparams.batch_size, 1),
                dtype=torch.long,
                device=self.hparams.device,
            )
            * SOS_token.item()
        )
        (
            sos_prompts,
            sos_as_before_bs_accuracy,
            sos_finished_accuracy,
            sos_same_number_as_bs_accuracy,
            sos_grammatical_accuracy,
        ) = self._calc_prompt_pred_metrics(sos_prompts, max_length)

        return (
            prompts,
            as_before_bs_accuracy,
            same_number_as_bs_accuracy,
            finished_accuracy,
            grammatical_accuracy,
            ood_prompts,
            ood_as_before_bs_accuracy,
            ood_same_number_as_bs_accuracy,
            ood_finished_accuracy,
            ood_grammatical_accuracy,
            sos_prompts,
            sos_as_before_bs_accuracy,
            sos_same_number_as_bs_accuracy,
            sos_finished_accuracy,
            sos_grammatical_accuracy,
        )

    def _calc_prompt_pred_metrics(self, prompts, max_length):
        prompt_pred = self._predict(
            max_length=max_length,
            src=self.test_prompts_src,
            prompt=prompts,
        )

        as_before_bs = [check_as_before_bs(p) for p in prompt_pred]
        same_number_as_bs = [check_same_number_as_bs(p) for p in prompt_pred]
        grammatical = [self.trainer.datamodule.grammar_rules(p) for p in prompt_pred]
        finished = [check_sequence_finished(p) for p in prompt_pred]

        as_before_bs_accuracy = sum(as_before_bs) / len(as_before_bs)
        same_number_as_bs_accuracy = sum(same_number_as_bs) / len(same_number_as_bs)
        finished_accuracy = sum(finished) / len(finished)
        grammatical_accuracy = sum(grammatical) / len(grammatical)
        return (
            prompts,
            as_before_bs_accuracy,
            finished_accuracy,
            same_number_as_bs_accuracy,
            grammatical_accuracy,
        )

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
        causal_mask = get_tgt_mask(y_input.size(1), device=self.hparams.device)

        # Standard training except we pass in y_input and causal_mask
        pred = self.model(
            src=y_input,
            mask=causal_mask,
            src_key_padding_mask=create_pad_mask(y_input),
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

        finished = torch.BoolTensor([False] * prompt.size(0)).to(self.hparams.device)

        for _ in range(max_length):
            # Get mask to mask out the next words
            tgt_mask = get_tgt_mask(size=(prompt.size(1)), device=self.hparams.device)

            # forward pass
            pred = self.model(
                src=prompt,
                mask=tgt_mask,
                src_key_padding_mask=create_pad_mask(prompt),
            )

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)

            # pick the prediction for the last token only
            next_items = self._pick_next_tokens(pred)[:, -1].view(-1, 1)

            # Concatenate previous input with predicted best word
            prompt = torch.cat((prompt, next_items), dim=1)

            # save if model predicts end of sentence
            finished.logical_or_(next_items.view(-1) == EOS_token.item())
            # Stop if model predicts end of sentence
            if torch.all(finished) is True:
                break
        return prompt.long().to(self.hparams.device)

    def _pick_next_tokens(self, pred: torch.Tensor) -> torch.Tensor:
        if self.hparams.next_token_pick_mode == "max":
            _, next_items = torch.max(pred, dim=1)
        elif self.hparams.next_token_pick_mode == "sample":
            next_items = torch.multinomial(
                torch.softmax(pred.squeeze(), dim=1).T, num_samples=1
            ).T
        else:
            raise ValueError(
                f"Unknown next_token_pick_mode: {self.hparams.next_token_pick_mode}, should be 'max' or 'sample'"
            )
        return next_items.to(self.hparams.device)

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
