import math
import subprocess
from itertools import product
from os.path import dirname
from random import choices
from typing import Optional, Dict, Any


import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf
from transformers.optimization import get_inverse_sqrt_schedule
from xlstm import xLSTMLMModel, xLSTMLMModelConfig
from xlstm.blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig

from mamba.mamba_lm import MambaLM, MambaLMConfig
from rule_extrapolation.data import (
    check_parity,
    check_same_number_as_bs,
    check_as_before_bs,
    check_bs_before_as,
    check_same_number_as_bs_cs,
    check_as_before_bs_before_cs,
    check_even_number_of_as,
    check_even_number_of_as_end,
    check_begins_with_opening_parenthesis,
    check_matched_parentheses,
    check_begins_with_b,
    check_matched_brackets,
    check_matched_parentheses_and_brackets,
    SOS_token,
    EOS_token,
    PAD_token,
    A_token,
    B_token,
    check_sequence_finished,
    generate_test_prompts,
    grammar_rules,
    prompt_grammar_rules,
    GrammarMetrics,
    pad,
)
from rule_extrapolation.model import (
    TransformerDecoder,
    create_pad_mask,
    get_tgt_mask,
    LinearLLM,
    LSTM_LLM,
)


class LightningGrammarModule(pl.LightningModule):
    """
    LightningModule for training a Transformer on sequence data coming from a PCFG grammar.
    """

    def __init__(
        self,
        num_tokens: int = 10,
        dim_model: int = 8,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        num_decoder_layers: int = 2,
        max_pred_length: int = 64,
        test_prompt_length: int = 6,
        dropout_p: float = 0.1,
        lr: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offline: bool = False,
        next_token_pick_mode: str = "max",
        layer_norm_eps: float = 2e-4,
        grammar: str = "aNbN",
        max_data_length: int = 256,
        batch_size: int = 64,
        relu_rescale: float = 1.0,
        adversarial_training: bool = False,
        num_warmup_steps: int = 1000,
        extrapolation_training: bool = False,
        optimizer: str = "adamw",
        dim_feedforward: int = 256,
        hidden_dim: int = 128,
        model="transformer",
        bias=True,
        dropout=0.4,
        plot1: bool = False,
        plot2: bool = False,
        n_layers=4,
        d_state=16,
        d_conv=4,
        d_model=8,
        num_blocks=7,
        xlstm_embedding_dim=128,
        slstm_at=[1],
    ):
        """
        :param optimizer:
        :param extrapolation_training:
        :param num_warmup_steps:
        :param relu_rescale:
        :param adversarial_training:
        :param batch_size:
        :param max_data_length:
        :param grammar:
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

        if (
            self.hparams.extrapolation_training is True
            and self.hparams.adversarial_training is True
        ):
            raise ValueError(
                "Cannot train with both extrapolation and adversarial training"
            )

        self.hparams["loss_fn"] = (
            nn.CrossEntropyLoss()
            if self.hparams.model != "linear"
            else nn.CrossEntropyLoss(ignore_index=PAD_token.item())
        )

        # calculate number of tokens:
        if self.hparams.grammar in ["aNbN", "abN", "aNbM", "aNbNaN", "baN", "bbaN"]:
            self.hparams["num_tokens"] = 5
        elif self.hparams.grammar == "aNbNcN":
            self.hparams["num_tokens"] = 6
        elif self.hparams.grammar == "parentheses":
            self.hparams["num_tokens"] = 5
        elif self.hparams.grammar in [
            "parentheses_and_brackets",
            "not_nested_parentheses_and_brackets",
            "separated_parentheses_and_brackets",
        ]:
            self.hparams["num_tokens"] = 7
        elif grammar == "brackets":
            raise NotImplementedError("num_tokens for brackets grammar is inconsistent")

        self._setup_model()

        # access grammar rule (e.g. check_as_before_bs)
        self.grammar_rules = grammar_rules(self.hparams.grammar)
        self.prompt_grammar_rules = prompt_grammar_rules(self.hparams.grammar)
        self._setup_test_prompts()

        self.result1 = 0
        self.result2 = 0
        self.result3 = 0

        self.rule12 = []  # type: list[float]
        self.rule1 = []  # type: list[float]
        self.rule2 = []  # type: list[float]
        self.rule_ = []  # type: list[float]

    def _setup_model(self):
        if self.hparams.model == "transformer":
            self.model: nn.Module = TransformerDecoder(
                num_tokens=self.hparams.num_tokens,
                dim_model=self.hparams.dim_model,
                num_heads=self.hparams.num_heads,
                num_decoder_layers=self.hparams.num_decoder_layers,  # num_layers
                dropout_p=self.hparams.dropout_p,
                dim_feedforward=self.hparams.dim_feedforward,  # hidden_dim
                layer_norm_eps=self.hparams.layer_norm_eps,
                relu_rescale=self.hparams.relu_rescale,
            )
        elif self.hparams.model == "linear":
            self.model: nn.Module = LinearLLM(  # type: ignore
                max_data_length=self.hparams.max_data_length,
                num_tokens=self.hparams.num_tokens,
                bias=self.hparams.bias,
                device=self.hparams.device,
                embedding_dim=self.hparams.dim_model,
            )

        elif self.hparams.model == "lstm":
            self.model: nn.Module = LSTM_LLM(  # type: ignore
                num_tokens=self.hparams.num_tokens,
                embedding_dim=self.hparams.dim_model,
                hidden_dim=self.hparams.hidden_dim,
                num_layers=self.hparams.num_layers,
                dropout_lstm=self.hparams.dropout,
                device=self.hparams.device,
            )
        elif self.hparams.model == "mamba":
            self.model: nn.Module = MambaLM(  # type: ignore
                lm_config=MambaLMConfig(
                    vocab_size=self.hparams.num_tokens,
                    d_model=self.hparams.d_model,
                    d_state=self.hparams.d_state,
                    d_conv=self.hparams.d_conv,
                    n_layers=self.hparams.n_layers,
                )
            )

        elif self.hparams.model == "xlstm":
            xlstm_cfg = f""" 
            vocab_size: {self.hparams.num_tokens}
            mlstm_block:
              mlstm:
                conv1d_kernel_size: 4
                qkv_proj_blocksize: 4
                num_heads: 4
            slstm_block:
              slstm:
                backend: cuda
                num_heads: 4
                conv1d_kernel_size: 4
                bias_init: powerlaw_blockdependent
              feedforward:
                proj_factor: 1.3
                act_fn: gelu
            context_length: {self.hparams.max_data_length + 2}
            num_blocks: {self.hparams.num_blocks}
            embedding_dim: {self.hparams.xlstm_embedding_dim}
            slstm_at: [1]
            """

            cfg = OmegaConf.create(xlstm_cfg)
            cfg = from_dict(
                data_class=xLSTMLMModelConfig,
                data=OmegaConf.to_container(cfg),
                config=DaciteConfig(strict=True),
            )

            self.model: nn.Module = xLSTMLMModel(cfg)

    @property
    def data_entropy(self):
        return math.log(n := (self.hparams.max_data_length // 2), math.e) / n

    def configure_optimizers(self):
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
            return {
                "optimizer": optimizer,
                "lr_scheduler": get_inverse_sqrt_schedule(
                    optimizer=optimizer, num_warmup_steps=self.hparams.num_warmup_steps
                ),
            }
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
            return {"optimizer": optimizer}
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

    def _setup_test_prompts(self) -> None:
        test_prompts = generate_test_prompts(
            grammar=self.hparams.grammar, length=self.hparams.test_prompt_length
        ).to(self.hparams.device)

        if (
            self.hparams.grammar != "parentheses_and_brackets"
            and self.hparams.grammar != "not_nested_parentheses_and_brackets"
            and self.hparams.grammar != "separated_parentheses_and_brackets"
        ):
            rules_met = [self.prompt_grammar_rules(t) for t in test_prompts]
            self.test_prompts_in_distribution = test_prompts[rules_met]
            self.test_prompts_out_of_distribution = test_prompts[
                [not r for r in rules_met]
            ]
        else:
            rules_met = [check_begins_with_opening_parenthesis(t) for t in test_prompts]
            self.test_prompts_in_distribution = test_prompts[rules_met]
            self.test_prompts_out_of_distribution = test_prompts[
                [not r for r in rules_met]
            ]

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

        self.__setup_adversarial_prompts()
        self.__setup_oracle_prompts()

    def __setup_adversarial_prompts(self) -> None:
        """
        Setup the prompts for adversarial training from the OOD test prompts
        """

        if self.hparams.adversarial_training is True and self.hparams.grammar == "aNbN":
            prompts = []

            for idx, prompt in enumerate(self.test_prompts_out_of_distribution):
                num_as = torch.sum(prompt == A_token.item())
                num_bs = torch.sum(prompt == B_token.item())

                if num_as >= num_bs:
                    prompt = self._extend_prompt(
                        prompt, num_as - num_bs + 1, value=B_token.item()
                    )
                else:
                    prompt = self._extend_prompt(
                        prompt, num_bs - num_as + 1, value=A_token.item()
                    )

                prompts.append(prompt.cpu().numpy())

            self.adversarial_prompts = (
                torch.from_numpy(pad(prompts)).long().to(self.hparams.device)
            )

    def __setup_oracle_prompts(self) -> None:
        """
        Setup the prompts for extrapolation training from the OOD test prompts
        """

        if (
            self.hparams.extrapolation_training is True
            and self.hparams.grammar == "aNbN"
        ):
            prompts = []

            for idx, prompt in enumerate(self.test_prompts_out_of_distribution):
                num_as = torch.sum(prompt == A_token.item())
                num_bs = torch.sum(prompt == B_token.item())

                if num_as >= num_bs:
                    prompt = self._extend_prompt(
                        prompt, num_as - num_bs, value=B_token.item()
                    )
                else:
                    prompt = self._extend_prompt(
                        prompt, num_bs - num_as, value=A_token.item()
                    )

                assert check_same_number_as_bs(prompt) == True

                prompts.append(prompt.cpu().numpy())

            self.extrapolation_prompts = (
                torch.from_numpy(pad(prompts)).long().to(self.hparams.device)
            )

    def _extend_prompt(self, prompt, length, value=A_token.item()):
        prompt = torch.cat(
            (
                prompt,
                torch.ones(
                    (length,),
                    dtype=torch.long,
                    device=self.hparams.device,
                )
                * value,
                torch.ones(
                    (1,),
                    dtype=torch.long,
                    device=self.hparams.device,
                )
                * EOS_token.item(),
            ),
            dim=0,
        )
        return prompt

    def on_fit_start(self):
        result = self.plot_figure_1()
        plt.imshow(result, cmap="Greys", vmin=0, vmax=0.01)
        plt.colorbar()
        plt.title("Initialization")
        plt.savefig("fig1_init1.png")

    def training_step(self, batch, batch_idx):
        # plotting
        if self.hparams.plot2 is True:
            if self.trainer.global_step == 0 or (
                self.current_epoch % 100 == 0
                and self.current_epoch <= 12500
                and self.current_epoch > 0
            ):
                # sum the probabilities of each category
                L12, L1, L2, L = self.plot_figure_1(True)
                self.rule12.append(sum(L12))
                self.rule1.append(sum(L1))
                self.rule2.append(sum(L2))
                self.rule_.append(sum(L))

                if self.current_epoch == 12500:
                    x_values = np.arange(0, (len(self.rule12)) * 100, 100)
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.plot(x_values, self.rule12, label="R1 and R2")
                    ax.plot(x_values, self.rule1, label="Only R1")
                    ax.plot(x_values, self.rule2, label="Only R2")
                    ax.plot(x_values, self.rule_, label="Not R1, not R2")
                    ax.set_xlabel("Epochs", fontsize=15)
                    ax.set_ylabel("Sum of probabilities", fontsize=15)
                    ax.set_title(
                        "Training dynamics of the probabilities of the sequences",
                        fontsize=18,
                    )
                    ax.legend()
                    plt.savefig("dynamics_transformer.png")

        if self.hparams.plot1 is True:
            if (
                self.trainer.global_step == 0
                or self.current_epoch == 900
                or self.current_epoch == 12500
            ):
                if self.trainer.global_step == 0:
                    self.result1 = self.plot_figure_1()
                    plt.imshow(self.result1, cmap="Greys", vmin=0, vmax=0.01)
                    plt.colorbar()
                    plt.title("Initialization")
                    plt.savefig("fig1_init2.png")
                elif self.current_epoch == 900:  # 900 for the seed 63656
                    self.result2 = self.plot_figure_1()
                elif self.current_epoch == 12500:
                    self.result3 = self.plot_figure_1()

                    # plot the results
                    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))

                    im1 = axes[0].imshow(self.result1, cmap="Greys", vmin=0, vmax=0.01)
                    axes[0].xaxis.set_tick_params(labelbottom=False)
                    axes[0].yaxis.set_tick_params(labelleft=False)
                    axes[0].set_xticks([])
                    axes[0].set_yticks([])

                    im2 = axes[1].imshow(self.result2, cmap="Greys", vmin=0, vmax=0.01)
                    axes[1].xaxis.set_tick_params(labelbottom=False)
                    axes[1].yaxis.set_tick_params(labelleft=False)
                    axes[1].set_xticks([])
                    axes[1].set_yticks([])

                    im3 = axes[2].imshow(self.result3, cmap="Greys", vmin=0, vmax=0.01)
                    axes[2].xaxis.set_tick_params(labelbottom=False)
                    axes[2].yaxis.set_tick_params(labelleft=False)
                    axes[2].set_xticks([])
                    axes[2].set_yticks([])
                    cbar = fig.colorbar(
                        im3, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02
                    )

                    axes[0].set_title("Initialization", fontsize=20)
                    axes[1].set_title("During training", fontsize=20)
                    axes[2].set_title("After training", fontsize=20)

                    plt.savefig("Figure1_transformer.svg", format="svg")

        # training
        panel_name = "Train"
        _, _, _, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        if self.hparams.adversarial_training is True:
            _, _, _, loss_adversarial = self._forward(
                self.adversarial_prompts, completion_loss=True
            )
            self.log(f"{panel_name}/loss_adversarial", loss_adversarial)

            with torch.no_grad():
                _, _, _, loss_adversarial_full = self._forward(
                    self.adversarial_prompts, completion_loss=False
                )
                self.log(
                    f"{panel_name}/loss_adversarial_prompt",
                    loss_adversarial_full - loss_adversarial,
                )

            loss += loss_adversarial

        if self.hparams.extrapolation_training is True:
            _, _, _, loss_extrapolation = self._forward(
                self.extrapolation_prompts, completion_loss=True
            )
            self.log(f"{panel_name}/loss_extrapolation", loss_extrapolation)

            with torch.no_grad():
                _, _, _, loss_extrapolation_full = self._forward(
                    self.extrapolation_prompts, completion_loss=False
                )
                self.log(
                    f"{panel_name}/loss_extrapolation_prompt",
                    loss_extrapolation_full - loss_extrapolation,
                )

            loss += loss_extrapolation

        return loss

    def plot_figure_1(self, plot_2=False):
        # generating all sequences of max length sequence_length
        length = 8
        prompts = []
        symbols = [A_token.item(), B_token.item()]
        for i in range(1, length + 1):
            sequences = torch.tensor(list(product(symbols, repeat=i)), dtype=torch.long)
            # add SOS
            sequences = torch.cat(
                (
                    torch.ones((sequences.shape[0], 1), dtype=torch.long) * SOS_token,
                    sequences,
                    torch.ones((sequences.shape[0], 1), dtype=torch.long) * EOS_token,
                ),
                dim=1,
            )
            prompts.extend(sequences.tolist())

        # calculate the probability of a sequence given by the model
        list_of_probab = []
        for sequence in prompts:
            prompt = torch.Tensor([sequence[:-1]]).long().to(self.hparams.device)
            tgt_mask = get_tgt_mask(size=(prompt.size(1)), device=self.hparams.device)

            if self.hparams.model == "transformer":
                pred = self.model(
                    src=prompt,
                    mask=tgt_mask,
                    src_key_padding_mask=create_pad_mask(prompt),
                )
            elif self.hparams.model == "linear" or self.hparams.model == "lstm":
                pred = self.model(src=prompt)
            elif self.hparams.model == "mamba":
                pred = self.model(prompt)
            elif self.hparams.model == "xlstm":
                pred = self.model(prompt)
                pred = pred.permute(0, 2, 1)
                # raise ValueError(f"shape of pred: {pred.shape}, pred {pred}")

            pred = pred.squeeze(0)
            pred = nn.functional.softmax(pred, dim=0)  # make the columns sum to 1
            probability = 0
            for i, element in enumerate(sequence[1:], 1):
                probability += math.log(pred[element][i - 1])

            probability = math.exp(probability)
            list_of_probab.append([sequence, probability])

        # separate the list with the rules
        rule1_met = [check_same_number_as_bs(np.array(t[0])) for t in list_of_probab]
        rule2_met = [check_as_before_bs(np.array(t[0])) for t in list_of_probab]

        not_rule1_not_rule2 = []
        rule1_not_rule2 = []
        rule2_not_rule1 = []
        rule1_and_rule2 = []

        # list of probabilities of each category
        for i, element in enumerate(list_of_probab):
            if not rule1_met[i] and not rule2_met[i]:
                not_rule1_not_rule2.append(element[1])
            elif rule1_met[i] and not rule2_met[i]:
                rule1_not_rule2.append(element[1])
            elif not rule1_met[i] and rule2_met[i]:
                rule2_not_rule1.append(element[1])
            else:
                rule1_and_rule2.append(element[1])

        if plot_2 is True:
            return (
                rule1_and_rule2,
                rule1_not_rule2,
                rule2_not_rule1,
                not_rule1_not_rule2,
            )
        else:
            # sample from each category and create a squace
            C_12 = torch.Tensor(choices(rule1_and_rule2, k=10)).reshape(2, 5)
            C_1 = torch.Tensor(choices(rule1_not_rule2, k=70)).reshape(14, 5)
            C_2 = torch.Tensor(choices(rule2_not_rule1, k=22)).reshape(2, 11)
            C = torch.Tensor(choices(not_rule1_not_rule2, k=154)).reshape(14, 11)

            result = torch.cat(
                (torch.cat((C_2, C_12), dim=1), torch.cat((C, C_1), dim=1))
            )
            return result

    def validation_step(self, batch, batch_idx):
        panel_name = "Val"

        X, X_expected, pred, loss = self._forward(batch)

        if self.hparams.model == "linear":
            self.log(f"{panel_name}/loss", nn.CrossEntropyLoss()(pred, X_expected))
            self.log(f"{panel_name}/loss_ignore_pad", loss)
        else:
            self.log(f"{panel_name}/loss", loss)

        self.log(f"{panel_name}/kl", loss - self.data_entropy)

        # pick most likely token and calculate and log accuracy
        pred_tokens = self._pick_next_tokens(pred)
        accuracy = torch.sum(pred_tokens == X_expected) / X_expected.numel()
        self.log(f"{panel_name}/accuracy", accuracy)

        (
            prompts,
            metrics,
            prompts_finished,
            metrics_finished,
            ood_prompts,
            ood_metrics,
            ood_prompts_finished,
            ood_metrics_finished,
            sos_prompts,
            sos_metrics,
            sos_prompts_finished,
            sos_metrics_finished,
        ) = self.eval_prompt_prediction()

        self._log_dict(name=f"{panel_name}/ID", dictionary=metrics.to_dict())
        self._log_dict(
            name=f"{panel_name}/ID/finished", dictionary=metrics_finished.to_dict()
        )

        self._log_dict(name=f"{panel_name}/OOD", dictionary=ood_metrics.to_dict())
        self._log_dict(
            name=f"{panel_name}/OOD/finished", dictionary=ood_metrics_finished.to_dict()
        )
        self._log_dict(name=f"{panel_name}/SOS", dictionary=sos_metrics.to_dict())
        self._log_dict(
            name=f"{panel_name}/SOS/finished", dictionary=sos_metrics_finished.to_dict()
        )

        if False:  # isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            logger: pl.loggers.wandb.WandbLogger = self.logger

            # log the prompts
            prompts2str = lambda data: [
                ["".join([str(t) for t in p])] for p in data.cpu().numpy().tolist()
            ]
            # convert the prompt tensors to strings
            prompts_str = prompts2str(prompts)
            ood_prompts_str = prompts2str(ood_prompts)
            sos_prompts_str = prompts2str(sos_prompts)

            prompts_finished_str = prompts2str(prompts_finished)
            ood_prompts_finished_str = prompts2str(ood_prompts_finished)
            sos_prompts_finished_str = prompts2str(sos_prompts_finished)

            columns = ["completion"]

            # data should be a list of lists
            logger.log_text(
                key="id_prompt_completions", columns=columns, data=prompts_str
            )
            logger.log_text(
                key="ood_prompt_completions", columns=columns, data=ood_prompts_str
            )
            logger.log_text(
                key="sos_prompt_completions", columns=columns, data=sos_prompts_str
            )

            logger.log_text(
                key="id_prompt_completions_finished",
                columns=columns,
                data=prompts_finished_str,
            )
            logger.log_text(
                key="ood_prompt_completions_finished",
                columns=columns,
                data=ood_prompts_finished_str,
            )
            logger.log_text(
                key="sos_prompt_completions_finished",
                columns=columns,
                data=sos_prompts_finished_str,
            )

    def _log_dict(self, name, dictionary):
        for key, value in dictionary.items():
            self.log(f"{name}/{key}", value)

    @property
    def test_prompts_src(self):
        ds = self.trainer.datamodule.test_dataset.data.view(-1)
        return ds[ds != PAD_token.item()].long().to(self.hparams.device)

    def eval_prompt_prediction(self, max_length: Optional[int] = None):
        if max_length is None:
            max_length = self.hparams.max_pred_length

        (
            prompts,
            metrics,
            prompts_finished,
            metrics_finished,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_in_distribution, max_length
        )

        (
            ood_prompts,
            ood_metrics,
            ood_prompts_finished,
            ood_metrics_finished,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_out_of_distribution, max_length
        )

        # prompt prediction for a batch of SOS tokens
        sos_prompts = (
            torch.ones(
                (self.hparams.batch_size, 1),
                dtype=torch.long,
                device=self.hparams.device,
            )
            * SOS_token.item()
        )
        (
            sos_prompts,
            sos_metrics,
            sos_prompts_finished,
            sos_metrics_finished,
        ) = self._calc_prompt_pred_metrics(sos_prompts, max_length)

        return (
            prompts,
            metrics,
            prompts_finished,
            metrics_finished,
            ood_prompts,
            ood_metrics,
            ood_prompts_finished,
            ood_metrics_finished,
            sos_prompts,
            sos_metrics,
            sos_prompts_finished,
            sos_metrics_finished,
        )

    def _calc_prompt_pred_metrics(self, prompts, max_length):
        prompt_pred = self._predict(max_length=max_length, prompt=prompts)

        metrics, finished = self._calc_grammar_metrics(prompt_pred)

        if sum(finished) == 0:
            prompt_pred_finished = torch.tensor(
                [], dtype=torch.long, device=self.hparams.device
            )
            metrics_finished = GrammarMetrics()
        else:
            # filter out finished prompts only
            # and cut them at the first EOS
            prompt_pred_finished = [
                p for p, f in zip(prompt_pred, finished) if f == True
            ]
            for i, p in enumerate(prompt_pred_finished):
                first_eos = torch.where(p == EOS_token.item())[0][0]
                prompt_pred_finished[i][first_eos:] = (
                    torch.ones_like(
                        prompt_pred_finished[i][first_eos:], device=p.device
                    )
                    * EOS_token.item()
                )

            prompt_pred_finished = torch.stack(prompt_pred_finished)

            metrics_finished, _ = self._calc_grammar_metrics(prompt_pred_finished)

        return (
            prompt_pred,
            metrics,
            prompt_pred_finished,
            metrics_finished,
        )

    def _calc_grammar_metrics(self, prompt_pred, eps: float = 1e-8):
        if self.hparams.grammar == "aNbNcN":
            rule_2 = [check_as_before_bs_before_cs(p) for p in prompt_pred]
            rule_1 = [check_same_number_as_bs_cs(p) for p in prompt_pred]
            rule_2_completion = [
                check_as_before_bs_before_cs(p[self.hparams.test_prompt_length + 1 :])
                # +1 is for the SOS token
                for p in prompt_pred
            ]
            rule_3 = []

        elif self.hparams.grammar in ["aNbN", "abN", "aNbM", "aNbNaN"]:
            rule_2 = [check_as_before_bs(p) for p in prompt_pred]
            rule_1 = [check_same_number_as_bs(p) for p in prompt_pred]

            if self.hparams.grammar == "aNbN":
                rule_3 = [check_parity(p) for p in prompt_pred]
            else:
                rule_3 = []

            if self.hparams.grammar != "aNbNaN":
                rule_2_completion = [
                    check_as_before_bs(p[self.hparams.test_prompt_length + 1 :])
                    # +1 is for the SOS token
                    for p in prompt_pred
                ]
            else:
                rule_2_completion = []
        elif self.hparams.grammar == "baN":
            rule_2 = [check_begins_with_b(p) for p in prompt_pred]
            rule_1 = [check_even_number_of_as(p) for p in prompt_pred]
            rule_2_completion = [
                check_begins_with_b(
                    p[self.hparams.test_prompt_length + 1 :]
                )  # +1 is for the SOS token
                for p in prompt_pred
            ]
            rule_3 = []
        elif self.hparams.grammar == "bbaN":
            rule_2 = [check_bs_before_as(p) for p in prompt_pred]
            rule_1 = [check_even_number_of_as_end(p) for p in prompt_pred]
            rule_2_completion = [
                check_bs_before_as(
                    p[self.hparams.test_prompt_length + 1 :]
                )  # +1 is for the SOS token
                for p in prompt_pred
            ]
            rule_3 = []
        elif (
            self.hparams.grammar == "parentheses_and_brackets"
            or self.hparams.grammar == "not_nested_parentheses_and_brackets"
        ):
            rule_2 = [check_matched_parentheses(p) for p in prompt_pred]
            rule_1 = [check_matched_brackets(p) for p in prompt_pred]
            rule_2_completion = [
                check_matched_parentheses(p[3:]) for p in prompt_pred  # omit SOS, ), (
            ]
            rule_3 = []
        elif self.hparams.grammar == "separated_brackets_and_parentheses":
            rule_1 = [
                check_matched_brackets(p) and check_matched_parentheses(p)
                for p in prompt_pred
            ]
            rule_2 = [check_matched_parentheses_and_brackets(p) for p in prompt_pred]
        else:
            rule_2 = []
            rule_1 = []
            rule_2_completion = []
            rule_3 = []

        # general metrics
        grammatical = [self.grammar_rules(p) for p in prompt_pred]
        finished = [check_sequence_finished(p) for p in prompt_pred]

        metrics = GrammarMetrics(
            rule_2_accuracy=sum(rule_2) / (len(rule_2) + eps),
            rule_1_accuracy=sum(rule_1) / (len(rule_1) + eps),
            finished_accuracy=sum(finished) / (len(finished) + eps),
            grammatical_accuracy=sum(grammatical) / (len(grammatical) + eps),
            rule_2_completion_accuracy=sum(rule_2_completion)
            / (len(rule_2_completion) + eps),
            rule_3_accuracy=sum(rule_3) / (len(rule_2) + eps),
        )

        return metrics, finished

    def _forward(self, X, completion_loss=False):
        """
        Forward pass for calculating the model predictions and the loss.
        :param completion_loss: calculate loss only on prompt completion
        :param X:
        :return:
        """

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        X_input = X[:, :-1]
        X_expected = X[:, 1:]

        # Get mask to mask out the next words
        causal_mask = get_tgt_mask(X_input.size(1), device=self.hparams.device)

        # Standard training except we pass in X_input and causal_mask

        if self.hparams.model == "transformer":
            pred = self.model(
                src=X_input,
                mask=causal_mask,
                src_key_padding_mask=create_pad_mask(X_input),
            )
        elif self.hparams.model == "linear" or self.hparams.model == "lstm":
            pred = self.model(src=X_input)
        elif self.hparams.model == "mamba":
            pred = self.model(X_input)
        elif self.hparams.model == "xlstm":
            pred = self.model(X_input)
            pred = pred.permute(0, 2, 1)

        if completion_loss is False:
            loss = self.hparams.loss_fn(pred, X_expected)
        else:
            loss = self.hparams.loss_fn(
                pred[
                    :, :, self.hparams.test_prompt_length + 1 :
                ],  # +1 is for the SOS token
                X_expected[:, self.hparams.test_prompt_length + 1 :],
            )

        return X, X_expected, pred, loss

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

        return self._predict(max_length, prompt)

    def _predict(
        self, max_length: int = 32, prompt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inner method for predicting a sequence.
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

        if self.hparams.model == "linear" or "xlstm":
            max_length = self.hparams.max_data_length - prompt.shape[1]

        for _ in range(max_length):
            # Get mask to mask out the next words
            tgt_mask = get_tgt_mask(size=(prompt.size(1)), device=self.hparams.device)

            # forward pass
            if self.hparams.model == "transformer":
                pred = self.model(
                    src=prompt,
                    mask=tgt_mask,
                    src_key_padding_mask=create_pad_mask(prompt),
                )

            elif self.hparams.model == "linear" or self.hparams.model == "lstm":
                pred = self.model(src=prompt)
            elif self.hparams.model == "mamba":
                pred = self.model(prompt)
            elif self.hparams.model == "xlstm":
                pred = self.model(prompt)
                pred = pred.permute(0, 2, 1)

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
            if len(pred.shape) > 2:
                next_items = torch.cat(
                    [
                        torch.multinomial(torch.softmax(p, dim=1).T, num_samples=1).T
                        for p in pred
                    ]
                )
            else:
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
