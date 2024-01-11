from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

from llm_non_identifiability.datamodule import GrammarDataModule
from llm_non_identifiability.runner import LightningGrammarModule


class LLMLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--notes",
            type=str,
            default=None,
            help="Notes for the run on Weights and Biases",
        )
        parser.add_argument(
            "--tags",
            type=str,
            nargs="*",  # 0 or more values expected => creates a list
            default=None,
            help="Tags for the run on Weights and Biases",
        )


if __name__ == "__main__":
    cli = LLMLightningCLI(
        LightningGrammarModule,
        GrammarDataModule,
        save_config_callback=None,
        run=True,
    )
