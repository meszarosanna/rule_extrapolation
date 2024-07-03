from pytorch_lightning.cli import LightningCLI

from rule_extrapolation.datamodule import GrammarDataModule
from rule_extrapolation.runner import LightningGrammarModule


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

        parser.link_arguments("data.grammar", "model.grammar")
        parser.link_arguments("data.max_length", "model.max_data_length")
        parser.link_arguments("data.batch_size", "model.batch_size")


if __name__ == "__main__":
    cli = LLMLightningCLI(
        LightningGrammarModule,
        GrammarDataModule,
        save_config_callback=None,
        run=True,
    )
