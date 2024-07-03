from rule_extrapolation.cli import LLMLightningCLI
from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule
from os.path import abspath, dirname, join


def test_cli_fast_dev_run():
    config_path = join(dirname(dirname(abspath(__file__))), "configs", "config.yaml")

    args = [
        "fit",
        "--config",
        config_path,
        "--trainer.fast_dev_run",
        "true",
        "--trainer.logger",
        "null",
        "--data.num_train",
        "32",
        "--data.num_val",
        "16",
        "--data.num_test",
        "8",
        "--data.max_length",
        "4",
    ]
    cli = LLMLightningCLI(
        LightningGrammarModule,
        GrammarDataModule,
        save_config_callback=None,
        run=True,
        args=args,
        parser_kwargs={"parse_as_dict": False},
    )
