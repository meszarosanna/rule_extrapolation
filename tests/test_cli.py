from llm_non_identifiability.cli import LLMLightningCLI
from llm_non_identifiability.runner import LightningGrammarModule
from llm_non_identifiability.datamodule import GrammarDataModule
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
        "128",
        "--data.num_val",
        "64",
        "--data.num_test",
        "32",
        "--data.max_length",
        "16",
    ]
    cli = LLMLightningCLI(
        LightningGrammarModule,
        GrammarDataModule,
        save_config_callback=None,
        run=True,
        args=args,
        parser_kwargs={"parse_as_dict": False},
    )
