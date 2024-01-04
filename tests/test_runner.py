from pytorch_lightning.trainer import Trainer

from llm_non_identifiability.datamodule import GrammarDataModule
from llm_non_identifiability.runner import LightningGrammarModule


def test_fit(n_train, n_val, n_test):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule()
    dm = GrammarDataModule(n_train=n_train, n_val=n_val, n_test=n_test)
    trainer.fit(runner, datamodule=dm)


def test_predict(n_train, n_val, n_test):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule()
    dm = GrammarDataModule(n_train=n_train, n_val=n_val, n_test=n_test)
    trainer.fit(runner, datamodule=dm)

    trainer.predict(runner, datamodule=dm)
