from copy import deepcopy
from os.path import join

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from rule_extrapolation.datamodule import GrammarDataModule
from rule_extrapolation.runner import LightningGrammarModule

if __name__ == "__main__":
    GRAMMAR = "aNbNaN"
    MAX_EPOCHS = 5000

    run = wandb.init(
        project="llm-non-identifiability",
        entity="causal-representation-learning",
        job_type="finetune",
    )
    model_name = "model-jtt1bx1l:best"
    model_artifact = run.use_artifact(
        f"causal-representation-learning/llm-non-identifiability/{model_name}",
        type="model",
    )
    artifact_dir = model_artifact.download()

    run.finish()

    # source: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html#checkpoint-loading
    checkpoint = "model.ckpt"
    PATH = join(artifact_dir, checkpoint)

    # To load a model along with its weights and hyperparameters use the following method
    model: LightningGrammarModule = LightningGrammarModule.load_from_checkpoint(
        PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        grammar=GRAMMAR,
        max_data_length=128,
        #   relu_rescale=100000.0,
        layer_norm_eps=0.0,
    )

    datamodule = GrammarDataModule(
        max_length=model.hparams.max_data_length,
        batch_size=model.hparams.batch_size,
        grammar=GRAMMAR,
    )

    bad_model = deepcopy(model)
    # bad_model.model.relu_rescale = torch.nn.Parameter(
    # torch.tensor(10000.0), requires_grad=False
    # )

    assert model.model.relu_rescale.requires_grad == False

    logger = WandbLogger(
        entity="causal-representation-learning",
        project="llm-non-identifiability",
        # name="finetune-bad",
        name="finetune-good",
        log_model=True,
    )

    trainer = Trainer(max_epochs=MAX_EPOCHS, logger=logger, check_val_every_n_epoch=50)
    trainer.fit(
        bad_model,
        datamodule=datamodule,
    )
