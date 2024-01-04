# Importing necessary libraries
import torch
import torch.nn as nn

import llm_non_identifiability.data
from llm_non_identifiability.data import generate_aNbN_grammar_data, batchify_data
from llm_non_identifiability.model import Transformer


def train_loop(model, opt, loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        X = torch.tensor(batch, dtype=torch.long).to(device)
        y = X.clone()

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(
            X,
            y_input,
            tgt_mask,
            model.create_pad_mask(X, 4),
            model.create_pad_mask(y_input, 4),
        )

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            X = torch.tensor(batch, dtype=torch.long).to(device)
            y = X.clone()

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(
                X,
                y_input,
                tgt_mask,
                model.create_pad_mask(X, 4),
                model.create_pad_mask(y_input, 4),
            )

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(
            f"Training loss: {train_loss:.4f}\t Validation loss: {validation_loss:.4f}"
        )

    return train_loss_list, validation_loss_list


def predict(model, input_sequence, max_length=10, SOS_token=2, EOS_token=3):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()

    y_input = torch.tensor([[SOS_token, 0, 0, 0, 1]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        pred = model(
            input_sequence,
            y_input,
            tgt_mask,
            model.create_pad_mask(input_sequence, 4),
            model.create_pad_mask(y_input, 4),
        )

        # print(pred)

        # next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        _, next_item = torch.max(pred[-1].view(-1), dim=-1)

        next_item = torch.tensor([[next_item]], device=device)
        # print(next_item)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


import pytorch_lightning as pl


class LightningGrammarModule(pl.LightningModule):
    def __init__(self, lr=0.01, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.save_hyperparameters()

        self.hparams.loss_fn = nn.CrossEntropyLoss()
        self.model = Transformer(
            num_tokens=5,
            dim_model=8,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout_p=0.1,
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

    def predict_step(self, batch, batch_idx=None, prompt=None, max_length=32):
        X, y = batch

        return self._predict(X[0].view(1, -1), max_length, prompt)

    def _predict(self, src, max_length=32, prompt=None):
        if prompt is None:
            prompt = torch.tensor(
                [[llm_non_identifiability.data.SOS_token.item(), 0, 0, 0, 1]],
                dtype=torch.long,
                device=self.hparams.device,
            )
        for _ in range(max_length):
            # Get mask to mask out the next words
            sequence_length = prompt.size(1)
            tgt_mask = self.model.get_tgt_mask(sequence_length).to(self.hparams.device)

            # Standard training except we pass in y_input and tgt_mask
            pred = self.model(
                src,
                prompt,
                tgt_mask,
                self.model.create_pad_mask(src, 4),
                self.model.create_pad_mask(prompt, 4),
            )

            _, next_item = torch.max(pred[-1].view(-1), dim=-1)

            next_item = torch.tensor([[next_item]], device=self.hparams.device)

            # Concatenate previous input with predicted best word
            prompt = torch.cat((prompt, next_item), dim=1)

            # Stop if model predicts end of sentence
            if (
                next_item.view(-1).item()
                == llm_non_identifiability.data.EOS_token.item()
            ):
                break
        return prompt.view(-1).tolist()


if __name__ == "__main__":
    train_data = generate_aNbN_grammar_data(9000)
    val_data = generate_aNbN_grammar_data(3000)

    train_dataloader = batchify_data(train_data)
    val_dataloader = batchify_data(val_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
        num_tokens=5,
        dim_model=8,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout_p=0.1,
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_list, validation_loss_list = fit(
        model, opt, loss_fn, train_dataloader, val_dataloader, 125
    )

    # Here we test some examples to observe how the model predicts
    examples = [
        torch.tensor([[2, 0, 0, 0, 0, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 0, 0, 1, 1, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device),
    ]

    for idx, example in enumerate(examples):
        result = predict(model, example)
        print(f"Example {idx}")
        print(f"Input: {example.view(-1).tolist()[1:-1]}")
        print(f"Continuation: {result[1:-1]}")
        print()
