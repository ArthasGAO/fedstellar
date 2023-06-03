import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import math

###############################
#    Multilayer Perceptron    #
###############################


class CNN(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve MNIST with PyTorch Lightning.
    """

    def __init__(
            self, metric=Accuracy, out_channels=2, lr_rate=0.005, seed=None
    ):  # low lr to avoid overfitting

        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.lr_rate = lr_rate
        self.metric = metric(num_classes=2, task="multiclass")
        # self.embedding_dim=300
        self.output_dim=out_channels
        self.filter_sizes = [2, 3, 4]
        self.n_filters = math.ceil(300 * len(self.filter_sizes) / 3)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.n_filters, kernel_size=(fs, 300)) for fs in self.filter_sizes
        ])
        self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters, self.output_dim)
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """ """
        x = x.unsqueeze(1)
        conved = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, n_filters, sent_len), ...] * len(filter_sizes)
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [(batch_size, n_filters), ...] * len(filter_sizes)
        cat = self.dropout(torch.cat(pooled, dim=1))
        out = self.fc(cat)
        return out

    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_id):
        """ """
        x, y = batch
        # y = y.to(torch.float32)
        logits = self(x)
        y = y.to(torch.long)
        loss = self.loss_fn(logits, y)
        self.log("Train/Loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        """ """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("TrainEnd/Loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """ """
        x, y = batch
        # y = y.to(torch.float32)
        logits = self(x)
        y = y.to(torch.long)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits,dim=1)
        metric = self.metric(out, y)
        self.log("Validation/Loss", loss, prog_bar=True)
        self.log("Validation/Accuracy", metric, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """ """
        x, y = batch
        # y = y.to(torch.float32)
        logits = self(x)
        y = y.to(torch.long)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits,dim=1)
        metric = self.metric(out, y)
        self.log("Test/Loss", loss, prog_bar=True)
        self.log("Test/Accuracy", metric, prog_bar=True)
        return loss
