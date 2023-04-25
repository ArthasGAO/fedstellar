import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import functional as F
from torchmetrics import Accuracy


###############################
#    Multilayer Perceptron    #
###############################


class RNN(pl.LightningModule):
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
        self.embedding_dim=300
        self.hidden_dim=256
        self.n_layers=1
        self.bidirectional=True
        self.output_dim=out_channels

        self.encoder = nn.LSTM(self.embedding_dim, 
            self.hidden_dim, 
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=0.5,
            batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.dropout = nn.Dropout(0.5)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """ """
        # output, (hn, cn) = self.encoder(x)

        packed_output, (hidden, cell) = self.encoder(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        out = self.fc(hidden)
        out = F.softmax(out, dim=1)
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

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
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
