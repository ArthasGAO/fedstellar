import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import functional as F
from torchmetrics import Accuracy


###############################
#    Multilayer Perceptron    #
###############################


class MLP(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve MNIST with PyTorch Lightning.
    """

    def __init__(
            self, metric=Accuracy, out_channels=100, lr_rate=0.001, seed=None
    ):  # low lr to avoid overfitting

        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.lr_rate = lr_rate
        self.metric = metric(num_classes=100, task="multiclass")
        self.input_size = 32 * 32 * 3

        self.linear1 = nn.Linear(self.input_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, out_channels)

    def forward(self, x):
        """ """
        batch_size, channels, width, height = x.size()

        # (b, 3, 32, 32) -> (b, 3*32*32)
        x = x.view(batch_size, -1)
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        x = self.linear5(out)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_id):
        """ """
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("Train/Loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """ """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("TrainEnd/Loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("Validation/Loss", loss, prog_bar=True)
        self.log("Validation/Accuracy", metric, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("Test/Loss", loss, prog_bar=True)
        self.log("Test/Accuracy", metric, prog_bar=True)
        return loss
