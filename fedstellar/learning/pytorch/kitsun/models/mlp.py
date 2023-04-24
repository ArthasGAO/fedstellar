import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import functional as F
from torchmetrics import Accuracy


###############################
#    Multilayer Perceptron    #
###############################


class MLP(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve SYSCALL with PyTorch Lightning.
    """

    def __init__(
            self, metric=Accuracy, out_channels=10, lr_rate=0.001, seed=None
    ):  # low lr to avoid overfitting

        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.lr_rate = lr_rate
        self.metric = metric(num_classes=10, task="multiclass")

        self.l1 = torch.nn.Linear(356, 2048)
        self.batchnorm1 = torch.nn.BatchNorm1d(2048)
        self.dropout = torch.nn.Dropout(0.5)
        self.l2 = torch.nn.Linear(2048, 512)
        self.batchnorm2 = torch.nn.BatchNorm1d(512)
        self.l3 = torch.nn.Linear(512, 256)
        self.batchnorm3 = torch.nn.BatchNorm1d(256)
        self.l4 = torch.nn.Linear(256, 128)
        self.batchnorm4 = torch.nn.BatchNorm1d(128)
        self.l5 = torch.nn.Linear(128, out_channels)

    def forward(self, x):
        """ """
        x = self.l1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.batchnorm4(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.l5(x)
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
