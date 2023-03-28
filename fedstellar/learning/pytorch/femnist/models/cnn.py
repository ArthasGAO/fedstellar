import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy

###############################
#    Multilayer Perceptron    #
###############################

IMAGE_SIZE = 28


class CNN(pl.LightningModule):
    """
    Convolutional Neural Network (CNN) to solve MNIST with PyTorch Lightning.
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=62,
            metric=Accuracy(num_classes=10, task="multiclass"),
            lr_rate=0.001,
            momentum=0,
            seed=None,
    ):
        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.metric = metric
        self.lr_rate = lr_rate
        self.momentum = momentum

        self.conv1 = nn.Conv2d(in_channels, 32, 7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, out_channels)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """ """
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)

    def configure_optimizers(self):
        """ """
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate, momentum=self.momentum)

    def training_step(self, batch, batch_id):
        """ """
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", metric, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
        return loss
