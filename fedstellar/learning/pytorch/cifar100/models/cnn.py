import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
import torch.nn.functional as F

###############################
#    Multilayer Perceptron    #
###############################

IMAGE_SIZE = 32


class CNN(pl.LightningModule):
    """
    Convolutional Neural Network (CNN) to solve MNIST with PyTorch Lightning.
    """

    def __init__(self,
            in_channels=1,
            out_channels=100,
            metric=Accuracy(num_classes=100, task="multiclass"),
            lr_rate=0.001,
            seed=None,
    ):
        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.config = {                                                                                                                                                                                                          
            'lr': 8.0505e-05,
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True,
            'weight_decay': 0.001
        } 
        super().__init__()
        self.metric = metric
        self.lr_rate = lr_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ), nn.Sequential( 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1028, kernel_size=3, padding=1),
            nn.BatchNorm2d(1028),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res3 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=1028, out_channels=1028, kernel_size=3, padding=1),
            nn.BatchNorm2d(1028),
            nn.ReLU(inplace=True)
        ), nn.Sequential( 
            nn.Conv2d(in_channels=1028, out_channels=1028, kernel_size=3, padding=1),
            nn.BatchNorm2d(1028),
            nn.ReLU(inplace=True))
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(2), 
            nn.Flatten(), 
            nn.Linear(1028, 100)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.conv5(x)
        x = self.res3(x) + x
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), 
                                lr=self.config['lr'], 
                                betas=(self.config['beta1'], self.config['beta2']), 
                                amsgrad=self.config['amsgrad'],
                                weight_decay=self.config['weight_decay'])

    def training_step(self, batch, batch_id):
        """ """
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("Train/Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("Validation/Loss", loss, prog_bar=True)
        self.log("Validation/Accuracy", metric, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(self(x), y)
        out = torch.argmax(logits, dim=1)
        metric = self.metric(out, y)
        self.log("Test/Loss", loss, prog_bar=True)
        self.log("Test/Accuracy", metric, prog_bar=True)
        return loss
