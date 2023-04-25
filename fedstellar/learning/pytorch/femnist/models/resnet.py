import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
import torch.nn.functional as F

###############################
#    Multilayer Perceptron    #
###############################

IMAGE_SIZE = 28


class ResNet(pl.LightningModule):
    """
    Convolutional Neural Network (CNN) to solve MNIST with PyTorch Lightning.
    """

    def __init__(self,
            in_channels=1,
            out_channels=62,
            metric=Accuracy(num_classes=62, task="multiclass"),
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
            'amsgrad': True
        } 
        super().__init__()
        self.metric = metric
        self.lr_rate = lr_rate

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512)
        # )
        
        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        )
        
        self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        )

        # self.res3 = nn.Sequential(nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # ), nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True))
        # )
        

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)), 
            nn.Flatten(), 
            nn.Linear(128, out_channels),
            
        )


        self.loss_fn = nn.CrossEntropyLoss()
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # res1
        x = self.layer1(x)
        x = self.res1(x) + x

        # res2
        x = self.layer2(x)
        x = self.res2(x) + x

        # # res3
        # x = self.layer3(x)
        # x = self.res3(x) + x
        
        x = self.classifier(x)
        x = self.log_softmax(x)
        return x



    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), 
                                lr=self.config['lr'], 
                                betas=(self.config['beta1'], self.config['beta2']), 
                                amsgrad=self.config['amsgrad'])

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
        self.log("Accuracy/Loss", metric, prog_bar=True)
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
