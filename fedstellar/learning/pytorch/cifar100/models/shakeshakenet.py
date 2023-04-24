import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
import torch.nn.functional as F

###############################
#    Multilayer Perceptron    #
###############################

IMAGE_SIZE = 32

class ShakeShakeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShakeShakeBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return F.relu(out1 + out2)


class ShakeShakeNet(pl.LightningModule):
    def __init__(self, 
                 in_channels=1,
                out_channels=100,
                metric=Accuracy(num_classes=100, task="multiclass"),
                lr_rate=0.001,
                seed=None,
                num_classes=100):
        self.config = {                                                                                                                                                                                                          
            'lr': 8.0505e-05,
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True,
            'weight_decay': 0.001
        } 

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
        
        super(ShakeShakeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 3, stride=1)
        self.layer2 = self._make_layer(32, 64, 3, stride=1)
        self.layer3 = self._make_layer(64, 128, 3, stride=1)
        self.layer4 = self._make_layer(128, 256, 3, stride=1)
        self.fc = nn.Linear(256, num_classes)

        self.metric = metric
        self.lr_rate = lr_rate
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(ShakeShakeBlock(in_channels, out_channels, stride if i == 0 else 1))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
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
