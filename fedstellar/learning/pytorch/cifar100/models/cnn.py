import lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix

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

        if metrics is None:
            metrics = [MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix]

        self.metrics = []
        if type(metrics) is list:
            try:
                for m in metrics:
                    self.metrics.append(m(num_classes=100))
            except TypeError:
                raise TypeError("metrics must be a list of torchmetrics.Metric")
            
        super().__init__()

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
        
        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

        self.epoch_num_steps = {"Train": 0, "Validation": 0, "Test": 0}
        self.epoch_loss_sum = {"Train": 0.0, "Validation": 0.0, "Test": 0.0}

        self.epoch_output = {"Train": [], "Validation": [], "Test": []}
        self.epoch_real = {"Train": [], "Validation": [], "Test": []}
        

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

    def log_epoch_metrics_and_loss(self, phase, print_cm=True, plot_cm=True):
        # Log loss
        epoch_loss = self.epoch_loss_sum[phase] / self.epoch_num_steps[phase]
        self.log(f"{phase}Epoch/Loss", epoch_loss, prog_bar=True, logger=True)
        self.epoch_loss_sum[phase] = 0.0

        # Log metrics
        for metric in self.metrics:
            if isinstance(metric, MulticlassConfusionMatrix):
                cm = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase]))
                print(f"{phase}Epoch/CM\n", cm) if print_cm else None
                if plot_cm:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(20, 14))
                    ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 8})
                    ax.set_xlabel("Predicted labels")
                    ax.set_ylabel("True labels")
                    ax.set_title("Confusion Matrix")
                    ax.set_xticks(range(100))
                    ax.set_yticks(range(100))
                    ax.xaxis.set_ticklabels([i for i in range(100)])
                    ax.yaxis.set_ticklabels([i for i in range(100)])
                    self.logger.experiment.add_figure(f"{phase}Epoch/CM", ax.get_figure(), global_step=self.epoch_global_number[phase])
                    plt.close()
            else:
                metric_name = metric.__class__.__name__.replace("Multiclass", "")
                metric_value = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase])).detach()
                self.log(f"{phase}Epoch/{metric_name}", metric_value, prog_bar=True, logger=True)

            metric.reset()

        self.epoch_output[phase].clear()
        self.epoch_real[phase].clear()

        # Reset step count
        self.epoch_num_steps[phase] = 0

        # Increment epoch number
        self.epoch_global_number[phase] += 1

    def log_metrics(self, phase, y_pred, y, print_cm=False):
        self.epoch_output[phase].append(y_pred.detach())
        self.epoch_real[phase].append(y.detach())

        for metric in self.metrics:
            if isinstance(metric, MulticlassConfusionMatrix):
                print(f"{phase}/CM\n", metric(y_pred, y)) if print_cm else None
            else:
                metric_name = metric.__class__.__name__.replace("Multiclass", "")
                metric_value = metric(y_pred, y)
                self.log(f"{phase}/{metric_name}", metric_value, prog_bar=True, logger=True)

    def step(self, batch, phase):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y_pred = torch.argmax(logits, dim=1)

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.log_metrics(phase, y_pred, y, print_cm=False)

        # Avoid memory leak when logging loss values
        self.epoch_loss_sum[phase] += loss
        self.epoch_num_steps[phase] += 1

        return loss

    def training_step(self, batch, batch_id):
        """
        Training step for the model.
        Args:
            batch:
            batch_id:

        Returns:
        """
        return self.step(batch, "Train")

    def on_train_epoch_end(self):
        self.log_epoch_metrics_and_loss("Train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Validation")

    def on_validation_epoch_end(self):
        self.log_epoch_metrics_and_loss("Validation")

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Test")

    def on_test_epoch_end(self):
        self.log_epoch_metrics_and_loss("Test")
