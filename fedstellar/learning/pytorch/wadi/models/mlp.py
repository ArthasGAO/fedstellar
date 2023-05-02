import lightning as pl
import torch
from torch.nn import functional as F
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall


###############################
#    Multilayer Perceptron    #
###############################


class MLP(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve WADI with PyTorch Lightning.
    """

    def __init__(
            self,
            metrics=None,
            out_channels=1, 
            lr_rate=0.001, 
            seed=1000
    ):  # low lr to avoid overfitting

        if metrics is None:
            metrics = [BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix]

        self.metrics = []
        if type(metrics) is list:
            try:
                for m in metrics:
                    self.metrics.append(m(num_classes=2))
            except TypeError:
                raise TypeError("metrics must be a list of torchmetrics.Metric")

        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.example_input_array = torch.zeros(1, 123)
        self.lr_rate = lr_rate

        self.l1 = torch.nn.Linear(123, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, 256)
        self.l4 = torch.nn.Linear(256, 128)
        self.l5 = torch.nn.Linear(128, 64)
        self.l6 = torch.nn.Linear(64, 32)
        self.l7 = torch.nn.Linear(32, 16)
        self.l8 = torch.nn.Linear(16, 8)
        self.l9 = torch.nn.Linear(8, out_channels)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

        self.epoch_num_steps = {"Train": 0, "Validation": 0, "Test": 0}
        self.epoch_loss_sum = {"Train": 0.0, "Validation": 0.0, "Test": 0.0}

        self.epoch_output = {"Train": [], "Validation": [], "Test": []}
        self.epoch_real = {"Train": [], "Validation": [], "Test": []}

    def forward(self, x):
        """ """
        batch_size, features = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        #x = x.view(batch_size, -1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        x = self.l4(x)
        x = torch.relu(x)
        x = self.l5(x)
        x = torch.relu(x)
        x = self.l6(x)
        x = torch.relu(x)
        x = self.l7(x)
        x = torch.relu(x)
        x = self.l8(x)
        x = torch.relu(x)
        x = self.l9(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def log_epoch_metrics_and_loss(self, phase, print_cm=True, plot_cm=True):
        # Log loss
        epoch_loss = self.epoch_loss_sum[phase] / self.epoch_num_steps[phase]
        self.log(f"{phase}Epoch/Loss", epoch_loss, prog_bar=True, logger=True)
        self.epoch_loss_sum[phase] = 0.0

        # Log metrics
        for metric in self.metrics:
            if isinstance(metric, BinaryConfusionMatrix):
                cm = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase]))
                print(f"{phase}Epoch/CM\n", cm) if print_cm else None
                if plot_cm:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 7))
                    ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
                    ax.set_xlabel("Predicted labels")
                    ax.set_ylabel("True labels")
                    ax.set_title("Confusion Matrix")
                    ax.set_xticks([0.5, 1.5])
                    ax.set_yticks([0.5, 1.5])
                    ax.xaxis.set_ticklabels(["Normal", "Attack"])
                    ax.yaxis.set_ticklabels(["Normal", "Attack"])
                    self.logger.experiment.add_figure(f"{phase}Epoch/CM", ax.get_figure(), global_step=self.epoch_global_number[phase])
                    plt.close()
            else:
                metric_name = metric.__class__.__name__.replace("Binary", "")
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
        self.epoch_real[phase].append(y.unsqueeze(1).float())

        for metric in self.metrics:
            if isinstance(metric, BinaryConfusionMatrix):
                print(f"{phase}/CM\n", metric(y_pred, y)) if print_cm else None
            else:
                metric_name = metric.__class__.__name__.replace("Binary", "")
                metric_value = metric(y_pred, y)
                self.log(f"{phase}/{metric_name}", metric_value, prog_bar=True, logger=True)

    def step(self, batch, phase):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy(logits, y.unsqueeze(1).float())

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.log_metrics(phase, logits, y, print_cm=False)

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












    def training_step(self, batch, batch_id):
        """ """
        x, y = batch
        out = self(x)
        self.training_step_outputs.append(out)
        self.training_step_real.append(y.unsqueeze(1).float())
        loss = F.binary_cross_entropy(out, y.unsqueeze(1).float())
        self.log("Train/Loss", loss, prog_bar=True)
        self.log_metrics("Train", out, y, print_cm=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """ """
        x, y = batch
        out = self(x)
        self.validation_step_outputs.append(out)
        self.validation_step_real.append(y.unsqueeze(1).float())
        loss = F.binary_cross_entropy(out, y.unsqueeze(1).float())
        self.log("Validation/Loss", loss, prog_bar=True)
        self.log_metrics("Validation", out, y, print_cm=False)
        return loss

    def test_step(self, batch, batch_idx):
        """ """
        x, y = batch
        out = self(x)
        self.test_step_outputs.append(out)
        self.test_step_real.append(y.unsqueeze(1).float())
        loss = F.binary_cross_entropy(out, y.unsqueeze(1).float())
        self.log("Test/Loss", loss, prog_bar=True)
        self.log_metrics("Test", out, y, print_cm=False)
        return loss


    def on_train_epoch_end(self):
        out = torch.cat(self.training_step_outputs)
        y = torch.cat(self.training_step_real)
        self.log_metrics("TrainEpoch", out, y, print_cm=True)

        self.training_step_outputs.clear()  # free memory
        self.training_step_real.clear()

    def on_validation_epoch_end(self):
        out = torch.cat(self.validation_step_outputs)
        y = torch.cat(self.validation_step_real)
        self.log_metrics("ValidationEpoch", out, y, print_cm=True)

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_real.clear()
    
    def on_test_epoch_end(self):
        out = torch.cat(self.test_step_outputs)
        y = torch.cat(self.test_step_real)
        self.log_metrics("TestEpoch", out, y, print_cm=True)

        self.test_step_outputs.clear()  # free memory
        self.test_step_real.clear()