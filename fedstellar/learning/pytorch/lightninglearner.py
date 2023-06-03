# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 

import logging
import pickle
import time
import traceback
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import numpy as np
from itertools import product
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary, TQDMProgressBar
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import copy
from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.learner import NodeLearner
from torch.nn import functional as F
from torchmetrics import Accuracy

###########################
#    LightningLearner     #
###########################
from fedstellar.learning.modelmetrics import ModelMetrics


class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None, logger=None):
        self.model = model
        # self.model = torch.compile(model)  # PyTorch 2.0
        self.latest_model = copy.deepcopy(self.model)
        self.data = data
        self.config = config
        self.logger = logger
        self.__trainer = None
        self.epochs = 1
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

        # FL information
        self.round = 0
        # self.local_step = 0
        # self.global_step = 0

        self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)

    def set_model(self, model):
        self.model = model
        self.latest_model = copy.deepcopy(self.model)

    def set_data(self, data):
        self.data = data

    def encode_parameters(self, params=None, contributors=None, metrics=None):
        if params is None:
            params = self.model.state_dict()
        if metrics is None:
            metrics = ModelMetrics()
        array = [val.cpu().numpy() for _, val in params.items()]
        return pickle.dumps((array, contributors, metrics))

    def decode_parameters(self, data):
        try:
            params, contributors, metrics = pickle.loads(data)

            params_dict = zip(self.model.state_dict().keys(), params)
            return (
                OrderedDict({k: torch.tensor(v) for k, v in params_dict}),
                contributors,
                metrics,
            )
        except DecodingParamsError:
            raise DecodingParamsError("Error decoding parameters")

    def check_parameters(self, params):
        # Check ordered dict keys
        if set(params.keys()) != set(self.model.state_dict().keys()):
            return False
        # Check tensor shapes
        for key, value in params.items():
            if value.shape != self.model.state_dict()[key].shape:
                return False
        return True

    def set_parameters(self, params):
        try:
            self.model.load_state_dict(params)
            self.latest_model = copy.deepcopy(self.model)
        except ModelNotMatchingError:
            raise ModelNotMatchingError("Not matching models")

    def get_parameters(self):
        return self.model.state_dict()

    def set_epochs(self, epochs):
        self.epochs = epochs

    def fit(self):
        try:
            if self.epochs > 0:
                self.create_trainer()
                self.__trainer.fit(self.model, self.data)
                self.__trainer = None
        except Exception as e:
            logging.error("[NodeLearner.fit] Something went wrong with pytorch lightning. {}".format(e))

    def interrupt_fit(self):
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self):
        try:
            if self.epochs > 0:
                self.create_trainer()
                self.__trainer.test(self.model, self.data, verbose=True)
                self.__trainer = None
                # results = self.__trainer.test(self.model, self.data, verbose=True)
                # loss = results[0]["Test/Loss"]
                # metric = results[0]["Test/Accuracy"]
                # self.__trainer = None
                # self.log_validation_metrics(loss, metric, self.round)
                # return loss, metric
            else:
                return None
        except Exception as e:
            logging.error("[NodeLearner.evaluate] Something went wrong with pytorch lightning. {}".format(e))
            return None

    """
    def predict(self):
        try:
            if self.epochs > 0:
                self.create_trainer()
                predictions = self.__trainer.predict(self.model, self.data)
                return predictions
        except Exception as e:
            logging.error("[NodeLearner.predict] Something went wrong with pytorch lightning. {}".format(e))
            return None
    """

    """
        Source: https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L14/helper_evaluation.py
    """
    def compute_confusion_matrix(self, node_name: str, backdoor: bool = False):
        model = self.model

        backdoor_dataloader = self.data.backdoor_dataloader()
        test_dataloader = self.data.test_dataloader()
        if backdoor:
            log_key = "Backdoor Confmat"
            data_loader = backdoor_dataloader
        else:
            log_key = "Testdata Confmat"
            data_loader = test_dataloader

        all_targets, all_predictions = [], []
        with torch.no_grad():
            for i, (features, targets) in enumerate(data_loader):
                # features = features.to(device)
                targets = targets
                logits = model(features)
                _, predicted_labels = torch.max(logits, 1)
                all_targets.extend(targets.to('cpu'))
                all_predictions.extend(predicted_labels.to('cpu'))

        all_predictions = all_predictions
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # confmat = ConfusionMatrix(task="multiclass", num_classes=model.out_channels)
        # matrix = confmat(t_all_predictions, t_all_targets)
        # print("Torch metrics confmat")
        # print(matrix)

        class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
        if class_labels.shape[0] == 1:
            if class_labels[0] != 0:
                class_labels = np.array([0, class_labels[0]])
            else:
                class_labels = np.array([class_labels[0], 1])
        n_labels = class_labels.shape[0]
        lst = []
        z = list(zip(all_targets, all_predictions))
        for combi in product(class_labels, repeat=2):
            lst.append(z.count(combi))
        mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)

        if backdoor:
            # Evaluate backdoor accuracy
            target_label = data_loader.dataset.target_label
            num_predicted_target = mat.sum(axis=0)[target_label]
            num_samples = len(data_loader.dataset)
            attacker_success = num_predicted_target / num_samples
            self.logger.log_metrics({"Test/ASR-backdoor": attacker_success}, step=self.logger.local_step)
            logging.info("Computed ASR Backdoor: {}".format(attacker_success))
        else:
            # Evaluate targeted data poisoning accuracy
            target_label = backdoor_dataloader.dataset.target_label
            target_changed_label = backdoor_dataloader.dataset.target_changed_label
            num_target_changed_predicted = mat[target_label][target_changed_label]
            num_target_samples = mat.sum(axis=1)[target_label]
            attacker_success = num_target_changed_predicted / num_target_samples
            self.logger.log_metrics({"Test/ASR-targeted": attacker_success}, step=self.logger.local_step)
            logging.info("Computed ASR Test: {}".format(attacker_success))

        class_names = data_loader.dataset.dataset.classes
        class_names_short = list(data_loader.dataset.dataset.class_to_idx.values())

        df = pd.DataFrame(mat, index=class_names, columns=class_names_short)
        # plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(df, fmt=',.0f', annot=True, cmap='Blues', cbar=False)
        heatmap.set(xlabel='predicted label', ylabel='true label', title=log_key + " " + node_name)
        fig = heatmap.get_figure()
        fig.tight_layout()
        fig.show()
        images = [fig]

        # class_columns = data_loader.dataset.classes
        # print(class_columns)
        # TODO jba: global or local step?
        logging.info("Logging confmat as image")
        self.logger.log_image(key=log_key + " -image", images=images, step=self.logger.local_step)
        time.sleep(5)
        logging.info("Logging confmat as table")
        self.logger.log_text(key=log_key + " -table", dataframe=df, step=self.logger.local_step)

        # clear figure such that plots do not become overlapping in wandb
        fig.clf()

        return mat

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        self.logger.log_metrics({"Test/Loss": loss, "Test/Accuracy": metric}, step=self.logger.global_step)
        pass

    def get_num_samples(self):
        return (
            len(self.data.train_dataloader().dataset),
            len(self.data.test_dataloader().dataset),
        )

    def init(self):
        self.close()

    def close(self):
        if self.logger is not None:
            pass

    def finalize_round(self):
        self.logger.global_step = self.logger.global_step + self.logger.local_step
        self.logger.local_step = 0
        pass

    def create_trainer(self):
        logging.info("[Learner] Creating trainer with accelerator: {}".format(self.config.participant["device_args"]["accelerator"]))
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            ),
            leave=True,
        )
        self.__trainer = Trainer(
            callbacks=[RichModelSummary(max_depth=1), progress_bar],
            max_epochs=self.epochs,
            accelerator=self.config.participant["device_args"]["accelerator"],
            devices="auto" if self.config.participant["device_args"]["accelerator"] == "cpu" else "1",  # TODO: only one GPU for now
            # strategy="ddp" if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            # strategy=self.config.participant["device_args"]["strategy"] if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            logger=self.logger,
            log_every_n_steps=20,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True
        )

    def create_trainer_no_logging(self):
        logging.info("[Learner] Creating trainer (logger disabled) with accelerator: {}".format(
            self.config.participant["device_args"]["accelerator"]))
        self.__trainer = Trainer(
            callbacks=[ModelSummary(max_depth=1)],
            max_epochs=self.epochs,
            accelerator=self.config.participant["device_args"]["accelerator"],
            devices=self.config.participant["device_args"]["devices"] if self.config.participant["device_args"][
                                                                             "accelerator"] != "cpu" else None,
            # strategy=self.config.participant["device_args"]["strategy"] if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

    def validate_neighbour(self):
        try:
            if self.epochs > 0:
                self.create_trainer_no_logging()
                results = self.__trainer.validate(self.model, self.data, verbose=False)
                loss = results[0]["Validation/Loss"]
                metric = results[0]["Validation/Accuracy"]
                self.__trainer = None
                return loss, metric
            else:
                return None, None
        except Exception as e:
            logging.error("[NodeLearner.validate_neighbour] Something went wrong with pytorch lightning. {}".format(e))
            return None, None

    def validate_neighbour_no_pl(self, neighbour_model) -> (float, float):
        # the standard PL (pytorch lightning) validation approach seems to break in multithreaded, thus workaround
        avg_loss = 0
        running_loss = 0
        val_dataloader = self.data.val_dataloader()
        with torch.no_grad():
            torch.autograd.set_detect_anomaly(True)
            for i, (inputs, labels) in enumerate(val_dataloader):
                outputs = neighbour_model(inputs)
                # _, predicted_labels = torch.max(outputs, 1)
                loss = F.cross_entropy(outputs, labels)
                running_loss = running_loss + loss.item()
        avg_loss = running_loss / (i + 1)
        # logging.info("Learner.validate_neighbour: computed loss: {}".format(avg_loss))
        val_acc = 0
        return avg_loss, val_acc

    def validate_neighbour_pl(self, neighbour_model) -> (float, float):
        try:
            # performing a deepcopy on the model creates errors with weak dependencies
            tmp_trainer = Trainer(callbacks=[ModelSummary(max_depth=1), TQDMProgressBar(refresh_rate=200)],
                                  max_epochs=self.epochs,
                                  accelerator=self.config.participant["device_args"]["accelerator"],
                                  devices=self.config.participant["device_args"]["devices"] if
                                  self.config.participant["device_args"][
                                      "accelerator"] != "cpu" else None,
                                  logger=self.logger,
                                  log_every_n_steps=20,
                                  enable_checkpointing=False,
                                  enable_model_summary=False,
                                  enable_progress_bar=True)
            results = tmp_trainer.validate(neighbour_model, self.data, verbose=True)
            loss = results[0]["Validation/Loss"]
            metric = results[0]["Validation/Accuracy"]
            return loss, metric
        except Exception as e:
            logging.error("[NodeLearner.validate_neighbour] Something went wrong with pytorch lightning. {}, {}"
                          .format(e, traceback.format_exc()))
            raise e
