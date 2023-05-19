# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 

import logging
import pickle
import time
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import numpy as np
from itertools import product

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary, TQDMProgressBar
from torchmetrics import ConfusionMatrix

from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.learner import NodeLearner

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
        self.data = data
        self.config = config
        self.logger = logger
        self.__trainer = None
        self.epochs = 1
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

        # FL information
        self.round = 0
        # self.local_step = 0
        # self.global_step = 0

        self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)

    def set_model(self, model):
        self.model = model

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
                results = self.__trainer.test(self.model, self.data, verbose=True)
                loss = results[0]["Test/Loss"]
                metric = results[0]["Test/Accuracy"]
                self.__trainer = None
                self.log_validation_metrics(loss, metric, self.round)
                return loss, metric
            else:
                return None
        except Exception as e:
            logging.error("[NodeLearner.evalaute] Something went wrong with pytorch lightning. {}".format(e))
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

    def compute_confusion_matrix(self):
        model = self.model
        data_loader = self.data.test_dataloader()

        all_targets, all_predictions = [], []
        with torch.no_grad():
            for i, (features, targets) in enumerate(data_loader):
                # features = features.to(device)
                targets = targets
                logits = model(features)
                _, predicted_labels = torch.max(logits, 1)
                all_targets.extend(targets.to('cpu'))
                all_predictions.extend(predicted_labels.to('cpu'))

        t_all_predictions = all_predictions
        t_all_targets = all_targets
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

        # TODO jba: find a way to retrieve all classes, this will not work in non-IID
        df = pd.DataFrame(mat, index=class_labels, columns=class_labels)
        heatmap = sns.heatmap(df, fmt=',.0f', annot=True)
        heatmap.set(xlabel='True Label', ylabel='Predicted Label')
        fig = heatmap.get_figure()
        images = [fig]

        # class_columns = data_loader.dataset.classes
        # print(class_columns)
        # TODO jba: global or local step?
        self.logger.log_image(key="ConfMat", images=images, step=self.logger.global_step)
        self.logger.log_text(key="ConfMat", dataframe=df, step=self.logger.global_step)

        return mat

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

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        self.logger.log_metrics({"Test/Loss": loss, "Test/Accuracy": metric}, step=self.logger.global_step)
        # self.logger.log_metrics({"Test/Loss": loss, "Test/Accuracy": metric}, step=round)
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
        logging.info("[Learner] Creating trainer with accelerator: {}".format(
            self.config.participant["device_args"]["accelerator"]))
        self.__trainer = Trainer(
            callbacks=[ModelSummary(max_depth=1), TQDMProgressBar(refresh_rate=200)],
            max_epochs=self.epochs,
            accelerator=self.config.participant["device_args"]["accelerator"],
            devices=self.config.participant["device_args"]["devices"] if self.config.participant["device_args"][
                                                                             "accelerator"] != "cpu" else None,
            # strategy=self.config.participant["device_args"]["strategy"] if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            logger=self.logger,
            log_every_n_steps=20,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
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
