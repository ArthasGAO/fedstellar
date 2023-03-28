# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 

import logging
import pickle
import time
from collections import OrderedDict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary, TQDMProgressBar

from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.learner import NodeLearner


###########################
#    LightningLearner     #
###########################


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

    def encode_parameters(self, params=None, contributors=None, weight=None):
        if params is None:
            params = self.model.state_dict()
        array = [val.cpu().numpy() for _, val in params.items()]
        return pickle.dumps((array, contributors, weight))

    def decode_parameters(self, data):
        try:
            params, contributors, weight = pickle.loads(data)
            params_dict = zip(self.model.state_dict().keys(), params)
            return (
                OrderedDict({k: torch.tensor(v) for k, v in params_dict}),
                contributors,
                weight,
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
            logging.error("Something went wrong with pytorch lightning. {}".format(e))

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
            logging.error("Something went wrong with pytorch lightning. {}".format(e))
            return None

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
        self.__trainer = Trainer(
            callbacks=[ModelSummary(max_depth=1), TQDMProgressBar(refresh_rate=200)],
            max_epochs=self.epochs,
            accelerator=self.config.participant["device_args"]["accelerator"],
            devices=self.config.participant["device_args"]["devices"] if self.config.participant["device_args"]["accelerator"] != "cpu" else None,
            # strategy=self.config.participant["device_args"]["strategy"] if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            logger=self.logger,
            log_every_n_steps=20,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
        )
