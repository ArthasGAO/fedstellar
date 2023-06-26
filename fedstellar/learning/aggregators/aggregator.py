# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
import copy
import logging
import threading
from typing import Dict, OrderedDict, List

from fedstellar.learning.modelmetrics import ModelMetrics
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.role import Role
from fedstellar.utils.observer import Events, Observable


class Aggregator(threading.Thread, Observable):
    """
    Class to manage the aggregation of models. It is a thread so, aggregation will be done in background if all models were added or timeouts have gone.
    Also, it is an observable so, it will notify the node when the aggregation was done.

    Args:
        node_name: (str): String with the name of the node.
    """

    def __init__(self, node_name="unknown", config=None, logger=None, learner=None, agg_round=0):
        self.node_name = node_name
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        threading.Thread.__init__(self, name="aggregator-" + node_name)
        self.daemon = True
        self.logger = logger
        self.learner = learner
        self.agg_round = agg_round
        Observable.__init__(self)
        self._models = {}
        self.__train_set = []
        self.__waiting_aggregated_model = False
        self.__aggregated_waited_model = False
        self.__stored_models = [] if self.role == Role.PROXY else None
        self.__lock = threading.Lock()
        self.__aggregation_lock = threading.Lock()
        self.__aggregation_lock.acquire()
        self.__thread_executed = False

    def run(self):
        """
        Wait for the aggregation to be done or timeout. Then, aggregate the models and notify it.
        """
        self.__thread_executed = True

        # Wait for all models to be added or TIMEOUT
        try:
            # TODO sync remove timeout
            """
            logging.info("[Aggregator] __aggregation_lock.acquire() during {} seconds".format(
                self.config.participant["AGGREGATION_TIMEOUT"]))
            self.__aggregation_lock.acquire(timeout=self.config.participant["AGGREGATION_TIMEOUT"])
            """
            self.__aggregation_lock.acquire()
        except Exception as e:
            logging.error("[Aggregator] Error waiting for aggregation: {}".format(e))

        # Check if node still running (could happen if aggregation thread was a residual thread)
        if not self.__train_set:
            logging.info(
                "[Aggregator] Shutting Down Aggregator Process | __train_set={} --> None or only me --> No aggregation".format(
                    self.__train_set))
            self.notify(
                Events.AGGREGATION_FINISHED_EVENT, None
            )  # To avoid residual training-thread
            return

        # Start aggregation
        n_model_aggregated = sum(
            [len(nodes.split()) for nodes in list(self._models.keys())]
        )
        if n_model_aggregated != len(self.__train_set):
            logging.info(
                "[Aggregator] __train_set={} || Missing models: {}".format(self.__train_set,
                                                                           set(self.__train_set) - set(
                                                                               self._models.keys())
                                                                           )
            )
        else:
            logging.info("[Aggregator] Aggregating models.")

        # Notify node
        logging.info("[Aggregator.run] num_aggregated: {}, round {}".format(len(self._models), self.agg_round))
        # self.logger.log_metrics({"num_aggregated": len(self.__models)}, step=self.logger.local_step)
        self.notify(Events.AGGREGATION_FINISHED_EVENT, self.aggregate(self._models))

    def aggregate(self, models):
        """
        Aggregate the models.
        """
        print("Not implemented")

    def set_nodes_to_aggregate(self, listnodes):
        """
        List with the name of nodes to aggregate.

        Args:
            listnodes: List of nodes to aggregate. Empty for no aggregation.
        """
        self.__train_set = listnodes

    def set_waiting_aggregated_model(self):
        """
        Indicates that the node is waiting for an aggregation. It won't participate in aggregation process.
        """
        logging.info("[Aggregator] set_waiting_aggregated_model = True")
        self.__waiting_aggregated_model = True

    def get_waiting_aggregated_model(self):
        """
        Indicates that the node is waiting for an aggregation. It won't participate in aggregation process.
        """
        return self.__waiting_aggregated_model

    def add_model(self, model: OrderedDict, nodes: List[str], metrics: ModelMetrics):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            nodes: Nodes that collaborated to get the model.
            metrics: ModelMetrics
        """
        # logging.info("[Aggregator.add_model] Entry point (round: {})".format(self.agg_round))
        logging.info("[Aggregator.add_model] Nodes who contributed to the model: {}".format(nodes))
        # if self.__waiting_aggregated_model and self.__stored_models is not None:
        #    self.notify(Events.STORE_MODEL_PARAMETERS_EVENT, model)
        # logging.info("Aggregator: nodes {} current metrics: {}".format(nodes, metrics))
        if self.__waiting_aggregated_model:
            logging.info("[Aggregator] Received an aggregated model from {} --> Overwriting local model".format(nodes))
            # Check if a node aggregator is in the list of nodes
            # if any([n.startswith("aggregator") for n in nodes.split()]):
            self.notify(Events.AGGREGATION_FINISHED_EVENT, model)
        else:
            if nodes is not None:
                self.__lock.acquire()

                # Start aggregation timeout
                if self.__train_set != [] and not self.__thread_executed:
                    logging.debug(
                        "[Aggregator] Starting aggregation thread (run -> timeout) | __train_set={} | __thread_executed={}".format(
                            self.__train_set, self.__thread_executed))
                    self.start()

                # Get a list of nodes added
                logging.info("[Aggregator.add_model] self.__models = {}".format(self._models.keys()))
                models_added = [n.split() for n in list(self._models.keys())]
                models_added = [
                    element for sublist in models_added for element in sublist
                ]  # Flatten list
                logging.info(
                    "[Aggregator.add_model] Adding model from nodes {} ||||| __train_set = {} | len(models_added) = {}".format(
                        nodes, self.__train_set, len(models_added)))

                # Check if aggregation is needed
                # __train_set has all my neighbors (and me)
                # models_added has all the neighbors which I already have their model parameters added
                if len(self.__train_set) > len(models_added):
                    # Check if all nodes are in the train_set
                    # if all([n in self.__train_set for n in nodes]):
                    # Check if all nodes are not aggregated
                    if all([n not in models_added for n in nodes]):
                        # Aggregate model
                        self._models[" ".join(nodes)] = (model, metrics)
                        logging.info(
                            "[Aggregator] Model added ({}/{}) from {}".format(
                                str(len(models_added) + len(nodes)),
                                str(len(self.__train_set)),
                                str(nodes),
                            )
                        )
                        # Remove node from __models if I am in the list
                        logging.info("[Aggregator] Models for aggregation: {}".format(self._models.keys()))
                        # Check if all models have been added
                        # If all is ok, release the aggregation lock
                        self.check_and_run_aggregation(force=False)
                        # Build response
                        response = models_added + nodes
                        # Unloock
                        self.__lock.release()

                        return response
                    else:
                        self.__lock.release()
                        logging.debug(
                            "[Aggregator] Can't add a model that has already been added {}".format(nodes)
                        )
                else:
                    self.__lock.release()
                    logging.debug("[Aggregator] Need not needed,  releasing lock")
            else:
                logging.debug("[Aggregator] __waiting_aggregated_model = False,  model received by diffusion")
        return None

    def broadcast_local_model(self):
        logging.info("[Aggregator.broadcast_local_model]. Partial aggregation: only local model")
        for node, (model, metrics) in list(self._models.items()):
            if node == self.node_name:
                return model, [node], ModelMetrics(
                    num_samples=metrics.num_samples)
        return None

    def get_full_aggregation(self):
        logging.info(
            "[Aggregator] Getting full aggregation from {}".format(self._models.keys()))
        dict_aux = {}
        nodes_aggregated = []
        total_samples = 0

        node: str
        model: OrderedDict
        metrics: ModelMetrics
        for node, (model, metrics) in list(self._models.items()):
            split_nodes = node.split()
            dict_aux[node] = (model, metrics)
            nodes_aggregated += split_nodes
            total_samples += metrics.num_samples

        # If there are no models to aggregate
        if len(dict_aux) == 0:
            logging.info("[Aggregator.get_full_aggregation] No models to aggregate")
            return None, None, None

        aggregated_model = self.aggregate(dict_aux)
        logging.info("[Aggregator.get_full_aggregation] num_aggregated: {}, round {}".format(len(self._models),
                                                                                             self.agg_round))
        self.agg_round += 1
        self.logger.log_metrics({"num_aggregated": len(self._models)}, step=self.logger.global_step)

        # Only use to compare models in terms of metrics, isolates logging on PseudoAggregation
        # nodes_aggregated = [self.node_name]

        return aggregated_model, nodes_aggregated, ModelMetrics(num_samples=total_samples)

    def get_partial_aggregation(self, except_nodes):
        """
        Get the partial aggregation of the models.

        Args:
            except_nodes: Nodes to exclude.

        Returns:
            (model, nodes, weights): Model, nodes and number of samples for the partial aggregation.
        """

        """
        logging.info("[Aggregator] Waiting partial aggregation")
        do_aggregate = False
        while not do_aggregate:
            models_added = [nodes.split() for nodes in list(self.__models.keys())]
            models_added = [
                element for sublist in models_added for element in sublist
            ]
            logging.info("[Aggregator] Spinning partial aggregation")
            do_aggregate = len(models_added) >= len(self.__train_set)
        """

        # self.check_and_run_aggregation(force=False)

        logging.info(
            "[Aggregator] Getting partial aggregation from {}, except {}".format(self._models.keys(), except_nodes))
        dict_aux = {}
        nodes_aggregated = []
        total_samples = 0

        node: str
        model: OrderedDict
        metrics: ModelMetrics
        for node, (model, metrics) in list(self._models.items()):
            split_nodes = node.split()
            if all([node not in except_nodes for node in split_nodes]):
                dict_aux[node] = (model, metrics)
                nodes_aggregated += split_nodes
                total_samples += metrics.num_samples

        # If there are no models to aggregate
        if len(dict_aux) == 0:
            logging.info("[Aggregator.get_partial_aggregation] No models to aggregate")
            return None, None, None

        aggregated_model = self.aggregate(dict_aux)
        logging.info("[Aggregator.get_partial_aggregation] num_aggregated: {}, round {}".format(len(self._models),
                                                                                                self.agg_round))
        self.agg_round += 1
        self.logger.log_metrics({"num_aggregated": len(self._models)}, step=self.logger.global_step)

        # Only use to compare models in terms of metrics, isolates logging on PseudoAggregation
        # nodes_aggregated = [self.node_name]

        return aggregated_model, nodes_aggregated, ModelMetrics(num_samples=total_samples)

    def check_and_run_aggregation(self, force=False):
        """
        Check if all models have been added and start aggregation if so.

        Args:
            force: If true, aggregation will be started even if not all models have been added.
        """
        models_added = [nodes.split() for nodes in list(self._models.keys())]
        models_added = [
            element for sublist in models_added for element in sublist
        ]  # Flatten list
        # Try Unloock
        logging.info("[Aggregator.check_and_run_aggregation] conditions: models_added: {}, __train_set: {}"
                     .format(models_added, self.__train_set))
        try:
            if (
                    force or len(models_added) >= len(self.__train_set)
            ) and self.__train_set != []:
                logging.info("[Aggregator] __aggregation_lock.release() --> __models = {}".format(self._models.keys()))
                self.__aggregation_lock.release()
        except threading.ThreadError as e:
            logging.error("[Aggregator.check_and_run_aggregation] Error releasing aggregation lock")
            pass

    def clear(self):
        """
        Clear all for a new aggregation.
        """
        observers = self.get_observers()
        next_round = self.agg_round + 1
        self.__init__(node_name=self.node_name,
                      config=self.config,
                      logger=self.logger,
                      learner=self.learner,
                      agg_round=next_round)
        for o in observers:
            self.add_observer(o)
