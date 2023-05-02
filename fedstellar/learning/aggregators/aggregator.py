# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


import logging
import threading

from fedstellar.role import Role
from fedstellar.utils.observer import Events, Observable


class Aggregator(threading.Thread, Observable):
    """
    Class to manage the aggregation of models. It is a thread so, aggregation will be done in background if all models were added or timeouts have gone.
    Also, it is an observable so, it will notify the node when the aggregation was done.

    Args:
        node_name: (str): String with the name of the node.
    """

    def __init__(self, node_name="unknown", config=None):
        self.node_name = node_name
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        threading.Thread.__init__(self, name="aggregator-" + node_name)
        self.daemon = True
        Observable.__init__(self)
        self.__train_set = []
        self.__waiting_aggregated_model = False
        self.__aggregated_waited_model = False
        self.__stored_models = [] if self.role == Role.PROXY else None
        self.__models = {}
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
            logging.info("[Aggregator] __aggregation_lock.acquire() during {} seconds".format(self.config.participant["AGGREGATION_TIMEOUT"]))
            self.__aggregation_lock.acquire(timeout=self.config.participant["AGGREGATION_TIMEOUT"])
        except Exception as e:
            logging.error("[Aggregator] Error waiting for aggregation: {}".format(e))

        logging.info("[Aggregator] Aggregating models, timeout reached.")

        # Check if node still running (could happen if aggregation thread was a residual thread)
        if not self.__train_set:
            logging.info("[Aggregator] Shutting Down Aggregator Process | __train_set={} --> None or only me --> No aggregation".format(self.__train_set))
            self.notify(
                Events.AGGREGATION_FINISHED_EVENT, None
            )  # To avoid residual training-thread
            return

        # Start aggregation
        n_model_aggregated = sum(
            [len(nodes.split()) for nodes in list(self.__models.keys())]
        )
        if n_model_aggregated != len(self.__train_set):
            logging.info(
                "[Aggregator] __train_set={} || Missing models: {}".format(self.__train_set, set(self.__train_set) - set(self.__models.keys())
                                                                           )
            )
        else:
            logging.info("[Aggregator] Aggregating models.")

        # Notify node
        self.notify(Events.AGGREGATION_FINISHED_EVENT, self.aggregate(self.__models))

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

    def add_model(self, model, nodes, weight):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            nodes: Nodes that collaborated to get the model.
            weight: Number of samples used to get the model.
        """
        logging.info("[Aggregator.add_model] Entry point")
        logging.info("[Aggregator.add_model] Nodes who contributed to the model: {}".format(nodes))
        # if self.__waiting_aggregated_model and self.__stored_models is not None:
        #    self.notify(Events.STORE_MODEL_PARAMETERS_EVENT, model)
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
                    logging.debug("[Aggregator] Starting aggregation thread (run -> timeout) | __train_set={} | __thread_executed={}".format(self.__train_set, self.__thread_executed))
                    self.start()

                # Get a list of nodes added
                logging.info("[Aggregator.add_model] self.__models = {}".format(self.__models.keys()))
                models_added = [n.split() for n in list(self.__models.keys())]
                models_added = [
                    element for sublist in models_added for element in sublist
                ]  # Flatten list
                logging.info("[Aggregator.add_model] Adding model from nodes {} ||||| __train_set = {} | len(models_added) = {}".format(nodes, self.__train_set, len(models_added)))

                # Check if aggregation is needed
                # __train_set tiene a todos mis vecinos (y yo)
                # models_added tiene a todos los vecinos los cuales ya tengo sus parámetros del modelo
                # Agrego
                if len(self.__train_set) > len(models_added):
                    # Check if all nodes are in the train_set
                    # if all([n in self.__train_set for n in nodes]):
                    # Check if all nodes are not aggregated
                    if all([n not in models_added for n in nodes]):
                        # Aggregate model
                        self.__models[" ".join(nodes)] = (model, weight)
                        logging.info(
                            "[Aggregator] Model added ({}/{}) from {}".format(
                                str(len(models_added) + len(nodes)),
                                str(len(self.__train_set)),
                                str(nodes),
                            )
                        )
                        # Remove node from __models if I am in the list
                        logging.info("[Aggregator] Models for aggregation: {}".format(self.__models.keys()))
                        # Check if all models have been added
                        # If all is ok, release the aggregation lock
                        self.check_and_run_aggregation()
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
            else:
                logging.debug("[Aggregator] __waiting_aggregated_model = False,  model received by difusion")
        return None

    def get_partial_aggregation(self, except_nodes):
        """
        Get the partial aggregation of the models.

        Args:
            except_nodes: Nodes to exclude.

        Returns:
            (model, nodes, weight): Model, nodes and number of samples for the partial aggregation.
        """
        logging.info("[Aggregator] Getting partial aggregation from {}, except {}".format(self.__models.keys(), except_nodes))
        dict_aux = {}
        nodes_aggregated = []
        aggregation_weight = 0
        models = self.__models.copy()
        for n, (m, s) in list(models.items()):
            splited_nodes = n.split()
            if all([n not in except_nodes for n in splited_nodes]):
                dict_aux[n] = (m, s)
                nodes_aggregated += splited_nodes
                aggregation_weight += s

        # If there are no models to aggregate
        if len(dict_aux) == 0:
            logging.info("[Aggregator.get_partial_aggregation] No models to aggregate")
            return None, None, None

        return (self.aggregate(dict_aux), nodes_aggregated, aggregation_weight)

    def check_and_run_aggregation(self, force=False):
        """
        Check if all models have been added and start aggregation if so.

        Args:
            force: If true, aggregation will be started even if not all models have been added.
        """
        models_added = [nodes.split() for nodes in list(self.__models.keys())]
        models_added = [
            element for sublist in models_added for element in sublist
        ]  # Flatten list
        # Try Unloock
        try:
            if (
                    force or len(models_added) >= len(self.__train_set)
            ) and self.__train_set != []:
                logging.info("[Aggregator] __aggregation_lock.release() --> __models = {}".format(self.__models.keys()))
                self.__aggregation_lock.release()
        except threading.ThreadError:
            pass

    def clear(self):
        """
        Clear all for a new aggregation.
        """
        observers = self.get_observers()
        self.__init__(node_name=self.node_name, config=self.config)
        for o in observers:
            self.add_observer(o)
