# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#
import json
import logging
import math
import os
from datetime import datetime, timedelta

from fedstellar.learning.pytorch.remotelogger import FedstellarWBLogger
from fedstellar.learning.pytorch.statisticslogger import FedstellarLogger

os.environ['WANDB_SILENT'] = 'true'

# Import the requests module
import requests

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

import random
import threading
import time

from pytorch_lightning.loggers import WandbLogger, CSVLogger

from fedstellar.base_node import BaseNode
from fedstellar.communication_protocol import CommunicationProtocol
from fedstellar.config.config import Config
from fedstellar.learning.aggregators.fedavg import FedAvg
from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.role import Role
from fedstellar.utils.observer import Events, Observer


class Node(BaseNode):
    """
    Class based on a base node that allows **FEDERATED LEARNING**.

    Metrics will be saved under a folder with the name of the node.

    Args:
        model: Model to be learned. Careful, model should be compatible with data and the learner.
        data: Dataset to be used in the learning process. Careful, model should be compatible with data and the learner.
        host (str): Host where the node will be listening.
        port (int): Port where the node will be listening.
        learner (NodeLearner): Learner to be used in the learning process. Default: LightningLearner.
        simulation (bool): If True, the node will be simulated. Default: True.
        encrypt (bool): If True, node will encrypt the communications. Default: False.

    Attributes:
        round (int): Round of the learning process.
        totalrounds (int): Total number of rounds of the learning process.
        learner (NodeLearner): Learner to be used in the learning process.
        aggregator (Aggregator): Aggregator to be used in the learning process.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(
            self,
            idx,
            experiment_name,
            model,
            data,
            hostdemo=None,
            host="127.0.0.1",
            port=None,
            config=Config,
            learner=LightningLearner,
            encrypt=False,
    ):
        # Super init
        BaseNode.__init__(self, experiment_name, hostdemo, host, port, encrypt, config)
        Observer.__init__(self)

        self.idx = idx
        logging.debug("[NODE] My idx is {}".format(self.idx))

        # Import configuration file
        self.config = config
        # Report the configuration to the controller (first instance)
        self.__report_status_to_controller()

        # Learning
        self.round = None
        self.totalrounds = None
        self.__model_initialized = False
        self.__initial_neighbors = []
        self.__start_thread_lock = threading.Lock()

        # Learner and learner logger
        # log_model="all" to log model
        # mode="disabled" to disable wandb
        if self.config.participant['tracking_args']['enable_remote_tracking']:
            logging.info("[NODE] Tracking W&B enabled")
            logging.getLogger("wandb").setLevel(logging.ERROR)
            if self.hostdemo:
                wandblogger = FedstellarWBLogger(project="framework-enrique", group=self.experiment_name, name=f"participant_{self.idx}")
            else:
                wandblogger = FedstellarWBLogger(project="framework-enrique", group=self.experiment_name, name=f"participant_{self.idx}")
            wandblogger.watch(model, log="all")
            self.learner = learner(model, data, config=self.config, logger=wandblogger)
        else:
            if self.config.participant['tracking_args']['local_tracking'] == 'csv':
                logging.info("[NODE] Tracking CSV enabled")
                csvlogger = CSVLogger(f"{self.log_dir}", name="metrics", version=f"participant_{self.idx}")
                self.learner = learner(model, data, config=self.config, logger=csvlogger)
            elif self.config.participant['tracking_args']['local_tracking'] == 'web':
                logging.info("[NODE] Tracking Web enabled")
                tensorboardlogger = FedstellarLogger(f"{self.log_dir}", name="metrics", version=f"participant_{self.idx}", log_graph=True)
                self.learner = learner(model, data, config=self.config, logger=tensorboardlogger)

        logging.info("[NODE] Role: " + str(self.config.participant["device_args"]["role"]))

        # Aggregator
        if self.config.participant["aggregator_args"]["algorithm"] == "FedAvg":
            self.aggregator = FedAvg(node_name=self.get_name(), config=self.config)

        self.aggregator.add_observer(self)

        self.shared_metrics = False

        # Store the parameters of the model
        self.__stored_model_parameters = []
        self.__timeout = datetime.now()

        # Train Set Votes
        self.__train_set = []
        self.__train_set_votes = {}
        self.__train_set_votes_lock = threading.Lock()

        # Locks
        self.__wait_votes_ready_lock = threading.Lock()
        self.__finish_aggregation_lock = threading.Lock()
        self.__finish_aggregation_lock.acquire()
        self.__wait_init_model_lock = threading.Lock()
        self.__wait_init_model_lock.acquire()
        # Grace period to wait for last transmission using Aggregator thread
        self.__wait_finish_experiment_lock = threading.Lock()
        self.__wait_finish_experiment_lock.acquire()

    #########################
    #    Node Management    #
    #########################

    def connect_to(self, h, p, full=False, force=False):
        """
        Connects a node to another. If learning is running, connections are not allowed (it should be forced).
        Careful, if connection is forced with a new node, it will produce timeouts in the network.

        Args:
            h (str): The host of the node.
            p (int): The port of the node.
            full (bool): If True, the node will be connected to the entire network.
            force (bool): If True, the node will be connected even though it should not be.

        Returns:
            node: The node that has been connected to.
        """
        # Check if learning is running
        if self.round is not None and not force:
            logging.info(
                "[NODE] Cant connect to other nodes when learning is running."
            )
            return None

        # Connect
        return super().connect_to(h, p, full, force)

    def stop(self):
        """
        Stop the node and the learning if it is running.
        """
        if self.round is not None:
            self.__stop_learning()
        self.learner.close()
        super().stop()

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data):
        """
        Set the data to be used in the learning process (learner).

        Args:
            data: Dataset to be used in the learning process.
        """
        self.learner.set_data(data)

    def set_model(self, model):
        """
        Set the model to use.
        Carefully, model, not only weights.

        Args:
            model: Model to be learned.
        """
        self.learner.set_model(model)

    ###############################################
    #         Network Learning Management         #
    ###############################################

    def set_start_learning(self, rounds=1, epochs=1):
        """
        Start the learning process in the entire network.
        If the node is not the "starting" node, it will wait for the starting node to start the learning process.

        NOTE:
        In first instance, this functionality only is used in one node of the network (defined by the controller).
        After that, the other participants can start this functionality when they receive Events.START_LEARNING_EVENT

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        if self._terminate_flag.is_set():
            logging.info(
                "[NODE] Node must be running to start learning"
            )
            return
        if self.round is None:
            # Start Learning
            logging.info("[NODE] I am the initializer node... | Broadcasting START_LEARNING | Rounds: {} | Epochs: {}".format(rounds, epochs))
            self.broadcast(
                CommunicationProtocol.build_start_learning_msg(rounds, epochs)
            )
            # Initialize model
            self.broadcast(CommunicationProtocol.build_model_initialized_msg())
            self.__wait_init_model_lock.release()
            self.__model_initialized = (
                True  # esto seguramente sobre, con locks es suficiente
            )
            # Learning Thread
            self.__start_learning_thread(rounds, epochs)
        else:
            logging.info("[NODE] Learning already started")

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.__stop_learning()
        else:
            logging.info("[NODE] Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self, rounds, epochs):
        learning_thread = threading.Thread(
            target=self.__start_learning, args=(rounds, epochs)
        )
        learning_thread.name = "learning_thread-" + self.get_name()
        learning_thread.daemon = True
        learning_thread.start()

    def __start_learning(self, rounds, epochs):
        """
        Start the learning process in the local node.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        self.__start_thread_lock.acquire()  # Used to avoid create duplicated training threads
        if self.round is None:
            self.round = 0
            self.totalrounds = rounds
            self.learner.init()
            self.__start_thread_lock.release()

            begin = time.time()

            # Send the model parameters (initial model) to neighbors
            self.__gossip_model_difusion(initialization=True)

            # Wait to guarantee new connection heartbeats convergence and fix neighbors
            wait_time = self.config.participant["WAIT_HEARTBEATS_CONVERGENCE"] - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)
            # TODO: Check this parameter
            self.__initial_neighbors = (
                self.get_neighbors()
            )  # used to restore the original list of neighbors after the learning round

            logging.info("[NODE.__start_learning] Learning started in node {} -> Round: {} | Epochs: {}".format(self.get_name(), self.round, epochs))
            self.learner.set_epochs(epochs)
            self.learner.create_trainer()
            self.__train_step()
            logging.info("[NODE.__start_learning] Thread __start_learning finished in node {}".format(self.get_name()))

    def __stop_learning(self):
        """
        Stop the learning process in the local node. Interrupts learning process if it's running.
        """
        logging.info("[NODE] Stopping learning")
        # Rounds
        self.round = None
        self.totalrounds = None
        # Leraner
        self.learner.interrupt_fit()
        # Aggregator
        self.aggregator.check_and_run_aggregation(force=True)
        self.aggregator.set_nodes_to_aggregate([])
        self.aggregator.clear()
        # Try to free wait locks
        try:
            self.__wait_votes_ready_lock.release()
        except threading.ThreadError:
            pass

    ####################################
    #         Model Aggregation         #
    ####################################

    def add_model(self, m):
        """
        Add a model. If the model isn't inicializated, the recieved model is used for it. Otherwise, the model is aggregated using the **aggregator**.

        Args:
            m: Encoded model. Contains model and their contributors
        """
        # Check if Learning is running
        if self.round is not None:
            try:
                if self.__model_initialized:
                    # Add model to aggregator
                    (
                        decoded_model,
                        contributors,
                        weight,
                    ) = self.learner.decode_parameters(m)
                    logging.info("[NODE.add_model] Model received from {} --using--> {} in the other node | Now I add the model using self.aggregator.add_model()".format(contributors, '__gossip_model_diffusion' if contributors is None and weight is None else '__gossip_model_aggregation'))
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.aggregator.add_model(
                            decoded_model, contributors, weight
                        )
                        if models_added is not None:
                            logging.info("[NODE.add_model] self.broadcast with MODELS_AGGREGATED = {}".format(models_added))
                            # TODO: Fix bug at MacBook. When CPU is high, only new nodes will be sent.
                            self.broadcast(
                                CommunicationProtocol.build_models_aggregated_msg(
                                    models_added
                                )
                            )
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    self.__model_initialized = True
                    logging.info("[NODE] Initialization Model Weights")
                    # Initialize model
                    model, _, _ = self.learner.decode_parameters(m)
                    self.learner.set_parameters(model)
                    self.__wait_init_model_lock.release()
                    self.broadcast(CommunicationProtocol.build_model_initialized_msg())

            except DecodingParamsError as e:
                logging.error("[NODE] Error decoding parameters: " + str(e))
                self.stop()

            except ModelNotMatchingError as e:
                logging.error("[NODE] Models not matching: " + str(e))
                self.stop()

            except Exception as e:
                logging.error("[NODE] Error adding model: " + str(e))
                self.stop()
                raise e
        else:
            logging.error(
                "[NODE] Tried to add a model while learning is not running"
            )

    #######################
    #    Training Steps    #
    #######################

    def __train_step(self):
        """
        Train the model in the local node.
        If the node is in the __train_set list, the training is performed. Otherwise, the node waits for the training to be performed by another node.
        Returns:

        """

        # Set train set
        if self.round is not None:
            # self.__train_set = self.__vote_train_set()
            for n in self.get_neighbors():
                if n.get_name() not in self.__train_set:
                    self.__train_set.append(n.get_name())
            self.__train_set.append(self.get_name()) if self.get_name() not in self.__train_set else None
            # if self.config.participant["device_args"]["role"] != Role.TRAINER:
            #     self.__train_set.append(self.get_name()) if self.get_name() not in self.__train_set else None
            # else:
            #     self.__train_set.remove(self.get_name()) if self.get_name() in self.__train_set else None
            logging.info("[NODE.__train_step] __train_set = {}".format(self.__train_set))
            self.__validate_train_set()

        # TODO: Improve in the future
        # is_train_set = self.get_name() in self.__train_set
        is_train_set = True
        if is_train_set and (self.config.participant["device_args"]["role"] == Role.AGGREGATOR or self.config.participant["device_args"]["role"] == Role.SERVER):
            logging.info("[NODE.__train_step] Role.AGGREGATOR/Role.SERVER process...")
            # Full connect train set
            if self.round is not None:
                self.__connect_and_set_aggregator()

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None and self.config.participant["device_args"]["role"] != Role.SERVER:  # El participante servidor no entrena (en CFL)
                self.__train()

            # Aggregate Model
            if self.round is not None:
                logging.info("[NODE.__train_step] self.aggregator.add_model with MY MODEL")
                self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.get_name()],
                    self.learner.get_num_samples()[0],
                )
                logging.info("[NODE.__train_step] self.broadcast with MODELS_AGGREGATED = MY_NAME")
                self.broadcast(
                    CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
                )
                if self.config.participant["device_args"]["role"] == Role.SERVER:
                    self.__gossip_model_difusion()
                else:
                    self.__gossip_model_aggregation()

        elif self.config.participant["device_args"]["role"] == Role.TRAINER:
            logging.info("[NODE.__train_step] Role.TRAINER process...")
            logging.info("[NODE.__train_step] __waiting_aggregated_model = {}".format(self.aggregator.get_waiting_aggregated_model()))
            if self.round is not None:
                self.__connect_and_set_aggregator()

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()

            # Aggregate Model
            if self.round is not None:
                logging.info("[NODE.__train_step] self.aggregator.add_model with MY MODEL")
                # Node has to aggregate its own model before sending it to the aggregator
                self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.get_name()],
                    self.learner.get_num_samples()[0],
                )

                logging.info("[NODE.__train_step] self.broadcast with MODELS_AGGREGATED = MY_NAME")
                self.broadcast(
                    CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
                )

                self.__gossip_model_aggregation()

                self.aggregator.set_waiting_aggregated_model()
                time.sleep(5)

        elif self.config.participant["device_args"]["role"] == Role.PROXY:
            # If the node is a proxy, it stores the parameters received from the neighbors.
            # When the node reaches a timeout or the number of parameters received is equal to a specific number, it will send the parameters to all the neighbors.
            logging.info("[NODE.__train_step] Role.PROXY process...")

            # Aggregate Model
            if self.round is not None:
                # self.aggregator.add_model(
                #    self.learner.get_parameters(),
                #    [self.get_name()],
                #    self.learner.get_num_samples()[0],
                # )

                self.broadcast(
                    CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
                )
                # Timeout to send the parameters to the neighbors?
                if datetime.now() > self.__timeout:
                    logging.info("[NODE.__train_step (PROXY)] Timeout reached. Sending parameters to neighbors...")
                    self.__gossip_model_aggregation()
                    self.__timeout = datetime.now() + timedelta(seconds=10)
                # The proxy node sets the waiting aggregated model flag to True
                # In this case, the proxy waits for params and add them to the local storage
                self.aggregator.set_waiting_aggregated_model()

        elif self.config.participant["device_args"]["role"] == Role.IDLE:
            # Role.IDLE functionality

            # Set Models To Aggregate
            # Node won't participate in aggregation process.
            # __waiting_aggregated_model = True
            # Then, when the node receives a PARAMS_RECEIVED_EVENT, it will run add_model, and it set parameters to the model
            self.aggregator.set_waiting_aggregated_model()

        else:
            logging.warning("[NODE.__train_step] Role not implemented yet")

        # Gossip aggregated model
        #if self.round is not None:
        #    self.__gossip_model_difusion()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()

    def __validate_train_set(self):
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in self.__train_set:
            if tsn not in self.get_network_nodes():
                if tsn != self.get_name():
                    self.__train_set.remove(tsn)

        logging.info(
            "[NODE.__validate_train_set] Train set of {} nodes: {}".format(
                len(self.__train_set), self.__train_set
            )
        )

    ##########################
    #    Connect Trainset    #
    ##########################

    def __connect_and_set_aggregator(self):
        # Set Models To Aggregate
        self.aggregator.set_nodes_to_aggregate(self.__train_set)
        logging.info("[NODE.__connect_and_set_aggregator] Aggregator set to: {}".format(self.__train_set))
        for node in self.__train_set:
            if node != self.get_name():
                h, p = node.split(":")
                if p.isdigit():
                    nc = self.get_neighbor(h, int(p))
                    # If the node is not connected, connect it (to avoid duplicated connections only a node connects to the other)
                    if nc is None and self.get_name() > node:
                        self.connect_to(h, int(p), force=True)
                else:
                    logging.info(
                        "[NODE] Node {} has an invalid port".format(
                            node.split(":")
                        )
                    )

        # Wait connections
        count = 0
        begin = time.time()
        while True:
            count = count + (time.time() - begin)
            if count > self.config.participant["TRAIN_SET_CONNECT_TIMEOUT"]:
                logging.info("[NODE] Timeout for train set connections.")
                break
            if (len(self.__train_set) == len(
                    [
                        nc
                        for nc in self.get_neighbors()
                        if nc.get_name() in self.__train_set
                    ]
            ) + 1
            ):
                break
            time.sleep(0.1)

    ############################
    #    Train and Evaluate    #
    ############################

    def __train(self):
        logging.info("[NODE.__train] Start training...")
        self.learner.fit()
        logging.info("[NODE.__train] Finish training...")

    def __evaluate(self):
        logging.info("[NODE.__evaluate] Start evaluation...")
        results = self.learner.evaluate()
        if results is not None:
            logging.info(
                "[NODE] Evaluated. Losss: {}, Metric: {}".format(
                    results[0], results[1]
                )
            )
            logging.info("[NODE.__evaluate] Finish evaluation...")

            if self.shared_metrics:
                logging.info(
                    "[NODE] Broadcasting metrics.".format(
                        len(self.get_neighbors())
                    )
                )
                encoded_msgs = CommunicationProtocol.build_metrics_msg(
                    self.get_name(), self.round, results[0], results[1]
                )
                self.broadcast(encoded_msgs)

    ######################
    #    Round finish    #
    ######################

    def __on_round_finished(self):
        # Remove trainset connections
        for nc in self.get_neighbors():
            if nc not in self.__initial_neighbors:
                self.rm_neighbor(nc)
        # Set Next Round
        self.aggregator.clear()
        logging.info("[NODE] Finalizing round: {}".format(self.round))
        self.learner.finalize_round()  # TODO: Fix to improve functionality
        self.round = self.round + 1
        self.learner.logger.log_metrics({"Round": self.round}, step=self.learner.logger.global_step)
        logging.info("[LightningLearner] Starting round: {}".format(self.round))
        # Clear node aggregation
        for nc in self.get_neighbors():
            nc.clear_models_aggregated()

        # If the federation is SDFL and the node is the aggregator, the node can transfer the aggregation role to another node
        if self.config.participant['scenario_args']["federation"] == "SDFL" and self.config.participant["device_args"]["role"] == "aggregator":
            self.__transfer_aggregator_role(schema="random")

        # Next Step or Finish
        logging.info(
            "[NODE] Round {} of {} finished.".format(
                self.round, self.totalrounds
            )
        )
        if self.round < self.totalrounds:
            self.__train_step()
        else:
            logging.debug("[NODE] FL finished | Models aggregated = {}".format([nc.get_models_aggregated() for nc in self.get_neighbors()]))
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            logging.info(
                "[NODE] FL experiment finished | Round: {} | Total rounds: {} | [!] Both to None".format(
                    self.round, self.totalrounds
                )
            )
            self.round = None
            self.totalrounds = None
            self.__model_initialized = False
            # logging.info("[NODE] FL experiment finished | __stop_learning()")
            # self.__stop_learning()  # TODO: 20-12-22 | This is a temporal fix to avoid the node to continue training after the FL experiment is finished

    def __transfer_aggregator_role(self, schema):
        if schema == "random":
            logging.info("[NODE.__transfer_aggregator_role] Transferring aggregator role using schema {}".format(schema))
            # Random
            nc = random.choice(self.get_neighbors())
            msg = CommunicationProtocol.build_transfer_leadership_msg()
            nc.send(msg)
            self.config.participant['device_args']["role"] = "trainer"
            logging.info("[NODE.__transfer_aggregator_role] Aggregator role transfered to {}.".format(nc.get_name()))
        else:
            logging.info("[NODE.__transfer_aggregator_role] Schema {} not found.".format(schema))

    #########################
    #    Model Gossiping    #
    #########################

    def __gossip_model_aggregation(self):
        logging.info("[NODE.__gossip_model_aggregation] Gossiping...")
        # Anonymous functions
        candidate_condition = lambda nc: nc.get_name() in self.__train_set and len(nc.get_models_aggregated()) < len(self.__train_set)
        status_function = lambda nc: (nc.get_name(), len(nc.get_models_aggregated()))
        model_function = lambda nc: self.aggregator.get_partial_aggregation(nc.get_models_aggregated())

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model_difusion(self, initialization=False):
        logging.info("[NODE.__gossip_model_difusion] Gossiping...")
        # Send model parameters using gossiping
        # Wait a model (init or aggregated)
        if initialization:
            self.__wait_init_model_lock.acquire()
            logging.info("[NODE.__gossip_model_difusion] Initialization=True")
            candidate_condition = lambda nc: not nc.get_model_initialized()
        else:
            self.__finish_aggregation_lock.acquire()
            logging.info("[NODE.__gossip_model_difusion] Initialization=False")
            candidate_condition = lambda nc: nc.get_model_ready_status() < self.round

        # Anonymous functions
        status_function = lambda nc: nc.get_name()
        model_function = lambda _: (
            self.learner.get_parameters(),
            None,
            None,
        )  # At diffusion, contributors are not relevant

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model(self, candidate_condition, status_function, model_function):
        logging.debug("[NODE.__gossip_model] Traceback", stack_info=True)
        # Initialize list with status of nodes in the last X iterations
        last_x_status = []
        j = 0

        while True:
            # Get time to calculate frequency
            begin = time.time()

            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info(
                    "[NODE] Stopping model gossip process.")
                return

            # Get nodes which need models
            logging.info("---------------------Feedback about neighbors---------------------")
            logging.info("[NODE.__gossip_model] Neighbors: {}".format(self.get_neighbors()))
            for nc in self.get_neighbors():
                logging.info("---------------------Feedback about neighbor {}---------------------".format(nc))
                logging.info("[NODE.__gossip_model] Neighbor: {} | My __train_set: {} | Nc.modelsaggregated: {}".format(nc, self.__train_set, nc.get_models_aggregated()))
                logging.info("[NODE.__gossip_model] Neighbor: {} | Candidate_condition return: {}".format(nc, candidate_condition(nc)))
                logging.info("[NODE.__gossip_model] Neighbor: {} | Status_function return: {}".format(nc, status_function(nc)))
                logging.info("---------------------End of feedback about neighbor {}---------------------".format(nc))
            logging.info("------------------------------------------------------------------")

            nei = [nc for nc in self.get_neighbors() if candidate_condition(nc)]
            logging.info("[NODE.__gossip_model] Selected (to exclude) based on condition: {}".format(nei))

            # Determine end of gossip
            if not nei:
                logging.info("[NODE] Gossip finished.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            if len(last_x_status) != self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]:
                last_x_status.append([status_function(nc) for nc in nei])
            else:
                last_x_status[j] = str([status_function(nc) for nc in nei])
                j = (j + 1) % self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]

                # Check if las messages are the same
                for i in range(len(last_x_status) - 1):
                    if last_x_status[i] != last_x_status[i + 1]:
                        break
                    logging.info(
                        "[NODE] Gossiping exited for {} equal rounds.".format(
                            self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]
                        )
                    )
                    return

            # Select a random subset of neighbors
            samples = min(self.config.participant["GOSSIP_MODELS_PER_ROUND"], len(nei))
            nei = random.sample(nei, samples)
            logging.info("[NODE.__gossip_model] Selected a random subset of neighbors (to exclude): {}".format(nei))

            # Generate and Send Model Partial Aggregations (model, node_contributors)
            for nc in nei:
                model, contributors, weights = model_function(nc)
                # Send Partial Aggregation
                if model is not None:
                    logging.info(
                        "[NODE] Gossiping model to {}.".format(
                            nc.get_name()
                        )
                    )
                    encoded_model = self.learner.encode_parameters(
                        params=model, contributors=contributors, weight=weights
                    )
                    logging.info("[NODE.__gossip_model] Building params message | Contributors: {}".format(contributors))
                    encoded_msgs = CommunicationProtocol.build_params_msg(encoded_model, self.config.participant["BLOCK_SIZE"])
                    logging.info("[NODE.__gossip_model] Sending params message to {}".format(nc))
                    # Send Fragments
                    for msg in encoded_msgs:
                        nc.send(msg)
                else:
                    logging.info("[NODE.__gossip_model] Model returned by model_function is None")
            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1 / self.config.participant["GOSSIP_MODELS_FREC"] - time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)

    ###########################
    #     Observer Events     #
    ###########################

    def update(self, event, obj):
        """
        Observer update method. Used to handle events that can occur in the different components and connections of the node.

        Args:
            event (Events): Event that has occurred.
            obj: Object that has been updated.
        """
        if len(str(obj)) > 300:
            logging.debug("[NODE.update (observer)] Event that has occurred: {} | Obj information: Too long [...]".format(event))
        else:
            logging.debug("[NODE.update (observer)] Event that has occurred: {} | Obj information: {}".format(event, obj))

        if event == Events.NODE_CONNECTED_EVENT:
            n, force = obj
            if self.round is not None and not force:
                logging.info(
                    "[NODE] Cant connect to other nodes when learning is running. (however, other nodes can be connected to the node.)"
                )
                n.stop()
                return

        elif event == Events.SEND_ROLE_EVENT:
            self.broadcast(CommunicationProtocol.build_role_msg(self.get_name(), self.config.participant["device_args"]["role"]))

        elif event == Events.ROLE_RECEIVED_EVENT:
            # Update the heartbeater with the role node
            # obj = (node_name, role)
            self.heartbeater.add_node_role(obj[0], obj[1])

        elif event == Events.AGGREGATION_FINISHED_EVENT:
            # Set parameters and communate it to the training process
            if obj is not None:
                logging.info("[NODE.update] Override the local model with obj received")
                self.learner.set_parameters(obj)
                # Share that aggregation is done
                self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
            else:
                logging.error(
                    "[NODE] Aggregation finished with no parameters"
                )
                # TODO: Testing 20-12-2022 (remove stop and add broadcast)
                # self.stop()
                # self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
            try:
                self.__finish_aggregation_lock.release()
                logging.info("[NODE.__finish_aggregation_lock] __finish_aggregation_lock.release()")
            except threading.ThreadError:
                pass

        elif event == Events.START_LEARNING_EVENT:
            self.__start_learning_thread(obj[0], obj[1])

        elif event == Events.STOP_LEARNING_EVENT:
            self.__stop_learning()

        elif event == Events.PARAMS_RECEIVED_EVENT:
            logging.info("[NODE] Params received")
            self.add_model(obj)

        elif event == Events.METRICS_RECEIVED_EVENT:
            name, round, loss, metric = obj
            self.learner.log_validation_metrics(loss, metric, round=round, name=name)

        elif event == Events.TRAIN_SET_VOTE_RECEIVED_EVENT:
            node, votes = obj
            self.__train_set_votes_lock.acquire()
            self.__train_set_votes[node] = votes
            self.__train_set_votes_lock.release()
            # Communicate to the training process that a vote has been received
            try:
                self.__wait_votes_ready_lock.release()
            except threading.ThreadError:
                pass

        elif event == Events.REPORT_STATUS_TO_CONTROLLER_EVENT:
            self.__report_status_to_controller()
            # self.__report_logs_to_controller()
            self.__report_resources()

        elif event == Events.STORE_MODEL_PARAMETERS_EVENT:
            if obj is not None:
                logging.info("[NODE.update] Store the model parameters received")
                self.__store_model_parameters(obj)
                # Share that aggregation is done
                self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
            else:
                logging.error(
                    "[NODE] Error storing the model parameters"
                )
                self.stop()
            try:
                self.__finish_aggregation_lock.release()
                logging.info("[NODE.__finish_aggregation_lock] __finish_aggregation_lock.release()")
            except threading.ThreadError:
                pass

        # Execute BaseNode update
        super().update(event, obj)

    def __report_status_to_controller(self):
        """
        Report the status of the node to the controller.
        The configuration is the one that the node has at the memory.

        Returns:

        """
        # Set the URL for the POST request
        url = f'http://{self.config.participant["scenario_args"]["controller"]}/scenario/{self.config.participant["scenario_args"]["name"]}/node/update'

        # Send the POST request if the controller is available
        try:
            response = requests.post(url, data=json.dumps(self.config.participant), headers={'Content-Type': 'application/json'})
        except requests.exceptions.ConnectionError:
            logging.error(f'Error connecting to the controller at {url}')
            return

        # If endpoint is not available, log the error
        if response.status_code != 200:
            logging.error(f'Error received from controller: {response.status_code}')
            logging.error(response.text)

    # def __report_logs_to_controller(self):
    #     """
    #     Report node logs to the controller.
    #
    #     Returns:
    #
    #     """
    #
    #     # Set the URL for the POST request
    #     url = f'http://{self.config.participant["scenario_args"]["controller"]}/scenario/{self.config.participant["scenario_args"]["name"]}/node/{self.config.participant["device_args"]["uid"]}/update/logs'
    #
    #     # Send the POST request if the controller is available
    #     try:
    #         with open(self.log_filename + ".log", 'r') as f:
    #             response = requests.post(url, data=f, headers={'Content-Type': 'text/plain'})
    #     except requests.exceptions.ConnectionError:
    #         logging.error(f'Error connecting to the controller at {url}')
    #         return
    #
    #     # If endpoint is not available, log the error
    #     if response.status_code != 200:
    #         logging.error(f'Error received from controller: {response.status_code}')
    #         logging.error(response.text)

    def __report_resources(self):
        """
        Report node resources to the controller.

        Returns:

        """
        step = int((datetime.now() - datetime.strptime(self.config.participant["scenario_args"]["start_time"], "%d/%m/%Y %H:%M:%S")).total_seconds())
        import sys
        import psutil
        # Gather CPU usage information
        cpu_percent = psutil.cpu_percent()
        # Gather CPU temperature information
        cpu_temp = 0
        if sys.platform == "linux":
            cpu_temp = psutil.sensors_temperatures()['coretemp'][0].current
        # Gather RAM usage information
        ram_percent = psutil.virtual_memory().percent
        # Gather disk usage information
        disk_percent = psutil.disk_usage("/").percent
        logging.info(f'Resources: CPU {cpu_percent}%, CPU temp {cpu_temp}C, RAM {ram_percent}%, Disk {disk_percent}%')
        self.learner.logger.log_metrics({"Resources/CPU_percent": cpu_percent, "Resources/CPU_temp": cpu_temp, "Resources/RAM_percent": ram_percent, "Resources/Disk_percent": disk_percent}, step=step)

        # Gather network usage information
        net_io_counters = psutil.net_io_counters()
        bytes_sent = net_io_counters.bytes_sent
        bytes_recv = net_io_counters.bytes_recv
        packets_sent = net_io_counters.packets_sent
        packets_recv = net_io_counters.packets_recv

        logging.info(f'Resources: Bytes sent {bytes_sent}, Bytes recv {bytes_recv}, Packets sent {packets_sent}, Packets recv {packets_recv}')
        self.learner.logger.log_metrics({"Resources/Bytes_sent": bytes_sent, "Resources/Bytes_recv": bytes_recv, "Resources/Packets_sent": packets_sent, "Resources/Packets_recv": packets_recv}, step=step)

        # Log uptime information
        uptime = psutil.boot_time()
        logging.info(f'Resources: Uptime {uptime}')
        self.learner.logger.log_metrics({"Resources/Uptime": uptime}, step=step)

        # Check if pynvml is available
        try:
            import pynvml
            pynvml.nvmlInit()
            devices = pynvml.nvmlDeviceGetCount()
            for i in range(devices):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_percent = gpu_mem.used / gpu_mem.total * 100
                logging.info(f'Resources: GPU-{i} {gpu_percent}%, GPU temp {gpu_temp}C, GPU mem {gpu_mem_percent}%')
                self.learner.logger.log_metrics({f"Resources/GPU{i}_percent": gpu_percent, f"Resources/GPU{i}_temp": gpu_temp, f"Resources/GPU{i}_mem_percent": gpu_mem_percent}, step=step)
        except ModuleNotFoundError:
            logging.info(f'pynvml not found, skipping GPU usage')

    def __store_model_parameters(self, obj):
        """
        Store the model parameters in the node.

        Args:
            obj: Model parameters.

        Returns:

        """
        # TODO: FIX THIS
        # (
        #     decoded_model,
        #     contributors,
        #    weight,
        # ) = self.learner.decode_parameters(obj)
        # if self.learner.check_parameters(decoded_model):
        self.__stored_model_parameters += obj
        logging.info("[NODE.__store_model_parameters (PROXY)] Stored model parameters: {}".format(len(self.__stored_model_parameters)))
