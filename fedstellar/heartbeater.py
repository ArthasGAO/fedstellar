# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


import logging
import threading
import time

from fedstellar.config.config import Config
from fedstellar.utils.observer import Events, Observable


#####################
#    Heartbeater    #
#####################


class Heartbeater(threading.Thread, Observable):
    """
    Thread based heartbeater that sends a beat message to all the neighbors of a node every `HEARTBEAT_PERIOD` seconds.

    It also maintains a list of active neighbors, which is created by receiving different heartbear messages.
    Neighbors from which a heartbeat is not received in ``NODE_TIMEOUT`` will be eliminated

    Communicates with node via observer pattern.

    Args:
        nodo_padre (Node): Node that use the heartbeater.
    """

    def __init__(self, node_name, neighbors, config: Config):
        Observable.__init__(self)
        threading.Thread.__init__(self, name="heartbeater-" + node_name)
        self.__node_name = node_name
        self.__terminate_flag = threading.Event()

        self.config = config

        self.__count = 0

        # List of neighbors
        self.__neighbors = neighbors
        self.__nodes = {}
        self.__nodes_role = {}

    def run(self):
        """
        Send a beat every HEARTBEAT_PERIOD seconds to all the neighbors of the node.
        Also, it will clear from the neighbors list the nodes that haven't sent a heartbeat in NODE_TIMEOUT seconds.
        It happend ``HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD`` per HEARTBEAT_PERIOD
        """
        while not self.__terminate_flag.is_set():
            # We do not check if the message was sent
            #   - If the model is sending, a beat is not necessary
            #   - If the connection its down timeouts will destroy connections
            self.notify(Events.SEND_BEAT_EVENT, None)
            self.get_nodes()  # TODO: testing
            self.__count += 1
            # Send role notify each 10 beats
            if self.__count % 2 == 0:
                self.notify(Events.SEND_ROLE_EVENT, None)

            # Wait and refresh node list
            for _ in range(self.config.participant_config["HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD"]):
                self.clear_nodes()
                time.sleep(
                    self.config.participant_config["HEARTBEAT_PERIOD"]
                    / self.config.participant_config["HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD"]
                )

    def clear_nodes(self):
        """
        Clear the list of neighbors.
        """
        for n in [
            node
            for node, t in list(self.__nodes.items())
            if time.time() - t > self.config.participant_config["NODE_TIMEOUT"]
        ]:
            logging.debug(
                "[HEARTBEATER] Removed {} from the network ".format(n)
            )
            self.__nodes.pop(n)
            self.__nodes_role.pop(n)

    def add_node(self, node):
        """
        Add a node to the list of neighbors.

        Args:
            node (Node): Node to add to the list of neighbors.
        """
        if node != self.__node_name:
            self.__nodes[node] = time.time()

    def add_node_role(self, node, role):
        """
        Add a node to the list of neighbors.

        Args:
            node (Node): Node name
            role: Role of the node
        """
        if node != self.__node_name:
            self.__nodes_role[node] = role

    def get_nodes(self):
        """
        Get the list of actual neighbors.

        Returns:
            list: List of neighbors.
        """
        node_list = list(self.__nodes.keys())
        if self.__node_name not in node_list:
            node_list.append(self.__node_name)

        logging.info("[HEARTBEATER] Nodes heartbeater: {}".format(node_list))
        logging.info("[HEARTBEATER] Nodes role: {}".format(self.__nodes_role))
        logging.info("[HEARTBEATER] Nodes reference basenode: {}".format(self.__neighbors))
        return node_list

    def stop(self):
        """
        Stop the heartbeater.
        """
        self.__terminate_flag.set()