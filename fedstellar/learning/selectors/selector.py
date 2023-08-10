

import logging
import threading
from fedstellar.utils.observer import Events, Observable


class Selector():

    def __init__(self, node_name="unknown", config=None):
        self.node_name = node_name
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        self.neighbors_list = []
        self.features = {}
        self.age_list = {}

    def add_node_features(self, node, features, latency):
        self.features[node] = {}
        self.features[node]["loss"] = features[0]
        self.features[node]["cpu_percent"] = features[1]
        self.features[node]["data_size"] = features[2]
        self.features[node]["bytes_received"] = features[3]
        self.features[node]["bytes_send"] = features[4]
        self.features[node]["availability"] = features[5]
        self.features[node]["latency"] = latency

    def set_neighbors(self, neighbors_list):

        self.neighbors_list = neighbors_list
        logging.info("[SELECTOR] Setting neighbors list {}".format(
            self.neighbors_list))

    def get_neighbors(self):

        return self.neighbors_list

    def add_neighbor(self, neighbor):

        self.neighbors_list.append(neighbor)
        logging.info("[Selector] add_neighbor {}".format(neighbor))

    def node_selection(self):
        """
        Template
        """

        pass

    def clear_selector_features(self):
        self.features = {}

    def init_age(self):

        for i in self.neighbors_list:
            self.age_list[i] = 1

        logging.info(
            "[SELECTOR] INITIATING SELECTOR WITH NEIGHBORS AND AGE self.age_list = {} ====================INITIATING==============".format(self.age_list))
