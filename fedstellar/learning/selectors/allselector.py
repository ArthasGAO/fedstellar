

import numpy as np
import logging
from sklearn.preprocessing import normalize


from fedstellar.learning.selectors.selector import Selector


class AllSelector(Selector):
    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]

    def node_selection(self, node):

        if len(node) == 0:
            logging.error(
                "[All Selector] Trying to select nodes when there is no nodes"
            )
            return None

        neighbors = self.neighbors_list.copy()
        neighbors.remove(node)
        logging.info("[All Selector]   neighbors = {}".format(neighbors))
        
        selected_nodes = neighbors
        logging.info(
            "[All SELECTOR] neighbors ={}, selected_nodes = {}".format(neighbors,selected_nodes))

        logging.info("[All selector]  selection FINISHED")


        return selected_nodes


