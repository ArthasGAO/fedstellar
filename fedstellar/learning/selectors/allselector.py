

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

        

        neighbors = self.neighbors_list.copy()
        
        if len(neighbors) == 0:
            logging.error(
                "[NewSelector] Trying to select neighbors when there is no neighbors"
            )
            return None
        logging.info("[All Selector]   neighbors = {}".format(neighbors))
        
        selected_nodes = neighbors
        logging.info(
            "[All SELECTOR] neighbors ={}, selected_nodes = {}".format(neighbors,selected_nodes))

        logging.info("[All selector]  selection FINISHED")


        return selected_nodes


