

import numpy as np
import logging
from sklearn.preprocessing import normalize


from fedstellar.learning.selectors.selector import Selector


class RandomSelector(Selector):
    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]

    def node_selection(self, node):

        if len(node) == 0:
            logging.error(
                "[NewSelector] Trying to select nodes when there is no nodes"
            )
            return None

        neighbors = self.neighbors_list.copy()
        neighbors.remove(node)
        logging.info("[New Selector]   neighbors = {}".format(neighbors))
        num_selected = max(1,int(len(neighbors)*0.8//1))
        
        selected_nodes = np.random.choice(
            neighbors, num_selected, replace=False).tolist()
        logging.info(
            "[Random SELECTOR] neighbors ={},num_selected ={}, selected_nodes = {}".format(neighbors,num_selected,selected_nodes))

        logging.info("[RandomSelector]  selection FINISHED")

        return selected_nodes


