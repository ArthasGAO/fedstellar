

import numpy as np
import logging
from sklearn.preprocessing import normalize


from fedstellar.learning.selectors.selector import Selector


class NewSelector(Selector):
    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]

    def node_selection(self, node):

        
        neighbors = self.neighbors_list.copy()
        
        logging.info(
            "[NEW SELECTOR] neighbors = {} ==================================".format(neighbors))
        
        if len(neighbors) == 0:
            logging.error(
                "[NewSelector] Trying to select neighbors when there is no neighbors"
            )
            return node
        
        num_selected = max(1,int(len(neighbors)*0.8//1))
        
        availabililty = []
        feature_array = np.empty((7, 0))

        logging.info(
            "[NEW SELECTOR] neighbors(after remove) = {} ==================================".format(neighbors))

        for node in neighbors:

            feature_list = list((self.features[node]["loss"],
                                 self.features[node]["cpu_percent"],
                                 self.features[node]["data_size"],
                                 self.features[node]["bytes_received"],
                                 self.features[node]["bytes_send"],
                                 self.features[node]["latency"],
                                 self.age_list[node]))
            # if loss not available, set loss to 100
            if feature_list[0] == -1:
                feature_list[0] = 100

            logging.info(
                "[New Selector]   feature_list = {}".format(feature_list))

            availabililty.append(self.features[node]["availability"])

            feature = np.array(feature_list).reshape(-1, 1).astype(np.float64)
            feature_array = np.append(feature_array, feature, axis=1)
        
        # 1 / cpu_percent
        feature_array[1, :] = 1/feature_array[1, :]
        
        
        # 1 / latency
        feature_array[5, :] = 1/feature_array[5, :]
        
        logging.info(
            "[New Selector]   feature_array = \n {}".format(feature_array))
        # Normalized features
        feature_array_normed = normalize(feature_array, axis=1, norm='l1')

        # Add weight to features
        # Loss, cpu, data size, data rec, data send, latency, age 
        weight = [10.0, 1.0, 1.0, 0.5, 0.5, 10.0, 3.0]
        weight = np.array(weight).reshape(-1, 1)
        feature_array_weighted = np.multiply(feature_array_normed, weight)
        logging.info("[NEW SELECTOR] feature_array_weighted = {}".format(
            feature_array_weighted))

        # Before availability
        scores = np.sum(feature_array_weighted, axis=0)

        # Add availability
        final_scores = np.multiply(scores, np.array(availabililty))
        logging.info("[NEW SELECTOR] final_scores = {}".format(final_scores))

        # Probability selection
        p = normalize([final_scores], axis=1, norm='l1')
        logging.info("[NEW SELECTOR] P = {}".format(p))

        selected_nodes = np.random.choice(
            neighbors, num_selected, replace=False, p=p[0]).tolist()
        
        selected_nodes.append(self.node_name)
        logging.info("[NewSelector] selection appending node = {}".format(self.node_name))
        
        logging.info(
            "[NEW SELECTOR] neighbors ={},num_selected ={}, p = {}, selected_nodes = {}".format(neighbors,num_selected,p,selected_nodes))

        logging.info("[NewSelector] selection FINISHED")

        # Update age dict
        for node in neighbors:
            logging.info(
                "[NEW SELECTOR] adding age condition node = {}".format(node))
            if node not in selected_nodes:
                self.age_list[node] = self.age_list[node] + 2

        return selected_nodes


