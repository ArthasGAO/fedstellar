

import numpy as np
import logging
from sklearn.preprocessing import normalize


from fedstellar.learning.selectors.selector import Selector


class NewSelector(Selector):
    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
    
    
    



    def first_round_selection(self, node):

        if len(node) == 0:
            logging.error(
                "[NewSelector] Trying to select nodes when there is no nodes"
            )
            return None
        
        neighbors = self.neighbors_list.copy()
        neighbors.remove(node)
        logging.info("[New Selector]   neighbors = {}".format(neighbors))
        num_selected = int(len(neighbors)*0.8//1)
        availabililty = []
        feature_array = np.empty((6,0))
        
        logging.info("[NEW SELECTOR] neighbors = {} ==================================".format(neighbors))

        for node in neighbors:
            
            logging.info("[NEW SELECTOR] AGE = {} ==================================".format(self.age_list[node]))
        
            
            feature_list = list((self.features[node]["cpu_percent"],
                                self.features[node]["data_size"],
                                self.features[node]["bytes_received"],
                                self.features[node]["bytes_send"],
                                self.features[node]["latency"],
                                self.age_list[node]))
            
            logging.info("[New Selector]   feature_list = {}".format(feature_list))
            
            availabililty.append(self.features[node]["availability"])
            
                        
            feature = np.array(feature_list).reshape(-1, 1).astype(np.float64)          
            feature_array = np.append(feature_array, feature, axis=1)
        
                        
        # Normalized features
        feature_array_normed = normalize(feature_array, axis=1, norm='l1')
        
        # Add weight to features
        weight = [1.0, 1.0, 0.5, 0.5, 1.0, 1.0]
        weight = np.array(weight).reshape(-1, 1)
        feature_array_weighted = np.multiply(feature_array_normed, weight)
        logging.info("[NEW SELECTOR] feature_array_weighted = {}".format(feature_array_weighted))
        

        # Before availability
        scores = np.sum(feature_array_weighted, axis=0)
        
        # Add availability
        final_scores = np.multiply(scores, np.array(availabililty))
        logging.info("[NEW SELECTOR] final_scores = {}".format(final_scores))
        
        # Probability selection
        p = normalize([final_scores], axis=1, norm='l1')
        logging.info("[NEW SELECTOR] P = {}".format(p))
        
        selected_nodes = np.random.choice(neighbors, num_selected, replace=False, p=p[0])
        logging.info("[NEW SELECTOR] selected_nodes = {}".format(selected_nodes))
        
            
            
            
        logging.info("[NewSelector] First round selection FINISHED")
        
        
        # Update age dict
        for node in neighbors:
            if node not in selected_nodes:
                self.age_list[node] = self.age_list[node]+ 5


        return selected_nodes
    
    def second_round_selection(self, nodes):

        if len(nodes) == 0:
            logging.error(
                "[NewSelector] Trying to select nodes when there is no nodes"
            )
            return None
        logging.info("[NewSelector] Second round selection performed")


        return nodes
    
    
