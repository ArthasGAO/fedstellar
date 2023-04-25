import json
import os, sys,time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # Parent directory where is the fedstellar module

from fedstellar.start_without_webserver import generate_controller_configs, create_particiants_configs

basic_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basic_config.json")


dataset_list = ["MNIST","FEMNIST","FASHIONMNIST","CIFAR10", "CIFAR100","Sent140","SYSCALL","KISTSUN"]
model_list = ["MLP", "CNN", "RNN"]
federation_list = ["DFL","CFL"]
topology_list = ["star", "fully",  "ring", "random"]
# topology_list = ["star"]
attack_list = ["Label Flipping", "Sample Poisoning", "Model Poisoning"]
poisoned_node_persent_list = [20, 40 , 60, 80, 100]
poisoned_sample_persent_list = [20, 40 , 60, 80, 100]
noise_type_list=["salt", "gaussian", "s&p"]
poisoned_ratio_list=[1,10,20]
# poisoned_node_persent_list = [20]
# poisoned_sample_persent = [20]
aggregation_list = ["FedAvg", "Krum", "Median", "TrimmedMean"]
# aggregation_list = ["FedAvg"]
targeted_list = [True, False]

with open(basic_config_path) as f:
        basic_config = json.load(f)
n_nodes = 3
start_port = 46500


dataset = dataset_list[7]
model=model_list[0]
federation= federation_list[0]
aggregation = aggregation_list[0]
topology= topology_list[1]
attack=attack_list[2]
poisoned_node=poisoned_node_persent_list[0]
poisoned_sample = poisoned_sample_persent_list[3]
noise_type = noise_type_list[1]
targeted = targeted_list[1]
poisoned_ratio = poisoned_ratio_list[1]

scenario_title = f"{dataset}_{model}_{federation}_{aggregation}_{topology}_{attack}_{targeted}_{poisoned_node}_{poisoned_sample}_{noise_type}_{poisoned_ratio}"


basic_config["targeted"] = targeted
basic_config["noise_type"] = noise_type
basic_config["poisoned_ratio"] = poisoned_ratio
basic_config["dataset"] = dataset
basic_config["model"] = model


basic_config["scenario_title"] = scenario_title
basic_config["federation"] = federation
basic_config["aggregation"] = aggregation
basic_config["topology"] = topology
basic_config["attack"] = attack
basic_config["poisoned_node_persent"] = poisoned_node
basic_config["poisoned_sample_persent"] = poisoned_sample
basic_config["n_nodes"] = n_nodes

with open(basic_config_path, "w") as f:
        json.dump(basic_config, f)
time.sleep(2)
basic_config = generate_controller_configs()
create_particiants_configs(basic_config,start_port= start_port)
start_port += basic_config["n_nodes"]
# time.sleep(600)
with open(basic_config_path, 'r') as f:
        basic_config = json.load(f)


# for federation in federation_list:
#         for aggregation in aggregation_list:
#                 for topology in topology_list:
#                         for attack in attack_list:
#                                 for poisoned_node in poisoned_node_persent_list:
#                                         for poisoned_sample in poisoned_sample_persent:
#                                                 scenario_title = f"{federation}_{aggregation}_{topology}_{attack}_{poisoned_node}{poisoned_sample}"
#                                                 basic_config["scenario_title"] = scenario_title
#                                                 basic_config["federation"] = federation
#                                                 basic_config["aggregation"] = aggregation
#                                                 basic_config["topology"] = topology
#                                                 basic_config["attack"] = attack
#                                                 basic_config["poisoned_node_persent"] = poisoned_node
#                                                 basic_config["poisoned_sample_persent"] = poisoned_sample
#                                                 basic_config["n_nodes"] = n_nodes

#                                                 with open(basic_config_path, "w") as f:
#                                                     json.dump(basic_config, f)
#                                                 time.sleep(2)
#                                                 basic_config = generate_controller_configs()
#                                                 create_particiants_configs(basic_config, )
#                                                 start_port += basic_config["n_nodes"]
#                                                 time.sleep(600)
#                                                 with open(basic_config_path) as f:
#                                                     basic_config = json.load(f)
                                                