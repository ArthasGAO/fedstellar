import datetime
import json
import os
import sys
from datetime import datetime
import time

N_EXPERIMENTS = 1

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # Parent directory where is the fedstellar module

from fedstellar.start_without_webserver import generate_controller_configs, create_particiants_configs


def get_scenario_name(basic_config):
    scenario_name = f'{basic_config["dataset"]}_{int(basic_config["is_iid"])}_{basic_config["model"]}_' \
                    f'{basic_config["aggregation"]}_' \
                    f'{basic_config["topology"].replace(" ", "")}_' \
                    f'{basic_config["attack"].replace(" ", "")}_{int(basic_config["targeted"])}_' \
                    f'N{basic_config["poisoned_node_persent"]}-S{basic_config["poisoned_sample_persent"]}_' \
                    f'R{basic_config["poisoned_ratio"]}_' \
                    f'{basic_config["noise_type"].replace(" ", "")}_' \
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    return scenario_name


basic_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basic_config.json")
example_node_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/participant.json.example')

dataset_list = ["MNIST", "FEMNIST", "FASHIONMNIST", "CIFAR10", "CIFAR100", "Sent140", "SYSCALL", "KISTSUN"]
model_list = ["MLP", "CNN", "RNN"]
federation_list = ["DFL", "CFL"]
topology_list = ["star", "fully", "ring", "random"]
# topology_list = ["star"]
# attack_list = ["Label Flipping", "Sample Poisoning", "Model Poisoning"]

# poisoned_node_persent_list = [20, 40, 60, 80, 100]
poisoned_node_persent_list = [60]  # 60 80
poisoned_sample_persent_list = [60]
noise_type_list = ["salt", "gaussian", "s&p"]
# poisoned_ratio_list = [1, 10, 20]
poisoned_ratio_list = [80]

targeted_list = [True, False]

with open(basic_config_path) as f:
    basic_config = json.load(f)
n_nodes = 10
start_port = 46500

dataset = dataset_list[2]
model = model_list[0]
federation = federation_list[0]
topology = topology_list[1]
poisoned_node = poisoned_node_persent_list[0]
poisoned_sample = poisoned_sample_persent_list[0]
noise_type = noise_type_list[0]
targeted = targeted_list[0]
poisoned_ratio = poisoned_ratio_list[0]

# scenario_title = f"{dataset}_{model}_{federation}_{aggregation}_{topology}_{attack}_{targeted}_{poisoned_node}_{poisoned_sample}_{noise_type}_{poisoned_ratio}"

basic_config["is_iid"] = True
basic_config["remote_tracking"] = True
basic_config["rounds"] = 10
basic_config["epochs"] = 5

basic_config["targeted"] = False
basic_config["noise_type"] = noise_type
basic_config["poisoned_ratio"] = poisoned_ratio
basic_config["dataset"] = dataset
basic_config["model"] = model

# basic_config["scenario_title"] = scenario_title
basic_config["federation"] = federation

basic_config["topology"] = topology
basic_config["n_nodes"] = n_nodes
basic_config["poisoned_node_persent"] = poisoned_node
basic_config["poisoned_sample_persent"] = poisoned_sample
attack_list = ["No Attack", "Model Poisoning", "Sample Poisoning", "Label Flipping"]
attack = attack_list[1]

# aggregation_list = ["FedAvg", "Krum", "Median", "TrimmedMean", "Sentinel"]
aggregation_list = ["Sentinel"]

with open(basic_config_path, "w") as f:
    json.dump(basic_config, f)
time.sleep(2)


if attack == "No Attack":
    # No Attack
    for i in range(N_EXPERIMENTS):
        for aggregation in aggregation_list:
            basic_config["attack"] = "No Attack"
            basic_config["aggregation"] = aggregation
            basic_config["poisoned_node_persent"] = 0
            basic_config["poisoned_sample_persent"] = 0
            basic_config["poisoned_ratio"] = 0

            basic_config['scenario_name'] = get_scenario_name(basic_config)
            start_port += basic_config["n_nodes"]

            with open(basic_config_path, "w") as f:
                json.dump(basic_config, f)
            time.sleep(2)
            basic_config = generate_controller_configs()
            create_particiants_configs(basic_config, example_node_config_path, start_port)
            time.sleep(300)
            with open(basic_config_path) as f:
                basic_config = json.load(f)


if attack == "Model Poisoning":
    # Model Poisoning
    for i in range(N_EXPERIMENTS):
        for aggregation in aggregation_list:
            for node_persent in poisoned_node_persent_list:
                # for poisoned_ratio in poisoned_ratio_list:

                basic_config["attack"] = "Model Poisoning"
                basic_config["aggregation"] = aggregation
                basic_config["poisoned_node_persent"] = node_persent
                basic_config["poisoned_sample_persent"] = poisoned_sample
                basic_config["poisoned_ratio"] = poisoned_ratio

                basic_config['scenario_name'] = get_scenario_name(basic_config)
                start_port += basic_config["n_nodes"]

                with open(basic_config_path, "w") as f:
                    json.dump(basic_config, f)
                time.sleep(2)

                basic_config = generate_controller_configs()
                create_particiants_configs(basic_config, example_node_config_path, start_port)
                time.sleep(300)
                with open(basic_config_path) as f:
                    basic_config = json.load(f)


if attack == "Sample Poisoning":
    # Label Flipping
    for i in range(N_EXPERIMENTS):
        for aggregation in aggregation_list:
            for node_persent in poisoned_node_persent_list:
                for poisoned_sample_persent in poisoned_sample_persent_list:

                    basic_config["attack"] = "Sample Poisoning"
                    basic_config["aggregation"] = aggregation
                    basic_config["poisoned_node_persent"] = node_persent
                    basic_config["poisoned_sample_persent"] = poisoned_sample
                    basic_config["poisoned_sample_persent"] = poisoned_ratio

                    basic_config['scenario_name'] = get_scenario_name(basic_config)
                    start_port += basic_config["n_nodes"]

                    with open(basic_config_path, "w") as f:
                        json.dump(basic_config, f)
                    time.sleep(2)

                    basic_config = generate_controller_configs()
                    create_particiants_configs(basic_config, example_node_config_path, start_port)
                    time.sleep(300)
                    with open(basic_config_path) as f:
                        basic_config = json.load(f)

if attack == "Label Flipping":
    # Sample Poisoning
    for i in range(N_EXPERIMENTS):
        for aggregation in aggregation_list:
            for node_persent in poisoned_node_persent_list:
                for poisoned_sample_persent in poisoned_sample_persent_list:
                    # for poisoned_ratio in poisoned_ratio_list:

                    basic_config["attack"] = "Label Flipping"
                    basic_config["aggregation"] = aggregation
                    basic_config["poisoned_node_persent"] = node_persent
                    basic_config["poisoned_sample_persent"] = poisoned_sample_persent
                    basic_config["poisoned_ratio"] = 0

                    basic_config['scenario_name'] = get_scenario_name(basic_config)
                    start_port += basic_config["n_nodes"]

                    with open(basic_config_path, "w") as f:
                        json.dump(basic_config, f)
                    time.sleep(2)
                    basic_config = generate_controller_configs()
                    create_particiants_configs(basic_config, example_node_config_path, start_port)
                    with open(basic_config_path, "w") as f:
                        json.dump(basic_config, f)
                    time.sleep(300)
                    with open(basic_config_path) as f:
                        basic_config = json.load(f)
