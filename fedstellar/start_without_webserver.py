import json
import os, sys
from datetime import datetime
import networkx
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Parent directory where is the fedstellar module
import fedstellar
from fedstellar.controller import Controller
import numpy as np
import math
import logging


logging.basicConfig(level=logging.DEBUG)

basic_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basic_config.json')
example_node_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/participant.json.example')
os.environ["FEDSTELLAR_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_controller_configs(basic_config_path=basic_config_path):
    basic_config = ''
    with open(basic_config_path) as f:
        basic_config = json.load(f)

    scenario_name = basic_config['scenario_name']
    if len(scenario_name) == 0:
        scenario_name = f'{basic_config["dataset"]}_{basic_config["is_iid"]}_{basic_config["model"]}_' \
                        f'{basic_config["aggregation"]}_' \
                        f'{basic_config["topology"].replace(" ", "")}_' \
                        f'{basic_config["attack"].replace(" ", "")}_{basic_config["targeted"]}_' \
                        f'N{basic_config["poisoned_node_percent"]}-S{basic_config["poisoned_sample_percent"]}_' \
                        f'R{basic_config["poisoned_ratio"]}_' \
                        f'{basic_config["noise_type"].replace(" ", "")}_' \
                        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(scenario_name)
    # if len(scenario_name) == 0:
    #    scenario_name = f'fedstellar_{basic_config["federation"]}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
    basic_config['scenario_name'] = scenario_name
    basic_config["topology"] = basic_config["topology"].lower()

    matrix = create_topo_matrix(basic_config)
    basic_config['matrix'] = matrix

    attack_matrix = create_attack_matrix(basic_config)
    basic_config['attack_matrix'] = attack_matrix

    config_dir = os.path.join(basic_config['config'], scenario_name)
    os.makedirs(config_dir, exist_ok=True)
    with open(f'{config_dir}/controller.json', 'w') as f:
        json.dump(basic_config, f, indent=1)

    os.makedirs(os.path.join(basic_config['logs'], scenario_name), exist_ok=True)
    os.makedirs(os.path.join(basic_config['models'], scenario_name), exist_ok=True)
    return basic_config


def create_topo_matrix(basic_config):
    matrix = basic_config['matrix']
    num_nodes = int(basic_config['n_nodes'])
    node_range = range(0, num_nodes)
    # if len(matrix) != 0:
    #    return matrix
    if basic_config["topology"] == "fully":
        matrix = []
        for i in node_range:
            node_adjcent = []
            for j in node_range:
                if i != j:
                    node_adjcent.append(1)
                else:
                    node_adjcent.append(0)
            matrix.append(node_adjcent)
    elif basic_config["topology"] == "ring":
        matrix = []
        for i in node_range:
            node_adjcent = []
            for j in node_range:
                if j == i + 1:
                    node_adjcent.append(1)
                elif j == i - 1:
                    node_adjcent.append(1)
                elif i == 0 and j == node_range[-1]:
                    node_adjcent.append(1)
                elif i == node_range[-1] and j == 0:
                    node_adjcent.append(1)
                else:
                    node_adjcent.append(0)
            matrix.append(node_adjcent)
    elif basic_config["topology"] == "random":
        random_seed = random.randint(0, 100)
        matrix = []
        graph = networkx.erdos_renyi_graph(num_nodes, 0.6, seed=random_seed)
        adj_matrix = networkx.adjacency_matrix(graph)
        matrix = adj_matrix.todense().tolist()
    elif basic_config["topology"] == "star":
        matrix = []
        for i in node_range:
            node_adjcent = []
            for j in node_range:
                if i == 0 and j != 0:
                    node_adjcent.append(1)
                elif i != 0 and j == 0:
                    node_adjcent.append(1)
                else:
                    node_adjcent.append(0)
            matrix.append(node_adjcent)
    return matrix

def create_attack_matrix(basic_config):
    attack_matrix = []
    attack = basic_config["attack"]
    num_nodes = int(basic_config['n_nodes'])
    node_range = range(0, num_nodes)
    federation = basic_config["federation"]
    nodes_index = []
    poisoned_node_percent = int(basic_config["poisoned_node_percent"])
    poisoned_sample_percent = int(basic_config["poisoned_sample_percent"])
    poisoned_ratio = int(basic_config["poisoned_ratio"])
    if federation == "DFL":
        nodes_index = node_range
    else:
        nodes_index = node_range[1:]

    n_nodes = len(nodes_index)
    # Number of attacked nodes, round up
    num_attacked = int(round(poisoned_node_percent/100 * n_nodes))
    if num_attacked > n_nodes:
        num_attacked = n_nodes

    # Get the index of attacked nodes
    attacked_nodes = random.sample(nodes_index, num_attacked)

    # Assign the role of each node
    for node in node_range:
        node_att = 'No Attack'
        attack_sample_percent = 0
        attack_ratio = 0
        if node in attacked_nodes:
            node_att = attack
            attack_sample_percent = poisoned_sample_percent / 100
            attack_ratio = poisoned_ratio / 100
        attack_matrix.append([node, node_att, attack_sample_percent, attack_ratio])
    return attack_matrix


def create_particiants_configs(basic_config, example_node_config_path=example_node_config_path, start_port=45000):
    start_date_scenario = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    scenario_name = basic_config['scenario_name']
    logging.info("Generating the scenario {} at {}".format(scenario_name, start_date_scenario))

    import shutil
    num_nodes = int(basic_config['n_nodes'])
    node_range = range(0, num_nodes)
    config_dir = basic_config['config']
    federation = basic_config['federation']
    attack_matrix = basic_config['attack_matrix']
    network_gateway = basic_config['network_gateway']
    network_gateway_list = network_gateway.split('.')

    for node in node_range:
        participant_file = os.path.join(config_dir, scenario_name, f'participant_{node}.json')
        os.makedirs(os.path.dirname(participant_file), exist_ok=True)
        # Create a copy of participant.json.example
        shutil.copy(example_node_config_path, participant_file)

        # Update IP, port, and role
        with open(participant_file) as f:
            participant_config = json.load(f)
        
        # Update IP, port, and role

        current_client = int(network_gateway_list[-1]) + node + 1

        participant_config['network_args']['ip'] = f'{network_gateway_list[0]}.{network_gateway_list[1]}.{network_gateway_list[2]}.{current_client}'
        participant_config['network_args']['ipdemo'] = participant_config['network_args']['ip']  # legacy code
        participant_config['network_args']['port'] = start_port + node
        # participant_config['device_args']['idx'] = i
        if node == 0:
            participant_config["device_args"]["start"] = True
        role = ""
        if federation == 'DFL':
            role = "aggregator"
        elif federation == 'CFL' and node == 0:
            role = "server"
        else:
            role = "trainer"

        participant_config["device_args"]["role"] = role

        # The following parameters have to be same for all nodes (for now)
        participant_config["scenario_args"]["rounds"] = int(basic_config["rounds"])
        participant_config["scenario_args"]["n_nodes"] = int(num_nodes)

        # Logging configuration
        participant_config['tracking_args']['enable_remote_tracking'] = basic_config["remote_tracking"]

        participant_config["data_args"]["dataset"] = basic_config["dataset"]
        participant_config["data_args"]["is_iid"] = basic_config["is_iid"]
        participant_config["model_args"]["model"] = basic_config["model"]
        participant_config["training_args"]["epochs"] = int(basic_config["epochs"])
        participant_config["device_args"]["accelerator"] = basic_config["accelerator"]  # same for all nodes
        participant_config["aggregator_args"]["algorithm"] = basic_config["aggregation"]
        participant_config["adversarial_args"]["attack_env"] = basic_config["attack"]
        participant_config["adversarial_args"]["poisoned_node_percent"] = int(basic_config["poisoned_node_percent"])
        participant_config["adversarial_args"]["targeted"] = basic_config["targeted"]
        participant_config["adversarial_args"]["noise_type"] = basic_config["noise_type"]

        # Get attack config for each node
        for atts in attack_matrix:
            if node == atts[0]:
                attack = atts[1]
                poisoned_sample_percent = atts[2]
                poisoned_ratio = atts[3]
        participant_config["adversarial_args"]["attacks"] = attack
        participant_config["adversarial_args"]["targeted"] = basic_config["targeted"]
        participant_config["adversarial_args"]["poisoned_sample_percent"] = poisoned_sample_percent
        participant_config["adversarial_args"]["poisoned_ratio"] = poisoned_ratio
        participant_config["adversarial_args"]["target_label"]=basic_config["target_label"]
        participant_config["adversarial_args"]["target_changed_label"]=basic_config["target_changed_label"]
        participant_config["adversarial_args"]["is_iid"]=basic_config["is_iid"]

        with open(participant_file, 'w') as f:
                    json.dump(participant_config, f, sort_keys=False, indent=4)

    args = {
                "scenario_name": scenario_name,
                "config": config_dir,
                "logs": basic_config['logs'],
                "models": basic_config['models'],
                "n_nodes": basic_config["n_nodes"],
                "matrix": basic_config["matrix"],
                "federation": basic_config["federation"],
                "topology": basic_config["topology"],
                "simulation": basic_config["simulation"],
                "env": basic_config["env"],
                "webserver": basic_config["webserver"],
                "attack_matrix": attack_matrix,
                "docker": basic_config["docker"],
                "webport": basic_config["webport"],
                "statsport": basic_config["statsport"],
                "python": basic_config["python"],
                "network_subnet": basic_config["network_subnet"],
                "network_gateway": basic_config["network_gateway"],
            }

    import argparse
    args = argparse.Namespace(**args)
    controller = Controller(args)  # Generate an instance of controller in this new process
    controller.load_configurations_and_start_nodes()
    # Generate/Update the scenario in the database
    # scenario_update_record(scenario_name=controller.scenario_name, start_time=controller.start_date_scenario, end_time="", status="running", title=basic_config["scenario_title"], description=basic_config["scenario_description"])

if __name__ == "__main__":
    # Parse args from command line
    basic_config = generate_controller_configs()
    create_particiants_configs(basic_config, start_port=45000)
