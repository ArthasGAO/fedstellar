import datetime
import json
import os
import sys
from datetime import datetime
import time
import platform
import docker

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # Parent directory where is the fedstellar module

from fedstellar.start_without_webserver import generate_controller_configs, create_particiants_configs

docker_client = docker.from_env()

# kill running processes (Ubuntu):
# pkill -9 -f node_start.py


def wait_docker_finished():
    fed_filter = {'label': 'fedstellar'}
    is_prev_finished = False
    while not is_prev_finished:
        fedstellar_nodes = docker_client.containers.list(filters=fed_filter)
        if len(fedstellar_nodes) != 0:
            print("Previous experiment still running")
            is_prev_finished = False
            time.sleep(30)
        else:
            print("*************** Previous experiment finished *************** \n")
            docker_client.networks.prune()
            is_prev_finished = True


def get_scenario_name(basic_config):
    scenario_name = f'{basic_config["dataset"]}_{int(basic_config["is_iid"])}_{basic_config["model"]}_' \
                    f'{basic_config["aggregation"]}_' \
                    f'{basic_config["topology"].replace(" ", "")}_' \
                    f'{basic_config["attack"].replace(" ", "")}_{int(basic_config["targeted"])}_' \
                    f'N{basic_config["poisoned_node_percent"]}-S{basic_config["poisoned_sample_percent"]}_' \
                    f'R{basic_config["poisoned_ratio"]}_' \
                    f'{basic_config["noise_type"].replace(" ", "").replace("&", "")}_' \
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    return scenario_name

basic_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basic_config.json")
example_node_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/participant.json.example')

fed_filter = {'label': 'fedstellar'}
containers = docker_client.containers.list(filters=fed_filter)
networks = docker_client.networks.list(filters=fed_filter)
if len(containers) != 0:
    print("Experiment still running")
    exit(-1)

docker_client.networks.prune()

# model_list = ["MLP", "CNN", "RNN"]
# federation_list = ["DFL", "CFL"]
# topology_list = ["star", "fully", "ring", "random"]
# topology_list = ["star"]
# attack_list = ["Label Flipping", "Sample Poisoning", "Model Poisoning"]

# poisoned_node_percent_list = [20, 40, 60, 80, 100]
# poisoned_node_percent_list = [30]  # 60 80
# poisoned_sample_percent_list = [100]
# noise_type_list = ["salt", "gaussian", "s&p"]
# poisoned_ratio_list = [1, 10, 20]

with open(basic_config_path) as f:
    basic_config = json.load(f)

start_port = 46500

python_windows = 'C:\\Users\\janos.LAPTOP-42CLK60G\\Repos\\fedstellar-robust\\.venv\\Scripts\\python'
python_macos = "/opt/homebrew/anaconda3/envs/fedstellar2/bin/python"
python_ubuntu = "/home/baltensperger/miniconda3/envs/fedstellar/bin/python"
if platform.system() == 'Linux':
    python = python_ubuntu
elif platform.system() == 'Windows':
    python = python_windows
else:
    python = python_macos
basic_config["python"] = python

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app_path = os.path.join(root_path, "app")
config_path = os.path.join(app_path, "config")
logs_path = os.path.join(app_path, "logs")
models_path = os.path.join(app_path, "models")

basic_config["config"] = config_path
basic_config["logs"] = logs_path
basic_config["models"] = models_path

basic_config["remote_tracking"] = True

"""
MacOS
"config": "/Users/janosch/Repos/fedstellar-robust/app/config",
"logs": "/Users/janosch/Repos/fedstellar-robust/app/logs",
"models": "/Users/janosch/Repos/fedstellar-robust/app/models",
"""

"""
Windows
"config": "C:\\Users\\janos.LAPTOP-42CLK60G\\Repos\\fedstellar-robust\\app\\config",
"logs": "C:\\Users\\janos.LAPTOP-42CLK60G\\Repos\\fedstellar-robust\\app\\logs",
"models": "C:\\Users\\janos.LAPTOP-42CLK60G\\Repos\\fedstellar-robust\\app\\models",
"""

basic_config["federation"] = "DFL"
basic_config["topology"] = "fully"
basic_config["is_iid"] = True
basic_config["dataset"] = "MNIST"
basic_config["model"] = "MLP"

basic_config["targeted"] = False
basic_config["noise_type"] = "s&p"
basic_config["poisoned_node_percent"] = 30
basic_config["poisoned_sample_percent"] = 100
basic_config["poisoned_ratio"] = 20

basic_config["n_nodes"] = 10
basic_config["rounds"] = 10
basic_config["epochs"] = 3

attack_list = ["No Attack", "Model Poisoning", "Sample Poisoning", "Label Flipping"]
attack = attack_list[1]

# poisoned_node_percent_list = [20, 50, 80]
poisoned_node_percent_list = [80]
poisoned_sample_percent_list = [30, 50, 100]
# poisoned_ratio_list = [1, 10, 20]
poisoned_ratio_list = [20]
# aggregation_list = ["FedAvg", "Krum", "Median", "TrimmedMean", "FlTrust", "Sentinel"]
# aggregation_list = ["Sentinel"]
aggregation_list = ["FedAvg"]


N_EXPERIMENTS = 1
EXPERIMENT_WAIT_SEC = 60 + 60 * basic_config["rounds"]


if attack == "No Attack":
    # No Attack
    for i in range(N_EXPERIMENTS):
        for aggregation in aggregation_list:
            basic_config["attack"] = "No Attack"
            basic_config["aggregation"] = aggregation
            basic_config["poisoned_node_percent"] = 0
            basic_config["poisoned_sample_percent"] = 0
            basic_config["poisoned_ratio"] = 0

            basic_config['scenario_name'] = get_scenario_name(basic_config)
            start_port += basic_config["n_nodes"]

            with open(basic_config_path, "w") as f:
                json.dump(basic_config, f, indent=4)
            time.sleep(2)
            basic_config = generate_controller_configs()
            create_particiants_configs(basic_config, example_node_config_path, start_port)
            time.sleep(EXPERIMENT_WAIT_SEC)
            with open(basic_config_path) as f:
                basic_config = json.load(f)

            wait_docker_finished()

if attack == "Model Poisoning":
    # Model Poisoning
    for i in range(N_EXPERIMENTS):
        for aggregation in aggregation_list:
            for node_percent in poisoned_node_percent_list:
                for poisoned_ratio in poisoned_ratio_list:

                    basic_config["attack"] = "Model Poisoning"
                    basic_config["aggregation"] = aggregation
                    basic_config["poisoned_node_percent"] = node_percent
                    basic_config["poisoned_sample_percent"] = 0
                    basic_config["poisoned_ratio"] = poisoned_ratio

                    basic_config['scenario_name'] = get_scenario_name(basic_config)
                    start_port += basic_config["n_nodes"]

                    with open(basic_config_path, "w") as f:
                        json.dump(basic_config, f, indent=4)
                    time.sleep(2)

                    basic_config = generate_controller_configs()
                    create_particiants_configs(basic_config, example_node_config_path, start_port)
                    time.sleep(EXPERIMENT_WAIT_SEC)
                    with open(basic_config_path) as f:
                        basic_config = json.load(f)

                    wait_docker_finished()

if attack == "Sample Poisoning":
    # Label Flipping
        for aggregation in aggregation_list:
            for node_percent in poisoned_node_percent_list:
                for poisoned_sample_percent in poisoned_sample_percent_list:
                    basic_config["attack"] = "Sample Poisoning"
                    basic_config["aggregation"] = aggregation
                    basic_config["poisoned_node_percent"] = node_percent
                    basic_config["poisoned_sample_percent"] = poisoned_sample_percent
                    basic_config["poisoned_sample_percent"] = poisoned_ratio

                    basic_config['scenario_name'] = get_scenario_name(basic_config)
                    start_port += basic_config["n_nodes"]

                    with open(basic_config_path, "w") as f:
                        json.dump(basic_config, f, indent=4)
                    time.sleep(2)

                    basic_config = generate_controller_configs()
                    create_particiants_configs(basic_config, example_node_config_path, start_port)
                    time.sleep(EXPERIMENT_WAIT_SEC)
                    with open(basic_config_path) as f:
                        basic_config = json.load(f)

                    wait_docker_finished()

if attack == "Label Flipping":
    # Sample Poisoning
    for i in range(N_EXPERIMENTS):
        for aggregation in aggregation_list:
            for node_percent in poisoned_node_percent_list:
                for poisoned_sample_percent in poisoned_sample_percent_list:
                    # for poisoned_ratio in poisoned_ratio_list:

                    basic_config["attack"] = "Label Flipping"
                    basic_config["aggregation"] = aggregation
                    basic_config["poisoned_node_percent"] = node_percent
                    basic_config["poisoned_sample_percent"] = poisoned_sample_percent
                    basic_config["poisoned_ratio"] = 0

                    basic_config['scenario_name'] = get_scenario_name(basic_config)
                    start_port += basic_config["n_nodes"]

                    with open(basic_config_path, "w") as f:
                        json.dump(basic_config, f, indent=4)
                    time.sleep(2)

                    basic_config = generate_controller_configs()
                    create_particiants_configs(basic_config, example_node_config_path, start_port)
                    time.sleep(EXPERIMENT_WAIT_SEC)
                    with open(basic_config_path) as f:
                        basic_config = json.load(f)

                    wait_docker_finished()
