import datetime
import json
import os
import sys
from datetime import datetime
import time
import platform
import docker
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # Parent directory where is the fedstellar module

from fedstellar.start_without_webserver import generate_controller_configs, create_particiants_configs

docker_client = docker.from_env()

# kill running processes (Ubuntu):
# pkill -9 -f node_start.py

def wait_docker_finished():
    fed_filter = {'label': 'fedstellar-jb'}
    is_prev_finished = False
    while not is_prev_finished:
        try:
            fedstellar_nodes = docker_client.containers.list(filters=fed_filter)
            if len(fedstellar_nodes) != 0:
                print("Previous experiment still running")
                is_prev_finished = False
                time.sleep(30)
            else:
                print("*************** Previous experiment finished *************** \n")
                docker_client.networks.prune(filters=fed_filter)
                is_prev_finished = True
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")


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


total, used, free = shutil.disk_usage("/")

print("Total: %d GiB" % (total // (2**30)))
print("Used: %d GiB" % (used // (2**30)))
print("Free: %d GiB" % (free // (2**30)))

basic_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basic_config.json")
example_node_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/participant.json.example')

fed_filter = {'label': 'fedstellar-jb'}
containers = docker_client.containers.list(filters=fed_filter)
networks = docker_client.networks.list(filters=fed_filter)
if len(containers) != 0:
    print("Experiment still running")
    exit(-1)
docker_client.networks.prune(filters=fed_filter)

with open(basic_config_path) as f:
    basic_config = json.load(f)

start_port = 46500

# import os; print(os.environ["CONDA_PREFIX"])
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
basic_config["logging"] = True
basic_config["wandb_project"] = "fedstellar"

basic_config["federation"] = "DFL"
basic_config["topology"] = "fully"
basic_config["is_iid"] = True
basic_config["dataset"] = "FASHIONMNIST"  # MNIST, FASHIONMNIST, CIFAR10
basic_config["model"] = "MLP"  # MLP, simplemobilenet
basic_config["accelerator"] = "cpu"

basic_config["poisoned_node_percent"] = 0
basic_config["poisoned_sample_percent"] = 0
basic_config["poisoned_ratio"] = 0

basic_config["targeted"] = False
basic_config["target_label"] = 3
basic_config["target_changed_label"] = 7

basic_config["n_nodes"] = 2
basic_config["rounds"] = 3
basic_config["epochs"] = 3

basic_config["noise_type"] = "salt"
attack_list = ["No Attack", "Model Poisoning", "Sample Poisoning", "Label Flipping"]
attack = attack_list[3]

poisoned_node_percent_list = [80, 50, 10]
poisoned_node_percent_list = [50]
poisoned_sample_percent_list = [100, 50, 30]
poisoned_sample_percent_list = [100]
# poisoned_ratio_list = [1, 10, 20]
poisoned_ratio_list = [80]

aggregation_list = ["Krum", "FedAvg", "TrimmedMean", "FlTrust", "Sentinel", "SentinelGlobal"]
# aggregation_list = ["FedAvg", "TrimmedMean", "FlTrust"]
# aggregation_list = ["TrimmedMean"]
aggregation_list = ["SentinelGlobal"]
basic_config["sentinel_distance_threshold"] = 0.1
basic_config["sentinelglobal_active_round"] = 3

N_EXPERIMENTS = 1
if basic_config["accelerator"] == "gpu":
    EXPERIMENT_WAIT_SEC = 60 + 10 * basic_config["rounds"]
else:
    EXPERIMENT_WAIT_SEC = 60 + 10 * basic_config["epochs"] * basic_config["rounds"]


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
                    for poisoned_ratio in poisoned_ratio_list:

                        basic_config["attack"] = "Sample Poisoning"
                        basic_config["aggregation"] = aggregation
                        basic_config["poisoned_node_percent"] = node_percent
                        basic_config["poisoned_sample_percent"] = poisoned_sample_percent
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
