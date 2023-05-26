import glob
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime

from dotenv import load_dotenv

from fedstellar.config.config import Config
from fedstellar.config.mender import Mender
from fedstellar.utils.topologymanager import TopologyManager

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Setup controller logger
class TermEscapeCodeFormatter(logging.Formatter):
    """A class to strip the escape codes from the """

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        escape_re = re.compile(r'\x1b\[[0-9;]*m')
        record.msg = re.sub(escape_re, "", str(record.msg))
        return super().format(record)


log_console_format = "[%(levelname)s] - %(asctime)s - Controller - %(message)s"
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter(log_console_format))
console_handler.setFormatter(TermEscapeCodeFormatter(log_console_format))
logging.basicConfig(level=logging.DEBUG,
                    handlers=[
                        console_handler,
                    ])


# Detect ctrl+c and run killports
def signal_handler(sig, frame):
    logging.info('You pressed Ctrl+C!')
    logging.info('Finishing all scenarios and nodes...')
    Controller.killports("tensorboa")
    Controller.killports("python")
    Controller.killdockers()
    # if sys.platform == "darwin":
    #     os.system("""osascript -e 'tell application "Terminal" to quit'""")
    # elif sys.platform == "linux":
    #     # Kill all python processes
    #     os.system("""killall python""")
    # else:
    #     os.system("""taskkill /IM cmd.exe /F""")
    #     os.system("""taskkill /IM powershell.exe /F""")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Controller:
    """
    Controller class that manages the nodes
    """

    def __init__(self, args):
        self.scenario_name = args.scenario_name if hasattr(args, "scenario_name") else None
        self.start_date_scenario = None
        self.cloud = args.cloud if hasattr(args, 'cloud') else None
        self.federation = args.federation
        self.topology = args.topology
        self.webserver = args.webserver
        self.webserver_port = args.webport if hasattr(args, "webport") else 5000
        self.statistics_port = args.statsport if hasattr(args, "statsport") else 5100
        self.simulation = args.simulation
        self.docker = args.docker if hasattr(args, 'docker') else None
        self.config_dir = args.config
        self.log_dir = args.logs
        self.model_dir = args.models
        self.env_path = args.env
        self.python_path = args.python
        self.matrix = args.matrix if hasattr(args, 'matrix') else None

        # Network configuration (nodes deployment in a network)
        self.network_subnet = args.network_subnet if hasattr(args, 'network_subnet') else None
        self.network_gateway = args.network_gateway if hasattr(args, 'network_gateway') else None

        self.config = Config(entity="controller")
        self.topologymanager = None
        self.n_nodes = 0
        self.mender = None if self.simulation else Mender()

    def start(self):
        """
        Start the controller
        """
        # First, kill all the ports related to previous executions
        # self.killports()

        banner = """
                            ______       _     _       _ _            
                            |  ___|     | |   | |     | | |           
                            | |_ ___  __| |___| |_ ___| | | __ _ _ __ 
                            |  _/ _ \/ _` / __| __/ _ \ | |/ _` | '__|
                            | ||  __/ (_| \__ \ ||  __/ | | (_| | |   
                            \_| \___|\__,_|___/\__\___|_|_|\__,_|_|   
                         Framework for Decentralized Federated Learning 
                       Enrique Tomás Martínez Beltrán (enriquetomas@um.es)
                    """
        print("\x1b[0;36m" + banner + "\x1b[0m")

        # Load the environment variables
        load_dotenv(self.env_path)

        # Save the configuration in environment variables
        logging.info("Saving configuration in environment variables...")
        os.environ["FEDSTELLAR_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["FEDSTELLAR_LOGS_DIR"] = self.log_dir
        os.environ["FEDSTELLAR_CONFIG_DIR"] = self.config_dir
        os.environ["FEDSTELLAR_MODELS_DIR"] = self.model_dir
        os.environ["FEDSTELLAR_PYTHON_PATH"] = self.python_path
        os.environ["FEDSTELLAR_STATISTICS_PORT"] = str(self.statistics_port)

        if self.webserver:
            self.run_webserver()
            self.run_statistics()
        else:
            logging.info("The controller without webserver is under development. Please, use the webserver (--webserver) option.")
            # self.load_configurations_and_start_nodes()
            if self.mender:
                logging.info("[Mender.module] Mender module initialized")
                time.sleep(2)
                mender = Mender()
                logging.info("[Mender.module] Getting token from Mender server: {}".format(os.getenv("MENDER_SERVER")))
                mender.renew_token()
                time.sleep(2)
                logging.info("[Mender.module] Getting devices from {} with group Cluster_Thun".format(os.getenv("MENDER_SERVER")))
                time.sleep(2)
                devices = mender.get_devices_by_group("Cluster_Thun")
                logging.info("[Mender.module] Getting a pool of devices: 5 devices")
                # devices = devices[:5]
                for i in self.config.participants:
                    logging.info("[Mender.module] Device {} | IP: {}".format(i['device_args']['idx'], i['network_args']['ipdemo']))
                    logging.info("[Mender.module] \tCreating artifacts...")
                    logging.info("[Mender.module] \tSending Fedstellar framework...")
                    # mender.deploy_artifact_device("my-update-2.0.mender", i['device_args']['idx'])
                    logging.info("[Mender.module] \tSending configuration...")
                    time.sleep(5)
            sys.exit(0)

        logging.info('Press Ctrl+C for exit from Fedstellar (global exit)')
        while True:
            time.sleep(1)

    def run_webserver(self):
        if sys.platform == "linux" and self.cloud:
            # Check if gunicon is installed
            try:
                subprocess.check_output(["gunicorn", "--version"])
            except FileNotFoundError:
                logging.error("Gunicorn is not installed. Please, install it with pip install gunicorn (only for Linux)")
                sys.exit(1)

            logging.info(f"Running Fedstellar Webserver (cloud): http://127.0.0.1:{self.webserver_port}")
            controller_env = os.environ.copy()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            webserver_path = os.path.join(current_dir, "webserver")
            with open(f'{self.log_dir}/server.log', 'w', encoding='utf-8') as log_file:
                # Remove option --reload for production
                subprocess.Popen(["gunicorn", "--workers", "4", "--threads", "4", "--bind", f"unix:/tmp/fedstellar.sock", "--access-logfile", f"{self.log_dir}/server.log", "app:app"], cwd=webserver_path, env=controller_env, stdout=log_file, stderr=log_file, encoding='utf-8')

        else:
            logging.info(f"Running Fedstellar Webserver (local): http://127.0.0.1:{self.webserver_port}")
            controller_env = os.environ.copy()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            webserver_path = os.path.join(current_dir, "webserver")
            with open(f'{self.log_dir}/server.log', 'w', encoding='utf-8') as log_file:
                subprocess.Popen([self.python_path, "app.py", "--port", str(self.webserver_port)], cwd=webserver_path, env=controller_env, stdout=log_file, stderr=log_file, encoding='utf-8')

    def run_statistics(self):
        import tensorboard
        import zipfile
        import warnings
        # Ignore warning from zipfile
        warnings.filterwarnings("ignore", category=UserWarning)
        # Get the tensorboard path
        tensorboard_path = os.path.dirname(tensorboard.__file__)
        # Include "index.html" in a zip file "webfiles.zip" which is in the tensorboard root folder. If the file "index.html" exists in the zip, it will be overwritten.
        with zipfile.ZipFile(os.path.join(tensorboard_path, "webfiles.zip"), "a") as zip:
            zip.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "webserver", "config", "statistics", "index.html"), "index.html")
            zip.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "webserver", "config", "statistics", "index.js"), "index.js")

        logging.info(f"Running Fedstellar Statistics")
        controller_env = os.environ.copy()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        webserver_path = os.path.join(current_dir, "webserver")
        with open(f'{self.log_dir}/statistics_server.log', 'w', encoding='utf-8') as log_file:
            subprocess.Popen(["tensorboard", "--host", "0.0.0.0", "--port", str(self.statistics_port), "--logdir", self.log_dir, "--reload_interval", "1", "--window_title", "Fedstellar Statistics"], cwd=webserver_path, env=controller_env, stdout=log_file, stderr=log_file, encoding='utf-8')

    @staticmethod
    def killports(term="python"):
        # kill all the ports related to python processes
        time.sleep(1)
        # Remove process related to tensorboard

        if sys.platform == "darwin":
            command = '''kill -9 $(lsof -i @localhost:1024-65545 | grep ''' + term + ''' | awk '{print $2}') > /dev/null 2>&1'''
        elif sys.platform == "linux":
            command = '''kill -9 $(lsof -i @localhost:1024-65545 | grep ''' + term + ''' | awk '{print $2}') > /dev/null 2>&1'''
        else:
            command = '''taskkill /F /IM ''' + term + '''.exe > nul 2>&1'''

        os.system(command)

    @staticmethod
    def killport(port):
        time.sleep(1)
        if sys.platform == "darwin":
            command = '''kill -9 $(lsof -i @localhost:''' + str(port) + ''' | grep python | awk '{print $2}') > /dev/null 2>&1'''
        elif sys.platform == "linux":
            command = '''kill -9 $(lsof -i :''' + str(port) + ''' | grep python | awk '{print $2}') > /dev/null 2>&1'''
        elif sys.platform == "win32":
            command = 'taskkill /F /PID $(FOR /F "tokens=5" %P IN (\'netstat -a -n -o ^| findstr :' + str(port) + '\') DO echo %P)'
        else:
            raise ValueError("Unknown platform")

        os.system(command)

    @staticmethod
    def killdockers():
        try:
            # kill all the docker containers which contain the word "fedstellar"
            command = '''docker kill $(docker ps -q --filter ancestor=fedstellar) > /dev/null 2>&1'''
            time.sleep(1)
            os.system(command)
            command = '''docker rm $(docker ps -a -q --filter ancestor=fedstellar) > /dev/null 2>&1'''
            time.sleep(1)
            os.system(command)

            # the same but for fedstellar-gpu
            command = '''docker kill $(docker ps -q --filter ancestor=fedstellar-gpu) > /dev/null 2>&1'''
            time.sleep(1)
            os.system(command)
            command = '''docker rm $(docker ps -a -q --filter ancestor=fedstellar-gpu) > /dev/null 2>&1'''
            time.sleep(1)
            os.system(command)

            # remove all docker networks which contain the word "fedstellar"
            command = '''docker network rm $(docker network ls | grep fedstellar | awk '{print $1}') > /dev/null 2>&1'''
            time.sleep(1)
            os.system(command)



        except Exception as e:
            raise Exception("Error while killing docker containers: {}".format(e))

    def load_configurations_and_start_nodes(self):
        if not self.scenario_name:
            self.scenario_name = f'fedstellar_{self.federation}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
        # Once the scenario_name is defined, we can update the config_dir
        self.config_dir = os.path.join(self.config_dir, self.scenario_name)
        os.makedirs(self.config_dir, exist_ok=True)

        os.makedirs(os.path.join(self.log_dir, self.scenario_name), exist_ok=True)
        self.model_dir = os.path.join(self.model_dir, self.scenario_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.start_date_scenario = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        logging.info("Generating the scenario {} at {}".format(self.scenario_name, self.start_date_scenario))

        # Get participants configurations
        print("Loading participants configurations...")
        print(self.config_dir)
        participant_files = glob.glob('{}/participant_*.json'.format(self.config_dir))
        participant_files.sort()
        if len(participant_files) == 0:
            raise ValueError("No participant files found in config folder")

        self.config.set_participants_config(participant_files)
        self.n_nodes = len(participant_files)
        logging.info("Number of nodes: {}".format(self.n_nodes))

        self.topologymanager = self.create_topology(matrix=self.matrix) if self.matrix else self.create_topology()

        # Update participants configuration
        is_start_node, idx_start_node = False, 0
        for i in range(self.n_nodes):
            with open(f'{self.config_dir}/participant_' + str(i) + '.json') as f:
                participant_config = json.load(f)
            participant_config['scenario_args']["federation"] = self.federation
            participant_config['scenario_args']['n_nodes'] = self.n_nodes
            participant_config['network_args']['neighbors'] = self.topologymanager.get_neighbors_string(i)
            participant_config['scenario_args']['name'] = self.scenario_name
            participant_config['scenario_args']['start_time'] = self.start_date_scenario
            participant_config['device_args']['idx'] = i
            participant_config['device_args']['uid'] = hashlib.sha1((str(participant_config["network_args"]["ip"]) + str(participant_config["network_args"]["port"]) + str(self.scenario_name)).encode()).hexdigest()
            participant_config['geo_args']['latitude'], participant_config['geo_args']['longitude'] = TopologyManager.get_coordinates(random_geo=True)

            participant_config['tracking_args']['log_dir'] = self.log_dir
            participant_config['tracking_args']['config_dir'] = self.config_dir
            participant_config['tracking_args']['model_dir'] = self.model_dir
            if participant_config["device_args"]["start"]:
                if not is_start_node:
                    is_start_node = True
                    idx_start_node = i
                else:
                    raise ValueError("Only one node can be start node")
            with open(f'{self.config_dir}/participant_' + str(i) + '.json', 'w') as f:
                json.dump(participant_config, f, sort_keys=False, indent=2)
        if not is_start_node:
            raise ValueError("No start node found")
        self.config.set_participants_config(participant_files)

        # Add role to the topology (visualization purposes)
        self.topologymanager.update_nodes(self.config.participants)
        self.topologymanager.draw_graph(path=f"{self.log_dir}/{self.scenario_name}/topology.png", plot=False)

        if self.simulation:
            if self.docker:
                self.start_nodes_docker(idx_start_node)
            else:
                self.start_nodes_cmd(idx_start_node)
        else:
            logging.info("Simulation mode is disabled, waiting for nodes to start...")

    def create_topology(self, matrix=None):
        import numpy as np
        if matrix is not None:
            if self.n_nodes > 2:
                topologymanager = TopologyManager(topology=np.array(matrix), scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True, undirected_neighbor_num=self.n_nodes - 1)
            else:
                topologymanager = TopologyManager(topology=np.array(matrix), scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True, undirected_neighbor_num=2)
        elif self.topology == "fully":
            # Create a fully connected network
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True, undirected_neighbor_num=self.n_nodes - 1)
            topologymanager.generate_topology()
        elif self.topology == "ring":
            # Create a partially connected network (ring-structured network)
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_ring_topology(increase_convergence=True)
        elif self.topology == "random":
            # Create network topology using topology manager (random)
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True,
                                              undirected_neighbor_num=3)
            topologymanager.generate_topology()
        elif self.topology == "star" and self.federation == "CFL":
            # Create a centralized network
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_server_topology()
        else:
            raise ValueError("Unknown topology type: {}".format(self.topology))

        # Assign nodes to topology
        nodes_ip_port = []
        for i, node in enumerate(self.config.participants):
            nodes_ip_port.append((node['network_args']['ip'], node['network_args']['port'], "undefined", node['network_args']['ipdemo']))

        topologymanager.add_nodes(nodes_ip_port)
        return topologymanager

    def start_nodes_docker(self, idx_start_node):
        logging.info("Starting nodes using Docker Compose...")
        docker_compose_template = textwrap.dedent("""
            services:
            {}
        """)

        participant_template = textwrap.dedent("""
            participant{}:
                image: fedstellar
                restart: always
                volumes:
                    - {}:/fedstellar
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.8 /fedstellar/fedstellar/node_start.py {}
                depends_on:
                    - participant{}
                networks:
                    fedstellar-net:
                        ipv4_address: {}
        """)
        participant_template = textwrap.indent(participant_template, ' ' * 4)

        participant_template_start = textwrap.dedent("""
            participant{}:
                image: fedstellar
                restart: always
                volumes:
                    - {}:/fedstellar
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        /bin/sleep 60 && ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.8 /fedstellar/fedstellar/node_start.py {}
                networks:
                    fedstellar-net:
                        ipv4_address: {}
        """)
        participant_template_start = textwrap.indent(participant_template_start, ' ' * 4)

        participant_gpu_template = textwrap.dedent("""
            participant{}:
                image: fedstellar-gpu
                restart: always
                volumes:
                    - {}:/fedstellar
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.8 /fedstellar/fedstellar/node_start.py {}
                depends_on:
                    - participant{}
                deploy:
                    resources:
                        reservations:
                            devices:
                                - driver: nvidia
                                  count: all
                                  capabilities: [gpu]
                networks:
                    fedstellar-net:
                        ipv4_address: {}
        """)
        participant_gpu_template = textwrap.indent(participant_gpu_template, ' ' * 4)

        participant_gpu_template_start = textwrap.dedent("""
            participant{}:
                image: fedstellar-gpu
                restart: always
                volumes:
                    - {}:/fedstellar
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        /bin/sleep 60 && ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.8 /fedstellar/fedstellar/node_start.py {}
                deploy:
                    resources:
                        reservations:
                            devices:
                                - driver: nvidia
                                  count: all
                                  capabilities: [gpu]
                networks:
                    fedstellar-net:
                        ipv4_address: {}
        """)
        participant_gpu_template_start = textwrap.indent(participant_gpu_template_start, ' ' * 4)

        network_template = textwrap.dedent("""
            networks:
                fedstellar-net:
                    driver: bridge
                    ipam:
                        config:
                            - subnet: {}
                              gateway: {}
        """)

        # Generate the Docker Compose file dynamically
        services = ""
        self.config.participants.sort(key=lambda x: x['device_args']['idx'])
        for node in self.config.participants:
            idx = node['device_args']['idx']
            path = f"/fedstellar/app/config/{self.scenario_name}/participant_{idx}.json"
            logging.info("Starting node {} with configuration {}".format(idx, path))
            logging.info("Node {} is listening on ip {}".format(idx, node['network_args']['ip']))
            # Add one service for each participant
            if idx != idx_start_node:
                if node['device_args']['accelerator'] == "gpu":
                    logging.info("Node {} is using GPU".format(idx))
                    services += participant_gpu_template.format(idx,
                                                                os.environ["FEDSTELLAR_ROOT"],
                                                                self.network_gateway,
                                                                path,
                                                                idx_start_node,
                                                                node['network_args']['ip'])
                else:
                    logging.info("Node {} is using CPU".format(idx))
                    services += participant_template.format(idx,
                                                            os.environ["FEDSTELLAR_ROOT"],
                                                            self.network_gateway,
                                                            path,
                                                            idx_start_node,
                                                            node['network_args']['ip'])
            else:  # Start the node with a delay of 60 seconds
                if node['device_args']['accelerator'] == "gpu":
                    logging.info("Node {} is using GPU".format(idx))
                    services += participant_gpu_template_start.format(idx,
                                                                      os.environ["FEDSTELLAR_ROOT"],
                                                                      self.network_gateway,
                                                                      path,
                                                                      node['network_args']['ip'])
                else:
                    logging.info("Node {} is using CPU".format(idx))
                    services += participant_template_start.format(idx,
                                                                  os.environ["FEDSTELLAR_ROOT"],
                                                                  self.network_gateway,
                                                                  path,
                                                                  node['network_args']['ip'])
        docker_compose_file = docker_compose_template.format(services)
        docker_compose_file += network_template.format(self.network_subnet, self.network_gateway)
        # Write the Docker Compose file in config directory
        with open(f"{self.config_dir}/docker-compose.yml", "w") as f:
            f.write(docker_compose_file)

        # Change log and config directory in dockers to /fedstellar/app, and change controller endpoint
        for node in self.config.participants:
            node['tracking_args']['log_dir'] = "/fedstellar/app/logs"
            node['tracking_args']['config_dir'] = f"/fedstellar/app/config/{self.scenario_name}"
            node['tracking_args']['model_dir'] = f"/fedstellar/app/models/{self.scenario_name}"
            if sys.platform == "linux":
                node['scenario_args']['controller'] = "host.docker.internal" + ":" + str(self.webserver_port)
            elif sys.platform == "darwin":
                node['scenario_args']['controller'] = "host.docker.internal" + ":" + str(self.webserver_port)
            else:
                node['scenario_args']['controller'] = "host.docker.internal" + ":" + str(self.webserver_port)

            # Write the config file in config directory
            with open(f"{self.config_dir}/participant_{node['device_args']['idx']}.json", "w") as f:
                json.dump(node, f, indent=4)
        # Start the Docker Compose file, catch error if any
        try:
            subprocess.check_call(["docker", "compose", "-f", f"{self.config_dir}/docker-compose.yml", "up", "-d"])
        except subprocess.CalledProcessError as e:
            logging.error("Docker Compose failed to start, please check if Docker is running and Docker Compose is installed.")
            logging.error(e)
            raise e

    def start_node(self, idx):
        command = f'cd {os.path.dirname(os.path.realpath(__file__))}; {self.python_path} -u node_start.py {str(self.config.participants_path[idx])} 2>&1'
        print("Starting node {} with command: {}".format(idx, command))
        if sys.platform == "darwin":
            print("MacOS detected")
            os.system("""osascript -e 'tell application "Terminal" to activate' -e 'tell application "Terminal" to do script "{}"'""".format(command))
        elif sys.platform == "linux":
            print("Linux OS detected")
            command = f'{self.python_path} -u {os.path.dirname(os.path.realpath(__file__))}/node_start.py {str(self.config.participants_path[idx])}'
            os.system(command + " 2>&1 &")
        elif sys.platform == "win32":
            print("Windows OS detected")
            command_win = f'cd {os.path.dirname(os.path.realpath(__file__))} {str("&&")} {self.python_path} -u node_start.py {str(self.config.participants_path[idx])} 2>&1'
            os.system("""start cmd /k "{}" """.format(command_win))
        else:
            raise ValueError("Unknown operating system")

    def start_nodes_cmd(self, idx_start_node):
        # Start the nodes
        # Get directory path of the current file
        for idx in range(0, self.n_nodes):
            if idx == idx_start_node:
                continue
            logging.info("Starting node {} with configuration {}".format(idx, self.config.participants[idx]))
            self.start_node(idx)

        time.sleep(3)
        # Start the node with start flag
        logging.info("Starting node {} with configuration {}".format(idx_start_node, self.config.participants[idx_start_node]))
        self.start_node(idx_start_node)

    @classmethod
    def remove_files_by_scenario(cls, scenario_name):
        import shutil
        shutil.rmtree(os.path.join(os.environ["FEDSTELLAR_CONFIG_DIR"], scenario_name))
        shutil.rmtree(os.path.join(os.environ["FEDSTELLAR_MODELS_DIR"], scenario_name))
        try:
            shutil.rmtree(os.path.join(os.environ["FEDSTELLAR_LOGS_DIR"], scenario_name))
        except PermissionError:
            # Avoid error if the user does not have enough permissions to remove the tf.events files
            logging.warning("Not enough permissions to remove the files, moving them to tmp folder")
            os.makedirs(os.path.join(os.environ["FEDSTELLAR_ROOT"], "app", "tmp", scenario_name), exist_ok=True)
            shutil.move(os.path.join(os.environ["FEDSTELLAR_LOGS_DIR"], scenario_name), os.path.join(os.environ["FEDSTELLAR_ROOT"], "app", "tmp", scenario_name))
        except FileNotFoundError:
            logging.warning("Files not found, nothing to remove")
        except Exception as e:
            logging.error("Unknown error while removing files")
            logging.error(e)
            raise e
