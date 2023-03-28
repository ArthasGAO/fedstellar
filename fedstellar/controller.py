import glob
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
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
    # logging.info("Remove configuration and topology files...")
    # Controller.remove_config_files()
    # logging.info("Remove configuration and topology files... Done")
    logging.info('Finishing all scenarios and nodes...')
    Controller.killports("tensorboa")
    Controller.killports("python")
    if sys.platform == "darwin":
        os.system("""osascript -e 'tell application "Terminal" to quit'""")
    elif sys.platform == "linux":
        # Kill all python processes
        os.system("""killall python""")
    else:
        os.system("""taskkill /IM cmd.exe /F""")
        os.system("""taskkill /IM powershell.exe /F""")
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
        self.config_dir = args.config
        self.log_dir = args.logs
        self.env_path = args.env
        self.python_path = args.python
        self.matrix = args.matrix if hasattr(args, 'matrix') else None

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
        os.environ["FEDSTELLAR_LOGS_DIR"] = self.log_dir
        os.environ["FEDSTELLAR_CONFIG_DIR"] = self.config_dir
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

    def load_configurations_and_start_nodes(self):
        if not self.scenario_name:
            self.scenario_name = f'fedstellar_{self.federation}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
        # Once the scenario_name is defined, we can update the config_dir
        self.config_dir = os.path.join(self.config_dir, self.scenario_name)
        os.makedirs(self.config_dir, exist_ok=True)

        os.makedirs(os.path.join(self.log_dir, self.scenario_name), exist_ok=True)
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

        # topology_json_path = f"{self.config_dir}/topology.json"
        # self.topologymanager.update_topology_3d_json(participants=self.config.participants, path=topology_json_path)

        if self.simulation:
            self.start_nodes(idx_start_node)
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

    def start_nodes(self, idx_start_node):
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
    def remove_config_files(cls):
        # Remove all json and png files in the folder os.environ["FEDSTELLAR_CONFIG_DIR"]
        for file in glob.glob(os.path.join(os.environ["FEDSTELLAR_CONFIG_DIR"], "*.json")):
            os.remove(file)
        for file in glob.glob(os.path.join(os.environ["FEDSTELLAR_CONFIG_DIR"], "*.png")):
            os.remove(file)

    @classmethod
    def remove_files_by_scenario(cls, scenario_name):
        import shutil
        shutil.rmtree(os.path.join(os.environ["FEDSTELLAR_CONFIG_DIR"], scenario_name))
        shutil.rmtree(os.path.join(os.environ["FEDSTELLAR_LOGS_DIR"], scenario_name))
