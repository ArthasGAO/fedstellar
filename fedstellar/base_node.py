# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
import json
import logging
import os
import socket
import threading
from datetime import datetime
from logging import Formatter, FileHandler
from logging.handlers import RotatingFileHandler

from fedstellar.communication_protocol import CommunicationProtocol
from fedstellar.encrypter import AESCipher, RSACipher
from fedstellar.gossiper import Gossiper
from fedstellar.heartbeater import Heartbeater
from fedstellar.node_connection import NodeConnection
from fedstellar.utils.observer import Events, Observer


class BaseNode(threading.Thread, Observer):
    """
    This class represents a base node in the network (without **FL**). It is a thread, so it's going to process all messages in a background thread using the CommunicationProtocol.

    Args:
        host (str): The host of the node.
        port (int): The port of the node.
        simulation (bool): If False, communication will be encrypted.

    Attributes:
        host (str): The host of the node.
        port (int): The port of the node.
        simulation (bool): If the node is in simulation mode or not. Basically, simulation nodes don't have encryption and metrics aren't sent to network nodes.
        heartbeater (Heartbeater): The heartbeater of the node.
        gossiper (Gossiper): The gossiper of the node.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(self, experiment_name, hostdemo=None, host="127.0.0.1", port=None, encrypt=False, config=None):
        self.experiment_name = experiment_name
        # Node Attributes
        self.hostdemo = hostdemo
        self.host = socket.gethostbyname(host)
        self.port = port
        self.encrypt = encrypt
        self.simulation = config.participant["scenario_args"]["simulation"]
        self.config = config

        # Super init
        threading.Thread.__init__(self, name="node-" + self.get_name())
        self._terminate_flag = threading.Event()

        # Setting Up Node Socket (listening)
        self.__node_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )  # TCP Socket
        if port is None:
            self.__node_socket.bind((host, 0))  # gets a random free port
            self.port = self.__node_socket.getsockname()[1]
        else:
            print("[BASENODE] Trying to bind to {}:{}".format(host, port))
            self.__node_socket.bind((host, port))
        self.__node_socket.listen(50)  # no more than 50 connections at queue

        # Setting up network resources
        if not self.simulation and config.participant["network_args"]:
            print("[BASENODE] Network parameters\n{}".format(config.participant["network_args"]))
            print("[BASENODE] Running tcconfig to set network parameters")
            os.system(f"tcset --device {config.participant['network_args']['interface']} --rate {config.participant['network_args']['rate']} --delay {config.participant['network_args']['delay']} --delay-distro {config.participant['network_args']['delay-distro']} --loss {config.participant['network_args']['loss']}")

        # Neighbors
        self.__neighbors = []  # private to avoid concurrency issues
        self.__nei_lock = threading.Lock()

        # Logging
        self.log_dir = os.path.join(config.participant['tracking_args']["log_dir"], self.experiment_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_filename = f"{self.log_dir}/participant_{config.participant['device_args']['idx']}" if self.hostdemo else f"{self.log_dir}/participant_{config.participant['device_args']['idx']}"
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        console_handler, file_handler, file_handler_only_debug, exp_errors_file_handler = self.setup_logging(self.log_filename)

        level = logging.DEBUG if config.participant["scenario_args"]["debug"] else logging.WARNING
        logging.basicConfig(level=level,
                            handlers=[
                                console_handler,
                                file_handler,
                                file_handler_only_debug,
                                exp_errors_file_handler
                            ])

        # Heartbeater and Gossiper
        self.gossiper = None
        self.heartbeater = None

    def get_addr(self):
        """
        Returns:
            tuple: The address of the node.
        """
        return self.host, self.port

    def get_name(self):
        """
        Returns:
            str: The name of the node.
        """
        return str(self.get_addr()[0]) + ":" + str(self.get_addr()[1])

    def get_name_demo(self):
        """
        Returns:
            str: The name of the node.
        """
        return str(self.hostdemo) + ":" + str(self.get_addr()[1])

    def setup_logging(self, log_dir):
        CYAN = "\x1b[0;36m"
        RESET = "\x1b[0m"
        info_file_format = f"%(asctime)s - %(message)s"
        debug_file_format = f"%(asctime)s - %(message)s\n[in %(pathname)s:%(lineno)d]"
        log_console_format = f"{CYAN}[%(levelname)s] - %(asctime)s - {self.get_name_demo()}{RESET}\n%(message)s" if self.hostdemo else f"{CYAN}[%(levelname)s] - %(asctime)s - {self.get_name()}{RESET}\n%(message)s"

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(Formatter(log_console_format))

        file_handler = FileHandler('{}.log'.format(log_dir), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(Formatter(info_file_format))

        file_handler_only_debug = FileHandler('{}_debug.log'.format(log_dir), mode='w')
        file_handler_only_debug.setLevel(logging.DEBUG)
        # Add filter to file_handler_only_debug for only add debug messages
        file_handler_only_debug.addFilter(lambda record: record.levelno == logging.DEBUG)
        file_handler_only_debug.setFormatter(Formatter(debug_file_format))

        exp_errors_file_handler = FileHandler('{}_error.log'.format(log_dir), mode='w')
        exp_errors_file_handler.setLevel(logging.WARNING)
        exp_errors_file_handler.setFormatter(Formatter(debug_file_format))

        return console_handler, file_handler, file_handler_only_debug, exp_errors_file_handler

    #######################
    #   Node Management   #
    #######################

    def start(self):
        """
        Starts the node (node loop in a thread). It will listen for new connections and process them. Heartbeater and Gossiper will be started too.

        Note that a node is a thread, so an instance can only be started once.
        """
        # Main Loop
        super().start()
        # Heartbeater and Gossiper
        self.heartbeater = Heartbeater(self.get_name(), self.__neighbors, self.config)
        self.gossiper = Gossiper(
            self.get_name(), self.__neighbors, self.config
        )  # thread safe, only read
        self.heartbeater.add_observer(self)
        self.gossiper.add_observer(self)
        self.heartbeater.start()
        self.gossiper.start()

    def stop(self):
        """
        Stops the node. Heartbeater and Gossiper will be stopped too.
        """
        self._terminate_flag.set()
        try:
            # Send a self message to the loop to avoid the wait of the next recv
            self.__send(self.host, self.port, b"")
        except Exception as e:
            pass

    ########################
    #   Main Thread Loop   #
    ########################

    def run(self):
        """
        Main loop of the node, when a node is running, this method is being executed. It will listen for new connections and process them.
        """
        # Process new connections loop
        logging.info("[BASENODE] Node started")
        while not self._terminate_flag.is_set():
            try:
                (ns, _) = self.__node_socket.accept()
                msg = ns.recv(self.config.participant["BLOCK_SIZE"])

                # Process new connection
                if msg:
                    msg = msg.decode("UTF-8")
                    callback = lambda h, p, fu, fc: self.__process_new_connection(
                        ns, h, p, fu, fc
                    )
                    if not CommunicationProtocol.process_connection(msg, callback):
                        ns.close()
            except Exception as e:
                logging.exception(e)

        # Stop Heartbeater and Gossiper
        self.heartbeater.stop()
        self.gossiper.stop()

        # Stop Node
        logging.info(
            "[BASENODE] Stopping node. Disconnecting from {} nodos".format(
                len(self.__neighbors)
            )
        )
        nei_copy_list = self.get_neighbors()
        for n in nei_copy_list:
            n.stop()
        self.__node_socket.close()

    def __process_new_connection(self, node_socket, h, p, full, force):
        try:
            # Check if connection with the node already exist
            self.__nei_lock.acquire()
            if self.get_neighbor(h, p, thread_safe=False) is None:

                # Check if ip and port are correct
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                result = s.connect_ex((h, p))
                s.close()

                # Encryption
                aes_cipher = None
                if self.encrypt:
                    # Asymmetric
                    rsa = RSACipher()
                    node_socket.sendall(rsa.get_key())
                    rsa.load_pair_public_key(node_socket.recv(len(rsa.get_key())))

                    # Symmetric
                    aes_cipher = AESCipher()
                    node_socket.sendall(aes_cipher.get_key())

                # Add neighbor
                if result == 0:
                    logging.info(
                        "{} Connection accepted with {}:{}".format(
                            self.get_name(), h, p
                        )
                    )
                    nc = NodeConnection(
                        self.get_name(), node_socket, (h, p), aes_cipher, config=self.config
                    )
                    nc.add_observer(self)
                    logging.info("[BASENODE.__process_new_connection] New neighbor: {}".format(nc.get_name()))
                    self.__neighbors.append(nc)
                    nc.start(force=force)

                    if full:
                        self.broadcast(
                            CommunicationProtocol.build_connect_to_msg(h, p),
                            exc=[nc],
                            thread_safe=False,
                        )
            else:
                node_socket.close()

            self.__nei_lock.release()

        except Exception as e:
            logging.info(
                "[BASENODE] Connection refused with {}:{}".format(h, p)
            )
            self.__nei_lock.release()
            node_socket.close()
            self.rm_neighbor(nc)

    #############################
    #  Neighborhood management  #
    #############################

    # Create a tcp socket and send data
    @staticmethod
    def __send(h, p, data, persist=False):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((h, p))
        s.sendall(data)
        if persist:
            return s
        else:
            s.close()
            return None

    def connect_to(self, h, p, full=False, force=False):
        """
        Connects a node to another.

        Args:
            h (str): The host of the node.
            p (int): The port of the node.
            full (bool): If True, the node will be connected to the entire network.
            force (bool): If True, the node will be connected even though it should not be.

        Returns:
            node: The node that has been connected to.
        """
        # logging.debug(inspect.stack())
        # logging.debug("[BASENODE_connect_to] Checking stack (traceback)", stack_info=True)

        if full:
            full = "1"
        else:
            full = "0"

        if force:
            force = "1"
        else:
            force = "0"

        try:
            # Check if connection with the node already exist
            h = socket.gethostbyname(h)
            self.__nei_lock.acquire()
            if self.get_neighbor(h, p, thread_safe=False) is None:

                # Send connection request
                msg = CommunicationProtocol.build_connect_msg(
                    self.host, self.port, full, force
                )
                s = self.__send(h, p, msg, persist=True)

                # Encryption
                aes_cipher = None
                if not self.simulation:
                    # Asymmetric
                    rsa = RSACipher()
                    rsa.load_pair_public_key(s.recv(len(rsa.get_key())))
                    s.sendall(rsa.get_key())
                    # Symmetric
                    aes_cipher = AESCipher(key=s.recv(AESCipher.key_len()))

                # Add socket to neighbors
                nc = NodeConnection(self.get_name(), s, (h, p), aes_cipher, config=self.config)
                nc.add_observer(self)
                logging.info("[BASENODE_connect_to] Connected to {}:{} -> New neighbor {}".format(h, p, nc.get_name()))
                self.__neighbors.append(nc)
                nc.start(force=force)
                self.__nei_lock.release()
                return nc

            else:
                logging.info(
                    "{} Already connected to {}:{}".format(self.get_name(), h, p)
                )
                self.__nei_lock.release()
                return None
        except Exception as e:
            logging.info(
                "{} Can't connect to the node {}:{}".format(self.get_name(), h, p)
            )
            # logging.exception(e)
            try:
                self.__nei_lock.release()
            except Exception as e:
                pass
            return None

    def disconnect_from(self, h, p):
        """
        Disconnects from a node.

        Args:
            h (str): The host of the node.
            p (int): The port of the node.
        """
        self.get_neighbor(h, p).stop()

    def get_neighbor(self, h, p, thread_safe=True):
        """
         Get a ``NodeConnection`` from the neighbors list.

         Args:
             h (str): The host of the node.
             p (int): The port of the node.
            thread_safe (bool): If True, the method will be thread safe.

         Returns:
             NodeConnection: The connection with the node.
         """

        if thread_safe:
            self.__nei_lock.acquire()

        return_node = None
        for n in self.__neighbors:
            if n.get_addr() == (h, p):
                return_node = n
                break

        if thread_safe:
            self.__nei_lock.release()

        return return_node

    def get_neighbors(self):
        """
        Returns:
            list: The neighbors of the node.
        """
        self.__nei_lock.acquire()
        n = self.__neighbors.copy()
        self.__nei_lock.release()
        return n

    def get_neighbors_names(self):
        """
        Returns:
            list: The names of the neighbors of the node.
        """
        self.__nei_lock.acquire()
        n = [nc.get_name() for nc in self.__neighbors]
        self.__nei_lock.release()
        return n

    def rm_neighbor(self, n):
        """
        Removes a neighbor from the neighbors list.

        Args:
            n (NodeConnection): The neighbor to be removed.
        """
        self.__nei_lock.acquire()
        try:
            logging.info("[BASENODE.rm_neighbor] Remove neighbor: {}".format(n.get_name()))
            self.__neighbors.remove(n)
            n.stop()
        except Exception as e:
            pass
        self.__nei_lock.release()

    def get_network_nodes(self):
        """
        Returns:
            list: The nodes of the network -> The neighbors of the node (by heartbeater).
        """
        return self.heartbeater.get_nodes()

    ##########################
    #     Msg management     #
    ##########################

    def broadcast(self, msg, exc=[], thread_safe=True):
        """
        Broadcasts a message to all the neighbors.

        Args:
            msg (str): The message to be broadcast.
            exc (list): The neighbors to be excluded.
            thread_safe (bool): If True, the broadcast will access the neighbors list in a thread safe mode.

        """
        if thread_safe:
            self.__nei_lock.acquire()

        logging.debug("[BASENODE.broadcast] {} --> to: {} | Excluded: {}".format(msg, self.__neighbors, exc))

        for n in self.__neighbors:
            if not (n in exc):
                n.send(msg)

        if thread_safe:
            self.__nei_lock.release()

    ###########################
    #     Observer Events     #
    ###########################

    def update(self, event, obj):
        """
        Observer update method. Used to handle events that can occur with the different components and connections of the node.

        Args:
            event (Events): Event that has occurred.
            obj: Information about the change or event.
        """
        if len(str(obj)) > 300:
            logging.debug("[BASENODE.update (observer)] Event that has occurred: {} | Obj information: Too long [...]".format(event))
        else:
            logging.debug("[BASENODE.update (observer)] Event that has occurred: {} | Obj information: {}".format(event, obj))

        if event == Events.END_CONNECTION_EVENT:
            self.rm_neighbor(obj)

        elif event == Events.NODE_CONNECTED_EVENT:
            # Este evento lo notifica NodeConnection. Previamente se ha tenido que conectar con el nodo.
            logging.debug("[BASENODE.update (observer) | Events.NODE_CONNECTED_EVENT] Connecting to: {}".format(obj[0]))
            n, _ = obj
            n.send(CommunicationProtocol.build_beat_msg(self.get_name()))

        elif event == Events.CONN_TO_EVENT:
            logging.debug("[BASENODE.update (observer) | Events.CONN_TO_EVENT] Connecting to: {} {}".format(obj[0], obj[1]))
            self.connect_to(obj[0], obj[1], full=False)

        elif event == Events.SEND_BEAT_EVENT:
            self.broadcast(CommunicationProtocol.build_beat_msg(self.get_name()))

        elif event == Events.GOSSIP_BROADCAST_EVENT:
            self.broadcast(obj[0], exc=obj[1])

        elif event == Events.PROCESSED_MESSAGES_EVENT:
            node, msgs = obj
            # Communicate to connections the new messages processed
            for nc in self.__neighbors:
                if nc != node:
                    nc.add_processed_messages(list(msgs.keys()))
            # Gossip the new messages
            if len(str(obj)) > 300:
                logging.debug("[BASENODE.update (observer) | Events.PROCESSED_MESSAGES_EVENT] Add messages to gossiper: Too long [...] | Node: {}".format(node))
            else:
                logging.debug("[BASENODE.update (observer) | Events.PROCESSED_MESSAGES_EVENT] Add messages to gossiper: {} | Node: {}".format(list(msgs.values()), node))
            self.gossiper.add_messages(list(msgs.values()), node)

        elif event == Events.BEAT_RECEIVED_EVENT:
            # Update the heartbeater with the active neighbor
            self.heartbeater.add_node(obj)
