# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
import socket
import time

import pytest

from fedstellar.base_node import BaseNode
from fedstellar.communication_protocol import CommunicationProtocol
from fedstellar.config.config import Config
from test.utils import set_test_settings, wait_network_nodes

set_test_settings()


@pytest.fixture
def two_nodes():
    n1 = BaseNode()
    n2 = BaseNode()
    n1.start()
    n2.start()

    yield n1, n2

    n1.stop()
    n2.stop()


@pytest.fixture
def four_nodes():
    n1 = BaseNode()
    n2 = BaseNode()
    n3 = BaseNode()
    n4 = BaseNode()
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    return n1, n2, n3, n4


###########################
#  Tests Infraestructure  #
###########################

def test_socket():
    host = socket.gethostbyname("192.168.0.7")
    port = 45001
    # Setting Up Node Socket (listening)
    __node_socket = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM
    )  # TCP Socket
    __node_socket.bind((host, port))
    __node_socket.listen(50)  # no more than 50 connections at queue

def test_node_paring(two_nodes):
    n1, n2 = two_nodes

    # Connect
    n1.connect_to(n2.host, n2.port)
    wait_network_nodes(two_nodes)
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == 1
    assert len(n1.get_network_nodes()) == len(n2.get_network_nodes()) == 2

    # Disconnect
    n2.disconnect_from(
        n1.host, n1.port
    )  # Direct disconnection are practically instantaneous, network nodes needs to wait timeout.
    time.sleep(Config.NODE_TIMEOUT + 1)
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == 0
    assert len(n1.get_network_nodes()) == 1


def test_connect_invalid_node():
    n = BaseNode()
    n.connect_to("google.es", "80")
    n.connect_to("um.es", "80")
    assert len(n.get_neighbors()) == 0


def test_full_connected(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Connect n1 n2
    n1.connect_to(n2.host, n2.port, full=True)
    time.sleep(0.1)
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == 1

    # Connect n3 n1
    n3.connect_to(n1.host, n1.port, full=True)
    time.sleep(0.1)
    assert (
            len(n1.get_neighbors())
            == len(n2.get_neighbors())
            == len(n3.get_neighbors())
            == 2
    )

    # Connect n4 n1
    n4.connect_to(n1.host, n1.port, full=True)
    wait_network_nodes(four_nodes)
    assert (
            len(n1.get_neighbors())
            == len(n2.get_neighbors())
            == len(n3.get_neighbors())
            == len(n4.get_neighbors())
            == 3
    )
    assert (
            len(n1.get_network_nodes())
            == len(n2.get_network_nodes())
            == len(n3.get_network_nodes())
            == len(n4.get_network_nodes())
            == 4
    )

    # Disconnect n1
    n1.stop()
    time.sleep(0.1)
    assert (
            len(n2.get_neighbors())
            == len(n3.get_neighbors())
            == len(n4.get_neighbors())
            == 2
    )

    # Disconnect n2
    n2.stop()
    time.sleep(0.1)
    assert len(n3.get_neighbors()) == len(n4.get_neighbors()) == 1

    # Disconnect n3
    n3.stop()
    time.sleep(0.1)
    assert len(n4.get_neighbors()) == 0

    # Disconnect n4
    n4.stop()


def test_non_full_connected(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Ring network
    n1.connect_to(n2.host, n2.port, full=False)
    n2.connect_to(n3.host, n3.port, full=False)
    n3.connect_to(n4.host, n4.port, full=False)
    n4.connect_to(n1.host, n1.port, full=False)

    # Wait
    wait_network_nodes(four_nodes)

    # Verify topology
    assert (
            len(n1.get_neighbors())
            == len(n2.get_neighbors())
            == len(n3.get_neighbors())
            == len(n4.get_neighbors())
            == 2
    )

    # Stop Nodes
    [n.stop() for n in four_nodes]


def test_multimsg(two_nodes):
    n1, n2 = two_nodes

    # Conexión
    n1.connect_to(n2.host, n2.port)
    time.sleep(0.1)

    msg = CommunicationProtocol.build_beat_msg(n1.get_name())

    n1.broadcast(msg + msg + msg)
    time.sleep(0.1)
    assert len(n2.get_neighbors()) == 1


def test_bad_msg(two_nodes):
    n1, n2 = two_nodes

    # Conexión
    n1.connect_to(n2.host, n2.port)
    time.sleep(0.1)

    # Create an error message
    n1.broadcast(b"saludos Enrique y Dani")
    time.sleep(0.1)
    assert len(n2.get_neighbors()) == 0


##############################
#    Test Fault Tolerance    #
##############################


def test_node_abrupt_down(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Connect n1 n2
    n1.connect_to(n2.host, n2.port, full=True)

    # Connect n3 n1
    n3.connect_to(n1.host, n1.port, full=True)
    time.sleep(0.1)
    assert (
            len(n1.get_neighbors())
            == len(n2.get_neighbors())
            == len(n3.get_neighbors())
            == 2
    )

    # Connect n4 n1
    n4.connect_to(n1.host, n1.port, full=True)
    time.sleep(0.1)
    assert (
            len(n1.get_neighbors())
            == len(n2.get_neighbors())
            == len(n3.get_neighbors())
            == len(n4.get_neighbors())
            == 3
    )

    # n4 (socket closed)
    for con in n4.get_neighbors():
        con._NodeConnection__socket.close()
    time.sleep(Config.HEARTBEAT_PERIOD + 0.5)
    assert (
            len(n1.get_neighbors())
            == len(n2.get_neighbors())
            == len(n3.get_neighbors())
            == 2
    )
    n4.stop()

    # n3 stops heartbeater
    n3.heartbeater.stop()
    n3.gossiper.stop()  # to avoid resending messages and refreshing socket timeout
    time.sleep(Config.NODE_TIMEOUT + 1)
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == 1
    assert len(n1.get_network_nodes()) == len(n2.get_network_nodes()) == 2
    n3.stop()

    # Disconnect n2 and n1
    n2.stop()
    n1.stop()
