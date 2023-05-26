# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 


import os

from dotenv import load_dotenv

envpath = os.path.join(os.path.dirname(__file__), '../app/.env')
envpath = os.path.abspath(envpath)
load_dotenv(envpath)


###########################
#  Tests Infraestructure  #
###########################


def test_env():
    # Load the environment variables
    envpath = os.path.join(os.path.dirname(__file__), '../app/.env')
    envpath = os.path.abspath(envpath)
    print(envpath)
    load_dotenv(envpath)
    print(os.environ.get("MENDER_TOKEN"))


def test_mender_get_users():
    from fedstellar.config.mender import Mender
    mender = Mender()
    mender.get_my_info()
    mender.get_devices_by_group("Cluster_Thun")

def test_mender_get_info_device():
    from fedstellar.config.mender import Mender
    mender = Mender()
    mender.get_info_device("ededededededed")
    mender.get_info_device("9814a04b-6bdc-4875-8e2f-654db387d5ea")

def test_mender_get_connected_device():
    from fedstellar.config.mender import Mender
    mender = Mender()
    mender.get_connected_device("9814a04b-6bdc-4875-8e2f-654db387d5ea")

def test_mender_get_artifacts():
    from fedstellar.config.mender import Mender
    mender = Mender()
    mender.get_artifacts()

def test_mender_upload_artifact():
    from fedstellar.config.mender import Mender
    mender = Mender()
    # /Users/enrique/Documents/PhD/fedstellar/app/my-update-2.0.mender
    # Load and execute the following script in the device
    # #!/bin/bash
    #
    # mkdir /home/${USER}/fedstellar
    # wget https://raw.githubusercontent.com/enriquetomasmb/mender/main/client.py -P /home/${USER}/fedstellar
    # python3 /home/${USER}/fedstellar/client.py
    # echo "DFL" > /tmp/dfl.txt
    mender.upload_artifact(artifact_path="/app/my-update-2.0.mender", description="Artifact created from Fedstellar framework")

def test_mender_deploy_artifact():
    from fedstellar.config.mender import Mender
    mender = Mender()
    mender.deploy_artifact_device("my-update-2.0", "9814a04b-6bdc-4875-8e2f-654db387d5ea")

def test_mender_get_info_deployment():
    from fedstellar.config.mender import Mender
    mender = Mender()
    mender.get_info_deployment("4339a9b7-a601-4915-bfd1-6bdec9657b9b")

