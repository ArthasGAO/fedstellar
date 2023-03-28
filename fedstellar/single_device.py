import os
import time

from fedstellar.config.config import Config
from fedstellar.learning.pytorch.mnist.mnist import MNISTDataModule
from fedstellar.learning.pytorch.mnist.models.mlp import MLP
from fedstellar.node import Node

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def main():

    config = Config(entity="participant", participant_config_file="/fedstellar/config/participant_config_server.yaml")

    # node = Node(
    #     MLP(),
    #     MNISTDataModule(sub_id=0, number_sub=1, iid=True),
    #     config=config,
    #     rol="trainer",
    #     simulation=True,
    # )
    # node.start()
    # time.sleep(5)
    # node.set_start_learning(rounds=10, epochs=5)
    #
    # while True:
    #     time.sleep(1)
    #     finish = True
    #     for f in [n.round is None for n in [node]]:
    #         finish = finish and f
    #
    #     if finish:
    #         break
    #
    # for n in [node]:
    #     n.stop()


if __name__ == "__main__":
    main()
