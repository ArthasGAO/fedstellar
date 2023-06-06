import os
import time
from datetime import datetime

from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule

from fedstellar.node import Node
from fedstellar.learning.pytorch.datamodule import DataModule
from fedstellar.learning.pytorch.mnist.mnist import MNISTDataset
from fedstellar.learning.pytorch.syscall.syscall import SYSCALLDataset
from fedstellar.learning.pytorch.cifar10.cifar10 import CIFAR10Dataset
from fedstellar.learning.pytorch.fashionmnist.fashionmnist import FashionMNISTDataset
from fedstellar.config.config import Config
from fedstellar.learning.pytorch.mnist.models.mlp import MNISTModelMLP
from fedstellar.learning.pytorch.mnist.models.cnn import MNISTModelCNN
from fedstellar.learning.pytorch.fashionmnist.models.mlp import FashionMNISTModelMLP
from fedstellar.learning.pytorch.fashionmnist.models.cnn import FashionMNISTModelCNN
from fedstellar.learning.pytorch.sentiment140.models.cnn import SentimentModelCNN
from fedstellar.learning.pytorch.sentiment140.models.rnn import SentimentModelRNN
from fedstellar.learning.pytorch.cifar100.models.cnn import Cifar100ModelCNN
from fedstellar.learning.pytorch.cifar10.models.cnn import Cifar10ModelCNN
from fedstellar.learning.pytorch.syscall.models.mlp import SyscallModelMLP
from fedstellar.learning.pytorch.syscall.models.autoencoder import SyscallModelAutoencoder
from fedstellar.learning.pytorch.cifar10.models.resnet import CIFAR10ModelResNet
from fedstellar.learning.pytorch.cifar10.models.fastermobilenet import FasterMobileNet
from fedstellar.learning.pytorch.cifar10.models.simplemobilenet import SimpleMobileNetV1
from fedstellar.node import Node

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def main():
    example_node_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'config/participant.json.example')
    config_path = "/Users/janosch/Repos/fedstellar-robust/app/single_node/participant_0.json"
    config = Config(entity="participant", participant_config_file=config_path)

    host = "127.0.0.1"
    port = 45600
    idx = 0
    n_nodes = 1
    indices_dir = "/Users/janosch/Repos/fedstellar-robust/app/single_node/"
    is_iid = True
    label_flipping = True
    data_poisoning = False
    model_poisoning = False
    poisoned_percent = 60
    poisoned_ratio = 10
    targeted = True
    target_label = 3
    target_changed_label = 7
    noise_type = "salt"

    dataset = CIFAR10Dataset(iid=is_iid)
    model = Cifar10ModelCNN()

    dataset = DataModule(dataset.trainset, dataset.testset, sub_id=idx, number_sub=n_nodes, indices_dir=indices_dir,
                         label_flipping=label_flipping, data_poisoning=data_poisoning,
                         poisoned_percent=poisoned_percent,
                         poisoned_ratio=poisoned_ratio, targeted=targeted, target_label=target_label,
                         target_changed_label=target_changed_label, noise_type=noise_type)

    node = Node(
        idx=0,
        experiment_name="Single_Node_Test",
        model=model,
        data=dataset,
        hostdemo=True,
        host=host,
        port=port,
        config=config,
        encrypt=False,
        model_poisoning=model_poisoning,
        poisoned_ratio=poisoned_ratio,
        noise_type=noise_type
    )

    node.start()
    time.sleep(5)
    node.set_start_learning(rounds=10, epochs=5)

    while True:
        time.sleep(1)
        finish = True
        for f in [n.round is None for n in [node]]:
            finish = finish and f

        if finish:
            break

    for n in [node]:
        n.stop()


if __name__ == "__main__":
    main()
