<!-- PROJECT LOGO -->
<br>
<p align="center">
  <a href="https://github.com/enriquetomasmb/fedstellar">
    <img src="docs/_static/fedstellar-logo.jpg" alt="fedstellar">
  </a>
  <h3 align="center">Fedstellar (Synchronous)</h3>

  <p align="center">
    Framework for Decentralized Federated Learning
    <br>
    Adapted for Sychronous Rounds
    <br>
  </p>
</p>

## About the project

Fedstellar is a modular, adaptable and extensible framework for creating centralized and decentralized architectures using Federated Learning. Also, the framework enables the creation of a standard approach for developing, deploying, and managing federated learning applications.
<br><br>
The framework enables developers to create distributed applications that use federated learning algorithms to improve user experience, security, and privacy. It provides features for managing data, managing models, and managing federated learning processes. It also provides a comprehensive set of tools to help developers monitor and analyze the performance of their applications.
<br>
<br>
The framework is developed by Enrique Tomás Martínez Beltrán in collaboration with the University of Murcia and Armasuisse.

<a href="https://um.es">
  <img src="docs/_static/umu.jpg" alt="fedstellar" width="200" height="60">
</a>
<a href="#">
  <img src="docs/_static/armasuisse.jpg" alt="fedstellar" width="200" height="60">
</a>
<br><br>
For any questions, please contact Enrique Tomás Martínez Beltrán <a href="mailto:enriquetomas@um.es">enriquetomas@um.es</a>.


## Adaptions
The initial framework has been adapted to a synchronous version for more comparable experiments. 
Therefore, at each round the nodes only share their own trained model to their neighbours. The aggregation is 
only performed when completed, i.e., when all model from all neighbours are received. The gossiping algorithm
has been removed. The find the necessary changes in the code base, search for `TODO Sync`.

For the synchronized version of Fedstellar, a number of adaptions have been implemented. 
First, before any node starts its initial round, the connection to all neighbors is ensured. 
This change is implemented in `node_start.py`. 
Additionally, the node connection timeout in `node_connection.py`
has been removed. Secondly, once the local training is finished, only the local model is broadcasted to
the corresponding neighbors. The partial aggregation has been removed. 
Instead of an aggregation timeout, the aggregator of each node waits until it has received all models. 
Consequently, this adjustment ensures that all nodes complete their local training round, followed by a complete aggregation. 
The technical specifications are available in the abstract aggregator class in `aggregator.py`.
As a final adaption, a synchronization barrier ensures that all nodes complete 
the current round before proceeding to the next. Details can be retrieved from the code in `node.py`, `__on_round_finished} 
With these changes in place, every node is ensured to aggregate each neighbor model and synchronize the round progression. 
Correspondingly, if a one or more nodes fail, the FL scenario is terminated holistically.

## Getting Started

The overview of Fedstellar is available on the [website](https://federatedlearning.inf.um.es/). The corresponding
documentation can be found [here](https://federatedlearning.inf.um.es/docs/).
To run the framework without using the frontend, see `test_bed.py`.

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Author

* **Enrique Tomás Martínez Beltrán** - [Website](https://enriquetomasmb.com) - [Email](mailto:enriquetomas@um.es)
* **Janosch Baltensperger** - [Email](mailto:janosch.baltensperger@uzh.ch)
