############
Installation
############

Fedstellar is a modular, adaptable and extensible framework for creating centralized and decentralized architectures using Federated Learning. Also, the framework enables the creation of a standard approach for developing, deploying, and managing federated learning applications.

The framework enables developers to create distributed applications that use federated learning algorithms to improve user experience, security, and privacy. It provides features for managing data, managing models, and managing federated learning processes. It also provides a comprehensive set of tools to help developers monitor and analyze the performance of their applications.

Prerequisites
=============
* Python 3.8 or higher
* pip3
* Docker

.. _deploy_venv:

Deploy a virtual environment
===================================

`Virtualenv`_ is a tool to build isolated Python environments.

It's a great way to quickly test new libraries without cluttering your
global site-packages or run multiple projects on the same machine which
depend on a particular library but not the same version of the library.

Since Python version 3.3, there is also a module in the standard library
called `venv` with roughly the same functionality.

Create virtual environment
--------------------------
In order to create a virtual environment called e.g. fedstellar using `venv`, run::

  $ python3 -m venv fedstellar-venv

Activate the environment
------------------------
Once the environment is created, you need to activate it. Just change
directory into it and source the script `Scripts/activate` or `bin/activate`.

With bash::

  $ cd fedstellar-venv
  $ . Scripts/activate
  (fedstellar-venv) $

With csh/tcsh::

  $ cd fedstellar-venv
  $ source Scripts/activate
  (fedstellar-venv) $

Notice that the prompt changes once you are activate the environment. To
deactivate it just type deactivate::

  (fedstellar-venv) $ deactivate
  $

After you have created the environment, you can install fedstellar following the instructions below.

Building from source
====================

Obtaining the framework
--------------------

You can obtain the source code from https://github.com/enriquetomasmb/fedstellar

Or, if you happen to have git configured, you can clone the repository::

    git clone https://github.com/enriquetomasmb/fedstellar.git


Now, you can move to the source directory::

        cd fedstellar

Dependencies
------------

Fedstellar requires the additional packages in order to be able to be installed and work properly.

You can install them using pip::

    pip3 install -r requirements.txt



Checking the installation
-------------------------
Once the installation is finished, you can check
by listing the version of the Fedstellar with the following command line::

    python app/main.py --version


Building the fedstellar docker image
====================================
You can build the docker image using the following command line in the root directory::

    docker build -t fedstellar .

You can check the image using::

        docker images

Running Fedstellar
==================
To run Fedstellar, you can use the following command line::

    python app/main.py --webserver [PARAMS]
    
You can show the PARAMS using::

    python app/main.py --help

For a correct execution of the framework, it is necessary to indicate the python path (absolute path)::

    python app/main.py --webserver --python /Users/enrique/fedstellar-venv/bin/python

or::

    python app/main.py --webserver --python C:/Users/enrique/fedstellar-venv/Scripts/python

The webserver will be available at http://127.0.0.1:5000 (by default)

To change the default port, you can use the following command line::

    python app/main.py --webserver --port 8080 --python /Users/enrique/fedstellar-venv/bin/python

Fedstellar Webserver
==================
You can login with the following credentials:

- User: admin
- Password: admin

If not working the default credentials, send an email to enriquetomas@um.es to get the credentials.


Possible issues during the installation or execution
====================================================

If webserver is not working, check the logs in app/logs/server.log

===================================

Network fedstellar_X  Error failed to create network fedstellar_X: Error response from daemon: Pool overlaps with other one on this address space

Solution: Delete the docker network fedstellar_X

    docker network rm fedstellar_X

===================================

Error: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?

Solution: Start the docker daemon

    sudo dockerd

===================================

Error: Cannot connect to the Docker daemon at tcp://X.X.X.X:2375. Is the docker daemon running?

Solution: Start the docker daemon

    sudo dockerd -H tcp://X.X.X.X:2375

===================================

If webserver is not working, kill all process related to the webserver

    ps aux | grep python
    kill -9 PID

===================================

