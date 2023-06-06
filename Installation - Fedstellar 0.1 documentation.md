Toggle table of contents sidebar

Fedstellar is a modular, adaptable and extensible framework for creating centralized and decentralized architectures using Federated Learning. Also, the framework enables the creation of a standard approach for developing, deploying, and managing federated learning applications.

The framework enables developers to create distributed applications that use federated learning algorithms to improve user experience, security, and privacy. It provides features for managing data, managing models, and managing federated learning processes. It also provides a comprehensive set of tools to help developers monitor and analyze the performance of their applications.

## Prerequisites[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#prerequisites "Permalink to this heading")

-   Python 3.8 or higher
    
-   pip3
    
-   Docker
    

## Deploy a virtual environment[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#deploy-a-virtual-environment "Permalink to this heading")

[\`Virtualenv\`\_](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#id1) is a tool to build isolated Python environments.

It’s a great way to quickly test new libraries without cluttering your global site-packages or run multiple projects on the same machine which depend on a particular library but not the same version of the library.

Since Python version 3.3, there is also a module in the standard library called venv with roughly the same functionality.

### Create virtual environment[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#create-virtual-environment "Permalink to this heading")

In order to create a virtual environment called e.g. fedstellar using venv, run:

```
$ python3 -m venv fedstellar-venv

```

### Activate the environment[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#activate-the-environment "Permalink to this heading")

Once the environment is created, you need to activate it. Just change directory into it and source the script Scripts/activate or bin/activate.

With bash:

```
$ cd fedstellar-venv
$ . Scripts/activate
(fedstellar-venv) $

```

With csh/tcsh:

```
$ cd fedstellar-venv
$ source Scripts/activate
(fedstellar-venv) $

```

Notice that the prompt changes once you are activate the environment. To deactivate it just type deactivate:

```
(fedstellar-venv) $ deactivate
$

```

After you have created the environment, you can install fedstellar following the instructions below.

## Building from source[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#building-from-source "Permalink to this heading")

### Obtaining the framework[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#obtaining-the-framework "Permalink to this heading")

You can obtain the source code from [https://github.com/enriquetomasmb/fedstellar](https://github.com/enriquetomasmb/fedstellar)

Or, if you happen to have git configured, you can clone the repository:

```
git clone https://github.com/enriquetomasmb/fedstellar.git

```

Now, you can move to the source directory:

### Dependencies[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#dependencies "Permalink to this heading")

Fedstellar requires the additional packages in order to be able to be installed and work properly.

You can install them using pip:

```
pip3 install -r requirements.txt

```

### Checking the installation[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#checking-the-installation "Permalink to this heading")

Once the installation is finished, you can check by listing the version of the Fedstellar with the following command line:

```
python app/main.py --version

```

## Building the fedstellar docker image (CPU version)[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#building-the-fedstellar-docker-image-cpu-version "Permalink to this heading")

You can build the docker image using the following command line in the root directory:

```
docker build -t fedstellar .

```

## Building the fedstellar docker image (GPU version)[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#building-the-fedstellar-docker-image-gpu-version "Permalink to this heading")

You can build the docker image using the following command line in the root directory:

```
docker build -t fedstellar-gpu -f Dockerfile-gpu .

```

Also, you have to follow the instructions in the following link to install nvidia-container-toolkit:

[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Checking the docker images[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#checking-the-docker-images "Permalink to this heading")

You can check the docker images using the following command line:

## Running Fedstellar[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#running-fedstellar "Permalink to this heading")

To run Fedstellar, you can use the following command line:

```
python app/main.py --webserver [PARAMS]

```

You can show the PARAMS using:

```
python app/main.py --help

```

For a correct execution of the framework, it is necessary to indicate the python path (absolute path):

```
python app/main.py --webserver --python /Users/enrique/fedstellar-venv/bin/python

```

or:

```
python app/main.py --webserver --python C:/Users/enrique/fedstellar-venv/Scripts/python

```

The webserver will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000/) (by default)

To change the default port, you can use the following command line:

```
python app/main.py --webserver --port 8080 --python /Users/enrique/fedstellar-venv/bin/python

```

## Fedstellar Webserver[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#fedstellar-webserver "Permalink to this heading")

You can login with the following credentials:

-   User: admin
    
-   Password: admin
    

If not working the default credentials, send an email to [enriquetomas@um.es](mailto:enriquetomas%40um.es) to get the credentials.

## Possible issues during the installation or execution[#](https://fedstellar.enriquetomasmb.com/7ecafa3f19d05b4e9da98f734fd69aca7baa4965/installation.html#possible-issues-during-the-installation-or-execution "Permalink to this heading")

If webserver is not working, check the logs in app/logs/server.log

___

Network fedstellar\_X Error failed to create network fedstellar\_X: Error response from daemon: Pool overlaps with other one on this address space

Solution: Delete the docker network fedstellar\_X

> docker network rm fedstellar\_X

___

Error: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?

Solution: Start the docker daemon

> sudo dockerd

___

Error: Cannot connect to the Docker daemon at [tcp://X.X.X.X:2375](tcp://X.X.X.X:2375). Is the docker daemon running?

Solution: Start the docker daemon

___

If webserver is not working, kill all process related to the webserver

> ps aux | grep python kill -9 PID

___