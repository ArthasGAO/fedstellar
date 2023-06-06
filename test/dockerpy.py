import docker
client = docker.from_env()

filter = {'label': 'fedstellar'}

containers = client.containers.list(filters=filter)
networks = client.networks.list(filters=filter)

print(containers)
print(networks)

# client.networks.prune()

networks = client.networks.list()

print(containers)
print(networks)
