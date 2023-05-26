#
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

import yaml

from test.utils import set_test_settings

set_test_settings()


def test_check_gpu():
    print("Checking GPU...")


def test_yaml_file(scenario_config="/Users/enrique/Documents/PhD/fedstellar/config/deployment_config.yaml"):
    with open(scenario_config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)
    return config
