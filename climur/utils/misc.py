"""Miscellaneous utility functions."""


import yaml
from typing import Dict


def load_configs(config_file: str) -> Dict:
    """Loads configs from config file.

    Args:
        config_file (str): Path of config (yaml) file.
    
    Returns:
        configs (dict): Dictionary of configs.
    """

    # load yaml file:
    with open(config_file, "r") as yaml_file:
        configs = yaml.safe_load(yaml_file)
    
    return configs

