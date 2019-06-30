# -*- coding: utf-8 -*-

import yaml


class ConfigParser:

    def __init__(self, cfg_type, cfg_path):
        self.config_path = cfg_path
        if cfg_type not in ['dirs', 'model', 'thresholds']:
            raise ValueError(
                f"Wrong cfg_type parameter: {cfg_type}."
                f"Should be 'model', 'thresholds' or 'dirs'."
            )
        self.parameters = self._get_parameters()

    def _get_parameters(self):
        """ Function returns directories

        :return:
        """
        with open(self.config_path, 'r') as stream:
            parameters = yaml.load(stream)
        return parameters
