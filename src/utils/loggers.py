import os
import numpy as np
from typing import Any
from torch import Tensor


class StatisticsLogger:

    def __init__(self, mode: str = ""):
        self.mode = mode
        self.data = dict()

    def _add(self, key: str, value: Any):
        if value is None:
            return
        if isinstance(value, Tensor):
            value = value.item()
        if key not in self.data:
            self.data[key] = list()
        self.data[key].append(value)

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self._add(key, value)

    def to_dict(self):
        log_dict = dict()
        for key, value in self.data.items():
            log_dict.update({
                f"{self.mode}/{key}_mean": np.mean(value)
            })
        return log_dict


class LocalLogger(object):

    def __init__(self, path: str):
        self.path = path
        self.created = set()

    def write(self, log_data: dict, log_type: str):
        log_path = os.path.join(self.path, log_type + '.csv')
        if log_type not in self.created:
            log_file = open(log_path, 'w')
            for col in log_data.keys():
                log_file.write(col + ',')
            log_file.write('\n')
            log_file.close()
            self.created.add(log_type)

        log_file = open(log_path, 'a')
        for col in log_data.values():
            log_file.write(f'{col},')
        log_file.write('\n')
        log_file.close()
