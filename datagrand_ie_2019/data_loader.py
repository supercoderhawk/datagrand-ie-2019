# -*- coding: UTF-8 -*-
"""
define training data loader class
"""
from pysenal.io import read_json, read_jsonline
from pysenal.utils.logger import get_logger


class CRFDataLoader(object):
    def __init__(self, filename):
        self.filename = filename
        self.logger = get_logger('[CRFDataLoader]')

    def load_data(self):
        if self.filename.endswith('.json'):
            data = read_json(self.filename)
        elif self.filename.endswith('.jsonl'):
            data = read_jsonline(self.filename)
        else:
            data = read_json(self.filename)
            self.logger.warning('Your file suffix is not json or jsonl')
        return data
