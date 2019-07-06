# -*- coding: UTF-8 -*-
"""data process script includes some data process logic"""
from datagrand_ie_2019.api_model import Model
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.data_process.generate_experiment_data import *


def run_batch():
    model = Model()
    model.run({'text': 'Your Content'})


if __name__ == '__main__':
    ExperimentDataGenerator().generate()
