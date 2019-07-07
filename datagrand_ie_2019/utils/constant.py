# -*- coding: UTF-8 -*-
import os
from .experiment_config import ExperimentConfig

__code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
__base_dir = os.path.join(__code_dir, os.pardir)
BASE_DIR = os.path.realpath(__base_dir) + '/'
DATA_DIR = BASE_DIR + 'data/'
MODEL_DIR = DATA_DIR + 'models/'
RAW_DATA_DIR = DATA_DIR + 'datagrand/'
EVALUATION_DIR = DATA_DIR + 'evaluation/'
TRAINING_FILE = DATA_DIR + 'training.json'
VALIDATION_FILE = DATA_DIR + 'validation.json'
TEST_FILE = DATA_DIR + 'test.json'
CONFIG_DIR = BASE_DIR + 'config/'
BRAT_CONFIG_DIR = CONFIG_DIR + 'brat/'
SUBMIT_DIR = DATA_DIR + 'submits/'

# evaluation
RESULT_MISSING = 'missing'
RESULT_ERROR = 'error'
RESULT_MORE = 'more'
DEFAULT_COLOR = 'yellow'

# experiment parameter
EXP_CONFIG = ExperimentConfig()

# type
DEFAULT_TYPE = 'a'
ENTITY_TYPES = ['a', 'b', 'c']

# CoNLL tags
SEQ_BIO = 'BIO'
SEQ_BILOU = 'BILOU'
