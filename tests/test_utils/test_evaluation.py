# -*- coding: UTF-8 -*-
from datagrand_ie_2019.utils.constant import (VALIDATION_FILE,
                                                                          VALIDATION_OOV_FILE, EVALUATION_DIR)
from datagrand_ie_2019.utils.evaluation import *


def test_entity_evaluator():
    evaluator = EntityEvaluator(VALIDATION_FILE, VALIDATION_OOV_FILE, entity_types=['iupac'])
    ret = evaluator.evaluate(EVALUATION_DIR + 'unigram_validation.json', is_percentage=True)
    assert ret['precision']['iupac'] == '94.95%'
    assert ret['recall']['iupac'] == '63.46%'
    assert ret['f1']['iupac'] == '76.07%'
    assert ret['oov_rate']['iupac'] == '63.46%'


def test_conll_evaluator():
    evaluator = LabelEvaluator(VALIDATION_FILE)
