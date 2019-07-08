# -*- coding: UTF-8 -*-
"""
experiment runner functions and settings
"""
from pysenal.io import *
from datagrand_ie_2019.crf_tagger import CRFTagger
from datagrand_ie_2019.nn_crf import NeuralNetworkCRF
from datagrand_ie_2019.nn_crf_config import NeuralNetworkCRFConfig
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.utils.evaluation import EntityEvaluator, KFoldEntityEvaluator
from datagrand_ie_2019.data_process.label2entity import label2entity
from datagrand_ie_2019.postprocess import Postprocessor


class CRFExperiment(object):
    def __init__(self, model_name):
        self.basename = model_name
        self.tagger = CRFTagger(self.basename)
        self.test_evaluator = EntityEvaluator(DATA_DIR + 'pre_data/test.json')

    def train(self, filename=TRAINING_FILE):
        self.tagger.train(filename)
        return self

    def cross_validation(self, cv_basedir):
        test_dest_dir = cv_basedir + 'evaluation/'
        training_data = read_json(cv_basedir + 'training.json')
        test_data = read_json(cv_basedir + 'test.json')
        test_eval_data = []
        pred_filename = test_dest_dir + self.basename + '.json'
        for idx, (single_fold_train, single_fold_test) in enumerate(zip(training_data, test_data)):
            tagger = CRFTagger(self.basename + '_{}'.format(idx), model_folder=cv_basedir + 'models/')
            tagger.train(single_fold_train)

            for test_sent in single_fold_test:
                labels, entities = tagger.inference_tokens(test_sent['tokens'])
                test_sent['entities'] = entities
                test_sent['labels'] = labels
            test_eval_data.append(single_fold_test)
        write_json(pred_filename, test_eval_data)
        evaluator = KFoldEntityEvaluator(10, cv_basedir + 'test.json')
        evaluator.evaluate(pred_filename)

    def inference_json(self, data):
        for paragraph in data:
            tokens = paragraph['tokens']
            labels, entities = self.tagger.inference_tokens(tokens)
            paragraph['entities'] = entities
            paragraph['labels'] = labels
        return data

    def inference_file(self, src_filename, dest_filename):
        result = self.inference_json(read_json(src_filename))
        write_json(dest_filename, result)

    def evaluation(self):
        test_pred_filename = EVALUATION_DIR + self.basename + '_test.json'
        test_true_filename = DATA_DIR + 'pre_data/test.json'
        self.inference_file(test_true_filename, test_pred_filename)

        print('=============')
        print('test result:')
        test_counter = self.test_evaluator.evaluate(test_pred_filename)
        print(test_counter)


class NeuralExperiment(object):
    def __init__(self):
        pass

    def train(self, config, dest_dir):
        model = NeuralNetworkCRF(mode=FIT, dest_dir=dest_dir, config=config, label2result=label2entity)
        model.fit()

    def inference(self, model_basename, test_filename, dest_filename):
        config = NeuralNetworkCRFConfig.from_json(model_basename + '.json')

        model = NeuralNetworkCRF(mode=INFERENCE, dest_dir=None, config=config, label2result=label2entity)
        sents = read_json(test_filename)
        ret = model.inference(sents)
        write_json(dest_filename, ret)

    def evaluation(self, model_name):
        true_filename = DATA_DIR + 'pre_data/test.json'
        dest_filename = EVALUATION_DIR + model_name + '.json'
        self.inference(MODEL_DIR + 'nn/' + model_name,
                       true_filename,
                       dest_filename)
        evaluator = EntityEvaluator(true_filename)
        ret = evaluator.evaluate(dest_filename)
        print(ret)
