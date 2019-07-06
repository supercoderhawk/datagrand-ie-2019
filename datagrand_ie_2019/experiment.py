# -*- coding: UTF-8 -*-
"""
experiment runner functions and settings
"""
from pysenal.io import *
from datagrand_ie_2019.crf_tagger import CRFTagger
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.utils.evaluation import EntityEvaluator
from datagrand_ie_2019.postprocess import Postprocessor


class CRFExperiment(object):
    def __init__(self, model_name, remark=''):
        if remark and not remark.startswith(('-', '_')):
            remark = '_' + remark
        self.remark = remark
        self.basename = model_name + remark
        self.tagger = CRFTagger(self.basename)
        self.test_evaluator = EntityEvaluator(DATA_DIR + 'pre_data/test.json')

    def train(self, filename=TRAINING_FILE):
        self.tagger.train(self.get_name(filename, self.remark))
        return self

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

    @staticmethod
    def get_name(filename, suffix):
        if suffix:
            return filename[:filename.rindex('.')] + suffix + filename[filename.rindex('.'):]
        else:
            return filename


if __name__ == '__main__':
    CRFExperiment('baseline').train().evaluation()
