# -*- coding: UTF-8 -*-f
from pysenal.io import *
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.pipeline import CrfNerPipeline
from datagrand_ie_2019.experiment import NeuralExperiment


class ResultRunner(object):

    def __init__(self):
        self.crf_pipeline = CrfNerPipeline('crf_more_feature')

    def runner(self):
        write_file(SUBMIT_DIR + 'nn_baseline_v2.txt', self.nn_runner())
        # write_file(SUBMIT_DIR + 'crf_more_feature.txt', self.crf_runner())

    def crf_runner(self):
        test_data = read_json(TEST_FILE)
        for sent in test_data:
            entities = self.crf_pipeline.run(sent['tokens'])
            sent['entities'] = entities
        return self.convert_result(test_data)

    def nn_runner(self):
        dest_filename = EVALUATION_DIR + 'nn_baseline_new_train_submit.json'
        NeuralExperiment().inference(MODEL_DIR + 'nn/nn_baseline_new_train', TEST_FILE, dest_filename)

        return self.convert_result(read_json(dest_filename))

    @classmethod
    def convert_result(cls, data):
        ret_sents = []
        for sent in data:
            ret_sent = cls.convert_sent(sent)
            ret_sents.append(ret_sent)
        return '\n'.join(ret_sents) + '\n'

    @classmethod
    def convert_sent(cls, sent):
        tokens = sent['tokens']
        token_texts = [t['text'] if isinstance(t, dict) else t for t in tokens]
        entities = sorted(sent['entities'], key=lambda e: e['start'])

        segments = []
        last_end = 0
        for entity in entities:
            start = entity['start']
            end = entity['end']
            if last_end < start:
                last_segment = '_'.join(token_texts[last_end:start]) + '/o'
                segments.append(last_segment)
            segment = '_'.join(token_texts[start:end]) + '/' + entity['type']
            segments.append(segment)
        if last_end < len(tokens):
            segment = '_'.join(token_texts[last_end:]) + '/o'
            segments.append(segment)
        return '  '.join(segments)


if __name__ == '__main__':
    ResultRunner().runner()
