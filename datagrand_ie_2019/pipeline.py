# -*- coding: UTF-8 -*-
"""
encapsulate general NER interface without modelhub dependency
"""
from datagrand_ie_2019.utils.constant import MODEL_DIR
from datagrand_ie_2019.postprocess import Postprocessor
from datagrand_ie_2019.crf_tagger import CRFTagger


class NerPipeline(object):
    def __init__(self, model_name, model_dir):
        self.mode_name = model_name
        self.mode_dir = model_dir

    def run(self, text):
        raise NotImplementedError('not implement run method')


class CrfNerPipeline(NerPipeline):
    def __init__(self, model_name, model_dir=MODEL_DIR):
        super().__init__(model_name, model_dir)
        self.tagger = CRFTagger(model_name, model_dir)

    def run(self, tokens):
        labels, entities = self.tagger.inference_tokens(tokens)
        final_entities = Postprocessor(tokens).post_process(entities)
        return final_entities


__all__ = ['CrfNerPipeline']

if __name__ == '__main__':
    text = 'Steve Job founded Apple.Inc.'
    CrfNerPipeline('unigram').run(text)
