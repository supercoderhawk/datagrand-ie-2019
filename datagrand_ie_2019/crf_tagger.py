# -*- coding: UTF-8 -*-
"""
implement CRF training, inference.
"""
import joblib
from sklearn_crfsuite import CRF
from .crf_feature import CRFFeature
from .utils.utils import *
from .data_process.label2entity import *
from .utils.constant import *
from .data_loader import CRFDataLoader


class CRFTagger(object):
    def __init__(self,
                 model_name,
                 model_folder=MODEL_DIR,
                 label_schema=SEQ_BILOU,
                 is_load_model=False,
                 feature_sets=None):
        self.model_path = os.path.join(model_folder, model_name + '.jl')
        self.label_schema = label_schema
        if is_load_model:
            self.__load_model()
        self.feature_sets = feature_sets if feature_sets else []

    def __load_model(self):
        if not os.path.exists(self.model_path):
            raise Exception('model doesn\'t exist!! at {0}'.format(self.model_path))
        self.model = joblib.load(self.model_path)

    def train(self, src_filename, max_iter=200):
        if self.feature_sets:
            if type(self.feature_sets) not in {list, tuple}:
                raise TypeError('feature sets must be list or tuple')
            CRFFeature.feature_sets = self.feature_sets

        data_loader = CRFDataLoader(src_filename)
        sents = data_loader.load_data()
        labels = [sent['labels'] for sent in sents]
        features = []

        for sent in sents:
            sent_tokens = sent['tokens']
            feat = CRFFeature(sent_tokens).sent2feature()
            features.append(feat)
        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=max_iter,
            all_possible_transitions=True,
        )
        crf.fit(features, labels)
        joblib.dump(crf, self.model_path)

    def inference_tokens(self, tokens):
        if not hasattr(self, 'model') or not self.model:
            self.__load_model()
        features = CRFFeature(tokens).sent2feature()
        pred_labels = self.model.predict_single(features)

        return pred_labels, label2entity(tokens, pred_labels, self.label_schema)
