# -*- coding: UTF-8 -*-
from pysenal.io import read_json, write_json
from datagrand_ie_2019.utils.constant import *


def labels_processor(filename):
    sents = []
    for sent in read_json(filename):
        labels = []
        if 'labels' in sent:
            for label in sent['labels']:
                if not label.startswith('O'):
                    label = label + '-iupac'
                labels.append(label)
            sent['labels'] = labels
        sents.append(sent)
    write_json(filename, sents)


def token_processor(filename,prefix):
    dirname, basename = os.path.split(filename)
    dest_filename = EVALUATION_DIR+prefix+'_'+basename
    for true_sent, pred_sent in zip(read_json(filename),read_json(dest_filename)):
        pred_sent['']

if __name__ == '__main__':
    labels_processor(TRAINING_FILE)
    labels_processor(VALIDATION_FILE)
    labels_processor(VALIDATION_OOV_FILE)
    labels_processor(TEST_FILE)
    labels_processor(TEST_OOV_FILE)
