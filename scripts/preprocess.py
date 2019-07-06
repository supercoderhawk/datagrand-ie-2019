# -*- coding: UTF-8 -*-
from pysenal.io import *
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.data_process.entity2label import Entity2Label


def process_training_data(src_filename, dest_filename):
    e2l = Entity2Label(resolve_conflict=False)
    data = []
    for idx, line in enumerate(read_lines_lazy(src_filename)):
        tokens = []
        entities = []
        labels = []
        index = 0
        for segment in line.split('  '):
            token_seq_str, tag = segment.rsplit('/', 1)
            token_seq = token_seq_str.split('_')
            seg_token_len = len(token_seq)
            seg_tokens = []
            for t_idx, token_str in enumerate(token_seq):
                start = index + t_idx
                token = {'text': token_str, 'start': start, 'end': start + 1}
                seg_tokens.append(token)
            if tag != 'o':
                entity = {'start': index, 'end': index + len(token_seq), 'type': tag}
                entities.append(entity)
                seg_labels = e2l.single({'type': tag}, index, index + len(token_seq))
            else:
                seg_labels = ['O'] * seg_token_len
            tokens.extend(seg_tokens)
            labels.extend(seg_labels)
            index += len(token_seq)

        item = {'tokens': tokens, 'entities': entities, 'labels': labels, 'index': idx}
        data.append(item)
    write_json(dest_filename, data)


def process_test_data():
    data = []
    for idx, line in enumerate(read_lines_lazy(RAW_DATA_DIR + 'test.txt')):
        token_texts = line.split('_')
        tokens = []
        for t_idx, token in enumerate(token_texts):
            tokens.append({'text': token, 'start': t_idx, 'end': t_idx + 1})
        data.append({'tokens': tokens})
    write_json(TEST_FILE, data)


def split_data():
    data = read_json(TRAINING_FILE)
    count = len(data)
    training_count = int(count * 0.9)
    write_json(DATA_DIR + 'pre_data/training.json', data[:training_count])
    write_json(DATA_DIR + 'pre_data/test.json', data[training_count:])


if __name__ == '__main__':
    # process_training_data(RAW_DATA_DIR + 'train.txt', DATA_DIR + 'training.json')
    # split_data()
    process_test_data()
