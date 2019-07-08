import random
from itertools import chain
from pysenal.io import *
from datagrand_ie_2019.utils.constant import *


def split_kfold(dest_dir, k=10):
    total_data = read_json(TRAINING_FILE)
    random.shuffle(total_data)
    random.shuffle(total_data)
    random.shuffle(total_data)
    random.shuffle(total_data)
    random.shuffle(total_data)
    data_len = len(total_data)
    per_count = data_len // k
    remainder = data_len % k
    index = 0
    shards = []
    for i in range(k):
        shard_count = per_count
        if i < remainder:
            shard_count += 1
        shards.append(total_data[index:index + shard_count])
        index += shard_count
    if index != data_len:
        raise ValueError('split error')
    kfold_training_data = []
    kfold_test_data = []
    for i in range(k):
        single_fold_train = shards[:i] + shards[i + 1:]
        t_data = list(chain(*single_fold_train))
        # a = t_data[0]['labels']
        assert isinstance(t_data[0]['labels'], list) is True
        kfold_training_data.append(t_data)
        kfold_test_data.append(shards[i])
    write_json(dest_dir + 'training.json', kfold_training_data)
    write_json(dest_dir + 'test.json', kfold_test_data)
    assert len(kfold_training_data) == k
    assert len(kfold_test_data) == k


if __name__ == '__main__':
    split_kfold(DATA_DIR + 'cv_10/')
