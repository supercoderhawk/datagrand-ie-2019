# -*- coding: UTF-8 -*-
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.experiment import *
from datagrand_ie_2019.nn_crf import *

# CRFExperiment('baseline').train().evaluation()
# CRFExperiment('crf_context_train').train(DATA_DIR + 'pre_data/training.json').evaluation()
# CRFExperiment('crf_context').train().evaluation()
# CRFExperiment('baseline_train').train(DATA_DIR + 'pre_data/training.json').evaluation()

# config = NeuralNetworkCRFConfig(training_filename=TRAINING_FILE,
#                                 dict_path=DATA_DIR + 'neural_vocab.txt',
#                                 model_name='test_final',
#                                 loss_function_name=LOSS_LOG_LIKELIHOOD,
#                                 learning_rate=0.001,
#                                 dropout_rate=0.5,
#                                 regularization_rate=5e-4,
#                                 skip_left=2,
#                                 skip_right=1,
#                                 word_embed_size=100,
#                                 hidden_layers=({'type': 'bidirectional_lstm', 'units': 50},),
#                                 dropout_position=NNCRF_DROPOUT_EMBEDDING,
#                                 batch_length=600,
#                                 batch_size=20,
#                                 hinge_rate=0.2,
#                                 label_schema=SEQ_BILOU)
# model = NeuralNetworkCRF(mode=FIT, dest_dir=MODEL_DIR + 'nn/', config=config, label2result=None)
# model.fit()
# NeuralExperiment().evaluation('test_train_short')

# CRFExperiment('crf_context_cv_10').cross_validation(DATA_DIR + 'cv_10/')