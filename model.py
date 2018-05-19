
###############################################
#### NOVEL GENERATING MODEL
#### VER 1.0 (2018-5-16)
#### HANYANG UNIV.
#### HYUNG-KWON KO
#### hyungkwonko@gmail.com
###############################################

from __future__ import print_function
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Bidirectional
# from keras.layers import TimeDistributed
# from keras.layers import Dropout
# from keras.models import load_model
# from keras.utils import plot_model # 모델 시각화

import argparse
from zprojectstudy.kt_contest.utils import *

ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='C://users//sunbl//desktop//gaechukja.txt')
ap.add_argument('-batch_size', type=int, default=15) # 배치 사이즈와 seq_length의 차이?
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=15)
ap.add_argument('-hidden_dim', type=int, default=128)
ap.add_argument('-generate_sent_num', type=int, default=15)
ap.add_argument('-sentvec_dim', type=int, default=50)
ap.add_argument('-epoch', type=int, default=200)
# ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
# ap.add_argument('-weights', default='C:\\Users\\sunbl\\PycharmProjects\\gthesis\\zprojectstudy\\kt_contest\\novel_layer2_hidden256_epoch10.h5')

args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
GENERATE_SENT_NUM = args['generate_sent_num']
LAYER_NUM = args['layer_num']
SENTVEC_DIM = args['sentvec_dim']

documents, SENT_SIZE, ix_to_char, char_to_ix = load_data(DATA_DIR)
sentence_vecs, doc2vec_model = sent_vec(documents, SENTVEC_DIM)
X, y = load_xy(SENT_SIZE, SEQ_LENGTH, SENTVEC_DIM, sentence_vecs)

# Load trained model if the weight is specified otherwise we can make new model and go through the training step
if WEIGHTS == '':
    model = generate_model(HIDDEN_DIM, SEQ_LENGTH, SENTVEC_DIM)
    epoch = 0
    while True: # run constantly to improve the performance
        print('\n\nEpoch: {}\n'.format(epoch))
        model.fit(X, y, batch_size=100, verbose=1, epochs=1) # nb_epoch is deprecated, use epochs instead, verbose is an optional function showing the progress
        epoch += 1
        # generate_text(model, GENERATE_SENT_NUM, SENT_SIZE, ix_to_char)
        if epoch % 10 == 0: #  save the weights of  model for every 10 epochs
            model.save_weights('novel_layer2_hidden{}_epoch{}.h5'.format(HIDDEN_DIM, epoch))
            generate_text(model, GENERATE_SENT_NUM, SENT_SIZE, SENTVEC_DIM, ix_to_char, doc2vec_model) # show output
elif WEIGHTS == 'novel_layer2_hidden256_epoch10.h5':
    trained_model = load_model(HIDDEN_DIM, SEQ_LENGTH, SENTVEC_DIM, WEIGHTS)
    generate_text(trained_model, GENERATE_SENT_NUM, SENT_SIZE, SENTVEC_DIM, ix_to_char, doc2vec_model)
else:
    print('\n\nNothing to do!') # pop up error message