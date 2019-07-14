
'''
@project name: Novel generation by means of Artificial intelligence
@version: Ver. 1.4 (last modification: 2018-7-1)
@written by: Hyung-Kwon Ko @ Hanyang Univ.
@last modified date: 2018-07-01 (Ver. 1.4)
'''

import argparse
from util import *

# Set parameters before running main function
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='banhanga.txt')

ap.add_argument('-batch_size', type=int, default=15) # 배치 사이즈와 seq_length의 차이?
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-timesteps', type=int, default=15) # 여기서 timesteps가 잘못 쓰였음.... 이게 batch size로 쓰였는데 원래는 timesteps = number of iterations 이다
ap.add_argument('-hidden_dim', type=int, default=100)
ap.add_argument('-generate_sent_num', type=int, default=15)
ap.add_argument('-sentvec_dim', type=int, default=15)
ap.add_argument('-epoch', type=int, default=100)
ap.add_argument('-weights', default='')
# ap.add_argument('-weights', default='C:\\Users\\sunbl\\Desktop\\edata\\code\\pycharm\\gthesis\\novel_project\\kt_contest\\m1_layer2_hidden128_epoch50_1.h5')
ap.add_argument('-weights2', default='C:\\Users\\sunbl\\Desktop\\edata\\code\\pycharm\\gthesis\\novel_project\\kt_contest\\m2_layer2_hidden128_epoch50_2.h5')
ap.add_argument('-weights3', default='C:\\Users\\sunbl\\Desktop\\edata\\code\\pycharm\\gthesis\\novel_project\\kt_contest\\m3_layer2_hidden128_epoch50_3.h5')
# ap.add_argument('-d2v_weights', default='')
ap.add_argument('-d2v_weights', default='C:\\Users\\sunbl\\Desktop\\edata\\code\\pycharm\\gthesis\\novel_project\\kt_contest\\doc2vec_m1.h5')
ap.add_argument('-d2v_weights2', default='C:\\Users\\sunbl\\Desktop\\edata\\code\\pycharm\\gthesis\\novel_project\\kt_contest\\doc2vec_m2.h5')
ap.add_argument('-d2v_weights3', default='C:\\Users\\sunbl\\Desktop\\edata\\code\\pycharm\\gthesis\\novel_project\\kt_contest\\doc2vec_m3.h5')
# ap.add_argument('-mode', default='train') # maybe we can have train / run / etc...

args = vars(ap.parse_args())

# set parameters
DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
TIMESTEPS = args['timesteps']
GENERATE_SENT_NUM = args['generate_sent_num']
LAYER_NUM = args['layer_num']
SENTVEC_DIM = args['sentvec_dim']
WEIGHTS = args['weights']
WEIGHTS2 = args['weights2']
WEIGHTS3 = args['weights3']
D2V_WEIGHTS = args['d2v_weights']
D2V_WEIGHTS2 = args['d2v_weights2']
D2V_WEIGHTS3 = args['d2v_weights3']
# N_PICK = args['n_pick']


# main function run
if __name__ == "__main__":
    # Pre-process text data sets
    documents, SENT_SIZE, ix_to_char, char_to_ix = load_data(DATA_DIR, sent_concat_num=1)
    documents2, SENT_SIZE2, ix_to_char2, char_to_ix2 = load_data(DATA_DIR, sent_concat_num=2)
    documents3, SENT_SIZE3, ix_to_char3, char_to_ix3 = load_data(DATA_DIR, sent_concat_num=3)

    # Generate or Load doc2vec model
    if D2V_WEIGHTS == '':
        print("\ndoc2vec model training\n")
        sentence_vecs, doc2vec_model = sent_vec(documents, SENTVEC_DIM)
        sentence_vecs2, doc2vec_model2 = sent_vec(documents2, SENTVEC_DIM)
        sentence_vecs3, doc2vec_model3 = sent_vec(documents3, SENTVEC_DIM)
        doc2vec_model.save('doc2vec_m1.h5')
        doc2vec_model2.save('doc2vec_m2.h5')
        doc2vec_model3.save('doc2vec_m3.h5')
    else:
        print("\ndoc2vec model loaded!\n")
        sentence_vecs, doc2vec_model = sent_vec(documents, SENTVEC_DIM, D2V_WEIGHTS)
        sentence_vecs2, doc2vec_model2 = sent_vec(documents2, SENTVEC_DIM, D2V_WEIGHTS2)
        sentence_vecs3, doc2vec_model3 = sent_vec(documents3, SENTVEC_DIM, D2V_WEIGHTS3)

    # Load sentence vectors (not Korean/English text but numbers)
    X, y = load_xy(SENT_SIZE, TIMESTEPS, SENTVEC_DIM, sentence_vecs)
    X2, y2 = load_xy(SENT_SIZE2, TIMESTEPS, SENTVEC_DIM, sentence_vecs2)
    X3, y3 = load_xy(SENT_SIZE3, TIMESTEPS, SENTVEC_DIM, sentence_vecs3)

    # Load trained model if the weight is specified otherwise we can make new model and go through the training steps
    if WEIGHTS == '':
        model = generate_model(HIDDEN_DIM, SENTVEC_DIM)
        model2 = generate_model(HIDDEN_DIM, SENTVEC_DIM)
        model3 = generate_model(HIDDEN_DIM, SENTVEC_DIM)
        epoch = 0
        print("train model!\n")
        while True: # run constantly to improve the performance
            print('\n\nEpoch: {}'.format(epoch))
            model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1) # verbose is an optional function showing the progress
            model2.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
            model3.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
            epoch += 1
            #  save the weights of  model for every 10 epochs
            if epoch % 500 == 0:
                # model.save_weights('m1_layer2_hidden{}_epoch{}_1.h5'.format(HIDDEN_DIM, epoch))
                # model2.save_weights('m2_layer2_hidden{}_epoch{}_2.h5'.format(HIDDEN_DIM, epoch))
                # model3.save_weights('m3_layer2_hidden{}_epoch{}_3.h5'.format(HIDDEN_DIM, epoch))
                k = 0
                # Generate text !!
                while True:
                    output = generate_text(model, doc2vec_model, model2, doc2vec_model2, model3, doc2vec_model3, GENERATE_SENT_NUM,
                                                                   SENT_SIZE, TIMESTEPS, SENTVEC_DIM, char_to_ix, ix_to_char, ix_to_char2, ix_to_char3) # show output
                    file = open('C://users//sunbl//desktop//five_output_{}.txt'.format(k), 'w')  # file save
                    file.write(output)
                    file.close()
                    k += 1
    else:
        print('model loaded!\n')
        trained_model1 = generate_model(HIDDEN_DIM, SENTVEC_DIM, WEIGHTS)
        trained_model2 = generate_model(HIDDEN_DIM, SENTVEC_DIM, WEIGHTS2)
        trained_model3 = generate_model(HIDDEN_DIM, SENTVEC_DIM, WEIGHTS3)
        while(1):
            generate_text(trained_model1, doc2vec_model, trained_model2, doc2vec_model2, trained_model3, doc2vec_model3, GENERATE_SENT_NUM,
                      SENT_SIZE, TIMESTEPS, SENTVEC_DIM, char_to_ix, ix_to_char, ix_to_char2, ix_to_char3)
    # else:
    # 	print('\n\nNothing to do!') # pop up error message