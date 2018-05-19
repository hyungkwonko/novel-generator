
###############################################
#### NOVEL GENERATING MODEL
#### VER 1.0 (2018-5-16)
#### HANYANG UNIV.
#### HYUNG-KWON KO
#### hyungkwonko@gmail.com
###############################################

# from collections import Counter
from __future__ import print_function
import numpy as np
import re
from gensim.models import doc2vec
from konlpy.tag import Kkma
from collections import namedtuple
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.optimizers import Adam

# Function to load the data from local repository and make it as a set of sentences comprised of its component words
def load_data(data_dir):
	# data = open("C://users//sunbl//desktop//gaechukja3.txt",'r').read()
	data = open(data_dir, 'r').read() # read out the whole data set
	# elements in the list will be removed using translate built-in function
	havetoerase = ['\n', '?', '!', '-', ':', ',', '(', ')', '<', '>', '金', '明', '天', '掃', '舍', ':'] # this characters will be removed
	data = data.translate({ord(x): '' for x in havetoerase})
	documents = re.split('\.|\"', data)
	while '' in documents: documents.remove('')
	while '  ' in documents: documents.remove('  ')
	documents = [i for i in documents if len(i) > 10] # will only care about the sentence containing more than 10 characters.
	# For example, if the sentence has 8 characters(hi peter.), it will be removed.
	# I guess this would be helpful making logical connection between sentences since they have a possiblity to cause overfitting problem.
	SENT_SIZE = len(documents) # numbef of sentences: 215
	print('Data length: {} characters'.format(len(data))) # number of characters in our data: 7184
	print('Vocabulary size: {} sentences'.format(SENT_SIZE)) # number of sentences in our data: 215
	ix_to_char = {ix:documents for ix, documents in enumerate(documents)} # {number : sentence} dictionary # {0: '로', 1: '마', 2: '인', 3: '이', 4: '야', 5: '기'....}
	char_to_ix = {documents:ix for ix, documents in enumerate(documents)} # the same as above but in reverse order # {'로': 0, '마': 1, '인': 2, '이': 3, '야': 4, '기': 5 ...}
	return documents, SENT_SIZE, ix_to_char, char_to_ix

# function converting the input data into 3-dimensional data structure for keras model
def load_xy(sent_size, seq_length, sentvec_dim, sentence_vecs):
	'''
	:param sent_size: number of sentences(will be 215 in this case)
	:param seq_length: sequences I will take into account for each step(will be 15 in this case)
	:param sentvec_dim: dimension of the sentence vector (will be 300 in this case)
	:param sentence_vecs: list of sentence vectors
	'''
	# make zero-intialized cast like calloc in C
	X = np.zeros((sent_size-seq_length, seq_length, sentvec_dim)) # reason for subtracting seq_length(15)?
	y = np.zeros((sent_size-seq_length, seq_length, sentvec_dim))
	for i in range(sent_size-seq_length): # run 215-15 times. first(0~14), second(1-15), ...last(199~213)
		for j in range(seq_length):
			X[i,j] = sentence_vecs[i+j] # X[0] = 0~14th sentences, X[1] = 1~15th, ... X[199] = 199~213rd
			y[i,j] = sentence_vecs[i+j+1] # y[0] = 1~15th, y[1] = 2~16th, ..., y[199] = 200~214th
	return X, y # return X and y and they will be the input and target value

# function to tokenize the sentence and merge the word with its PoS(part of speech)
def tokenize(doc):
	kkma = Kkma()
	return ["/".join(t) for t in kkma.pos(doc)] # If we put the sentence list, output it with tagged PoS

# function to make sentence vectors using list of sentences
def sent_vec(documents, sentvec_dim):
	train_docs = [tokenize(row) for row in documents]
	docs = []
	analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
	for i, text in enumerate(train_docs):
		tags = [i]
		docs.append(analyzedDocument(text, tags))
	'''
	min_count: If the appearance of the word is less than this number, then we will ignore that word
	window: number of words around the target word we will take into consideration
	worker: 
	'''
	model = doc2vec.Doc2Vec(vector_size=sentvec_dim, window=5, min_count=2, workers=4, alpha=0.025, min_alpha=1e-4, seed=777)
	model.build_vocab(docs)  # Building vocabulary
	model.train(docs, epochs=1000, total_examples=model.corpus_count)
	sentence_vecs = np.zeros((len(train_docs), sentvec_dim))
	for i in range(len(train_docs)):
		sentence_vecs[i] = model.docvecs[i] # move docvecs to the zero-initialized cast
	return sentence_vecs, model # return the set of sentenve vectors

# model generating function
# https://keras.io/layers/recurrent/
def generate_model(hidden_dim, seq_length, sentvec_dim):
	'''
	:param hidden_dim: dimension of each matrix in the cell(have to be power of 2, for example 256, 512. why?)
	:param seq_length: sequences I will take into account for each step(will be 15 in this case)
	:param sentvec_dim: dimension of the sentence vector (will be 300 in this case)
	return_state: whether to return both h_t and c_t. default is false since we generally don't use c_t(LTM, long term memory)
	return _sequence: whether to print in sequence. For example, in case of many to many model, we have to set it TRUE. also for the stacked LSTM layers(obvious)
	stateful: "Stateless" is like resetting LSTM to an "initial state" for every new batch, and 'stateful' means you continue from where you are.
	In both cases(stateless/stateful) LSTM is learning because the transition probabilities are updated.(this explanation bases on the markov chain)
	naturally, we cannot shuffle the training orfer of batch if we set stateful==True(so training should go like batch1 -> batch2 -> ... -> batch last,  we can specify as shuffle=False)
	:return: generated model
	'''
	model = Sequential()
	# model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(seq_length, sentvec_dim), name='BiLSTM_layer')) # be careful what is inside the LSTM bracket and what is not
	model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(None, sentvec_dim), name='input_LSTM_layer'))
	model.add(Dropout(0.5))
	# model.add(LSTM(hidden_dim, return_sequences=True, name='LSTM_layer'))
	model.add(LSTM(hidden_dim, return_sequences=True, name='LSTM_layer'))
	model.add(Dropout(0.5))
	model.add(TimeDistributed(Dense(sentvec_dim), name='Dense_layer'))  # wrapper layer, required to make 3d input to 2d output
	# model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
	optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss="mean_squared_error", optimizer=optimizer)
	print("This is the architecture of our model!\n")
	print(model.summary())
	return model

# if there is trained  model, we can just load it
def load_model(hidden_dim, seq_length, sentvec_dim, weights):
	model = Sequential()
	# model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(seq_length, sentvec_dim), name='BiLSTM_layer'))
	model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(None, sentvec_dim), name='input_LSTM_layer'))
	model.add(Dropout(0.2))
	model.add(LSTM(hidden_dim, return_sequences=True, name='LSTM_layer'))
	model.add(TimeDistributed(Dense(sentvec_dim), name='Dense_layer'))
	model.load_weights(weights, by_name=True)
	return model

# function to pick an index from a probability array
def prob_pick(prob, temperature, n_pick):
	'''
	:param prob: probabilty array
	:param temperature:
	if temperature = 1.0, default value
	if temperature is big (much bigger than 1), More variety of words will be picked-up from the vocabulary, because the probability gap between words decreases.
	if temperature is small (close to 0), the highest one will be picked (work like argmax function)
	:return: the index of the picked-up value (be careful that it is just the index, not the value)
	'''
	prob = np.asarray(prob).astype('float64')
	prob = np.log(prob) / temperature
	exp_prob = np.exp(prob)
	prob = exp_prob / np.sum(exp_prob)
	probability = np.random.multinomial(n_pick, prob, 1)
	return np.argmax(probability)

# function to generate text
def generate_text(model, generate_sent_num, sent_size, sentvec_dim, ix_to_char, doc2vec_model):
	'''
	:param model: LSTM based model we made above
	:param generate_length: how many sentences are we gonna generate
	:param sent_size: number of sentences in the data
	:param sentvec_dim: dimension of the sentence vector
	:param ix_to_char: this convert index to sentence
	:param doc2vec_model: document2vector model we made above
	:return y_final: THE FINAL OUTCOME(novel we have generated)
	'''
	# starting with random character
	ix = np.random.randint(sent_size) # pick one random value smaller than sent_size (random initial value)
	# ix = 2 # we can just set it as the third value(0, 1, 2.. so 2 = third)
	y_final = [ix_to_char[ix]] # return assigned sentence
	X = np.zeros((1, generate_sent_num, sentvec_dim))
	for i in range(generate_sent_num):
		# appending the last predicted character to sequence
		X[0,i,:] = doc2vec_model.docvecs[ix] # new input x
		print(ix_to_char[ix], end='.' + "\n") # end는 캐릭터별로 만들거를 붙여쓰려고 예를 들면 a 엔터 p 엔터 p 엔터 l  엔터 e 이렇게 안뽑고 apple 처럼 concatenated form으로 뽑을라고
		y_vec = model.predict(X[:, :i+1, :])[0,i] # predicted y vector
		y_candidate = doc2vec_model.docvecs.most_similar([y_vec], topn=5)  # 가장 비슷한 거 3개 뽑는다.

		# 샘플 펑션으로 랜덤하게 픽 하는거 구현해야함
		# 예를 들면 topn=5개 중에서 3개 뽑는거 이런거..
		# y_candidate_prob = [y_candidate[0][1], y_candidate[1][1], y_candidate[2][1], y_candidate[4][1], y_candidate[4][1]]
		# y_candidate_index = [y_candidate[0][0], y_candidate[1][0], y_candidate[2][0], y_candidate[3][0], y_candidate[4][0]]
		# prob_pick(y_candidate_prob, temperature=0.8, n_pick=3)
		# # print(y_candidate[0])  # (4, 0.7207441329956055)
		# print(y_candidate[0][0])  # 4
		# print(y_candidate[0][1])  # 0.7207441329956055
		print(y_candidate)
		print("1st sentence: " + ix_to_char[y_candidate[0][0]])
		print("2nd sentence: " + ix_to_char[y_candidate[1][0]])
		print("3rd sentence: " + ix_to_char[y_candidate[2][0]])
		print("4th sentence: " + ix_to_char[y_candidate[3][0]])
		print("5th sentence: " + ix_to_char[y_candidate[4][0]])
		number = input("please choose one sentece: ")
		print("\n")
		ix = y_candidate[int(number)-1][0]
		# ix = int(number)
		y_final.append(ix_to_char[ix]) # append into the list
	print("\n")
	print(y_final)
	return ('. ').join(y_final) # concatenate the output
