
###############################################
#### NOVEL GENERATING MODEL
#### VER 1.2 (2018-5-21)
#### HANYANG UNIV.
#### HYUNG-KWON KO
#### hyungkwonko@gmail.com
###############################################

from __future__ import print_function
# from collections import Counter
import numpy as np
import re
from gensim.models import doc2vec
from konlpy.tag import Kkma
from collections import namedtuple
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.optimizers import Adam

# Function to load the data from local repository and make it as a set of sentences comprised of its component words
def load_data(data_dir, sent_concat_num):
	# data = open("C://users//sunbl//desktop//gaechukja.txt",'r').read()
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
	if(sent_concat_num==1):
		print("Model{}: No sentences concatenated".format(sent_concat_num))
	elif(sent_concat_num==2):
		documents2 = []
		for i in range(len(documents) - 1):
			documents2.append(documents[i] + ". " + documents[i + 1]) # concatenate 2 sentences
		documents = documents2
		print("Model{}: {} sentences concatenated.".format(sent_concat_num, sent_concat_num))
	elif(sent_concat_num==3):
		documents3 = []
		for i in range(len(documents) - 2):
			documents3.append(documents[i] + ". " + documents[i + 1] + ". " + documents[i + 2])
		documents = documents3
		print("Model{}: {} sentences concatenated.".format(sent_concat_num, sent_concat_num))
	SENT_SIZE = len(documents) # numbef of sentences: 211
	print('The model{} contains: {} characters'.format(sent_concat_num, len(data))) # number of characters in our data: 7184
	print('Vocabulary size: {} sentences'.format(SENT_SIZE)) # number of sentences in our data: 211
	ix_to_char = {ix:documents for ix, documents in enumerate(documents)} # {number : sentence} dictionary # {0: '로', 1: '마', 2: '인', 3: '이', 4: '야', 5: '기'....}
	char_to_ix = {documents:ix for ix, documents in enumerate(documents)} # the same as above but in reverse order # {'로': 0, '마': 1, '인': 2, '이': 3, '야': 4, '기': 5 ...}
	return documents, SENT_SIZE, ix_to_char, char_to_ix

# function converting the input data into 3-dimensional data structure for keras model
# only able to work with (sent_size//timesteps)*timesteps amount of sentences, if inappropriate input size is assigned, will not work
def load_xy(sent_size, timesteps, sentvec_dim, sentence_vecs):
	'''
	:param sent_size: number of sentences(will be 211 in this case)
	:param timesteps: How many steps I will go through(will be 15 in this case)
	:param sentvec_dim: dimension of the sentence vector (will be 50 in this case)
	:param sentence_vecs: list of sentence vectors
	'''
	# make zero-intialized cast like calloc in C
	X = np.zeros((sent_size//timesteps, timesteps, sentvec_dim)) # sent_size//timesteps = sequence length(will be 14 in this case) that  we will put in one line
	y = np.zeros((sent_size//timesteps, timesteps, sentvec_dim))
	for i in range(sent_size//timesteps): # run 211//15 times. = 14
		for j in range(timesteps):
			X[i,j] = sentence_vecs[timesteps*i + j] # X[0] = 0~14th sentences, X[1] = 15~29th, ... X[13] = 196~210rd
			y[i,j] = sentence_vecs[timesteps*i + j + 1] # y[0] = 1~15th, y[1] = 16~30th, ..., y[13] = 197~211h
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
	model = doc2vec.Doc2Vec(vector_size=sentvec_dim, window=5, min_count=2, workers=4, alpha=0.025, min_alpha=1e-4, seed=777)
	model.build_vocab(docs)  # Building vocabulary
	model.train(docs, epochs=1000, total_examples=model.corpus_count)
	sentence_vecs = np.zeros((len(train_docs), sentvec_dim))
	for i in range(len(train_docs)):
		sentence_vecs[i] = model.docvecs[i] # move docvecs to the zero-initialized cast
	return sentence_vecs, model # return the set of sentenve vectors

# model generating function
# https://keras.io/layers/recurrent/
def generate_model(hidden_dim, sentvec_dim):
	'''
	:param hidden_dim: dimension of each matrix in the cell(have to be power of 2, for example 256, 512. why?)
	:param sentvec_dim: dimension of the sentence vector (will be 300 in this case)
	return_state: whether to return both h_t and c_t. default is false since we generally don't use c_t(LTM, long term memory)
	return _sequence: whether to print in sequence. For example, in case of many to many model, we have to set it TRUE. also for the stacked LSTM layers(obvious)
	stateful: "Stateless" is like resetting LSTM to an "initial state" for every new batch, and 'stateful' means you continue from where you are.
	In both cases(stateless/stateful) LSTM is learning because the transition probabilities are updated.(this explanation bases on the markov chain)
	naturally, we cannot shuffle the training orfer of batch if we set stateful==True(so training should go like batch1 -> batch2 -> ... -> batch last,  we can specify as shuffle=False)
	:return: generated model
	'''
	model = Sequential()
	model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True, stateful=False),
							input_shape=(None, sentvec_dim), name='BiLSTM_input_layer')) # be careful what is inside the LSTM bracket and what is not
	# model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(None, sentvec_dim), name='input_LSTM_layer')) # we can use also LSTM
	model.add(Dropout(0.3))
	model.add(LSTM(hidden_dim, return_sequences=True, stateful=False, name='LSTM_layer'))
	model.add(Dropout(0.3))
	model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True, stateful=False, name='BiLSTM_output_layer')))
	model.add(TimeDistributed(Dense(sentvec_dim), name='Dense_layer'))  # wrapper layer, required to make 3d input to 2d output
	# model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
	optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss="mean_squared_error", optimizer=optimizer)
	print("\n\n\n")
	print("This is the architecture of our model.")
	print(model.summary())
	return model

# if there is trained  model, we can just load it
def load_model(hidden_dim, sentvec_dim, weights):
	model = Sequential()
	model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True, stateful=False), input_shape=(None, sentvec_dim), name='BiLSTM_input_layer'))
	model.add(Dropout(0.3))
	model.add(LSTM(hidden_dim, return_sequences=True, stateful=False, name='LSTM_layer'))
	model.add(Dropout(0.3))
	model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True, stateful=False, name='BiLSTM_output_layer')))
	model.add(TimeDistributed(Dense(sentvec_dim), name='Dense_layer'))  # wrapper layer, required to make 3d input to 2d output
	model.load_weights(weights, by_name=True)
	return model

# function to pick an index from a probability array
def pick_candidate(prob, temperature, n_pick):
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
	idx = np.array([[], []])  # initializatio
	while(len(idx[1])<n_pick): # run while it has full count
		probability = np.random.multinomial(n_pick, prob, 1)
		idx = np.where(probability==1)
	return idx[1] # array containing picked indexes

# function to generate text using five different models
def generate_text(model, doc2vec_model, model2, doc2vec_model2, model3, doc2vec_model3, generate_sent_num, sent_size,
				  timesteps, sentvec_dim, char_to_ix, ix_to_char, ix_to_char2, ix_to_char3):
	ix = np.random.randint((sent_size//timesteps) * timesteps) # pick one random value smaller than sent_size (random initial value)
	ix_list = []
	ix_list.append(ix)
	y_final = [ix_to_char[ix]] # return assigned sentence
	X = np.zeros((1, generate_sent_num, sentvec_dim))
	for i in range(generate_sent_num): # appending the last predicted sentence to sequence
		print(ix_to_char[ix], end='.' + "\n") # print out the current sentence
		X0, ix0 = generate_text_all_previous_step(model, doc2vec_model, X, ix, i)
		X1, ix1 = generate_text_one_previous_step(model, doc2vec_model, X, ix, i)
		X2, ix2 = generate_text_two_previous_step(model, doc2vec_model, X, ix, i)
		X3, ix3 = generate_text_two_to_two(model, doc2vec_model, ix_to_char, char_to_ix, model2, doc2vec_model2, ix_to_char2, X, ix_list, i)
		X4, ix4 = generate_text_three_to_three(model, doc2vec_model, ix_to_char, char_to_ix, model3, doc2vec_model3, ix_to_char3, X, ix_list, i)
		ix_set = [ix0, ix1, ix2, ix3, ix4]
		X_set = [X0, X1, X2, X3, X4]
		ix, X, ix_list = pick_ix_X(ix_set, X_set, ix_to_char, ix_list) # show and pick one
		y_final.append(ix_to_char[ix]) # append into the output list
	print("\n")
	print("GENERATED PARAGRAPH: " + ('. ').join(y_final)) # concatenated output
	return ('. ').join(y_final)

# model 1 (generate from all previous steps in a cumulative manner)
def generate_text_all_previous_step(model, doc2vec_model, X_old, ix, i):
	X_old[0,i,:] = doc2vec_model.docvecs[ix] # new input x
	y_vec = model.predict(X_old[:, :i+1, :])[0,i] # predicted y vector
	y_candidate = doc2vec_model.docvecs.most_similar([y_vec], topn=7)  # pick 7 candidates first and then we will finally choose 3 among them probabilistically
	y_candidate_prob = []
	for i in range(len(y_candidate)):
		y_candidate_prob.append(y_candidate[i][1])
	y_candidate_index = []
	for i in range(len(y_candidate)):
		y_candidate_index.append(y_candidate[i][0])
	# process of choosing 'n_pick' number of candidates among 7
	ix_candidate = pick_candidate(y_candidate_prob, temperature=1, n_pick=1)
	ix_new = y_candidate[ix_candidate[0]][0]
	X_new = X_old
	return X_new, ix_new # return the index 'ix'
# model 2 (generate from one previous step, uses the same structure and doc2vec model as above)
def generate_text_one_previous_step(model, doc2vec_model, X_old, ix, i):
	X_old[0,i,:] = doc2vec_model.docvecs[ix] # new input x
	X_temp = X_old[0,i,:].reshape(1,1,len(X_old[0,i,:])) # reshape (50,) -> (1,1,50)
	y_vec = model.predict(X_temp)[0,0] # predicted y vector
	y_candidate = doc2vec_model.docvecs.most_similar([y_vec], topn=7)  # pick 7 candidates first and then we will finally choose 3 among them probabilistically
	y_candidate_prob = []
	for i in range(len(y_candidate)):
		y_candidate_prob.append(y_candidate[i][1])
	y_candidate_index = []
	for i in range(len(y_candidate)):
		y_candidate_index.append(y_candidate[i][0])
	# process of choosing 'n_pick' number of candidates among 7
	ix_candidate = pick_candidate(y_candidate_prob, temperature=1, n_pick=1)
	ix_new = y_candidate[ix_candidate[0]][0]
	X_new = X_old
	return X_new, ix_new # return the index 'ix'
# model 3 (generate from two previous steps, uses the same structure and doc2vec model as above)
def generate_text_two_previous_step(model, doc2vec_model, X_old, ix, i):
	if(i<1): #  since we cannot put two inputs at first(X_-1, X_0), we need to use previous model to generate the first input X_1. after this step, we can use two X_i's
		X_new, ix_new = generate_text_all_previous_step(model, doc2vec_model, X_old, ix, i)
	else:
		X_old[0,i,:] = doc2vec_model.docvecs[ix] # new input x
		y_vec = model.predict(X_old[:, i-1:i+1, :])[0,1] # uses input as X_old[:, i-1:i+1, :] which means we will use two steps
		y_candidate = doc2vec_model.docvecs.most_similar([y_vec], topn=7)  # pick 7 candidates first and then we will finally choose 3 among them probabilistically
		y_candidate_prob = []
		for i in range(len(y_candidate)):
			y_candidate_prob.append(y_candidate[i][1])
		y_candidate_index = []
		for i in range(len(y_candidate)):
			y_candidate_index.append(y_candidate[i][0])
		# process of choosing 'n_pick' number of candidates among 7
		ix_candidate = pick_candidate(y_candidate_prob, temperature=1, n_pick=1)
		ix_new = y_candidate[ix_candidate[0]][0]
		X_new = X_old
	return X_new, ix_new # return the index 'ix'
# model 4 (generate two sentences using two previous steps)
def generate_text_two_to_two(model, doc2vec_model, ix_to_char, char_to_ix, model2, doc2vec_model2, ix_to_char2, X_old, ix_list, i):
	if(i<1): #  since we cannot put two inputs at first(X_-1, X_0), we need to use previous model to generate the first input X_1. after this step, we can use two X_i's
		X_new, ix_new = generate_text_all_previous_step(model, doc2vec_model, X_old, ix_list[i], i)
	else:
		#1. ix_list받은거를 통해서 이제 문장 concatenate 시켜서 한개로 만들고
		sent_temp = ix_to_char[ix_list[i-1]] + ". " + ix_to_char[ix_list[i]]
		#2. 그 다음에 infer_vector통해서 벡터 찾고
		sent_token = tokenize(sent_temp)
		x = doc2vec_model2.infer_vector(sent_token) # is there a way to exclude dot?? (not solved yet)
		similar_doc = doc2vec_model2.docvecs.most_similar([x], topn=1)  # pick the most similar one
		ix_temp = similar_doc[0][0]
		#3. 그거 넣어서 다음 벡터 예측하고
		X_old[0, i, :] = doc2vec_model2.docvecs[ix_temp]  # new input x
		X_temp = X_old[0, i, :].reshape(1, 1, len(X_old[0, i, :]))  # reshape (50,) -> (1,1,50)
		y_vec = model2.predict(X_temp)[0, 0]
		#4. 그 벡터로 또 most_similar해서 리스트에서 문장 1개 뽑고
		y = doc2vec_model2.docvecs.most_similar([y_vec], topn=1)
		next_sent = ix_to_char2[y[0][0]]
		#5. 그 뽑은거 분리시켜서 앞에꺼만 쓰고 그거의 ix찾고 그 ix 해당하는 X벡터 찾아서 집어넣음
		first_sent, _ = next_sent.split(".")
		ix_new = char_to_ix[first_sent]
		X_old[0, i, :] = doc2vec_model.docvecs[ix_new]  # new input x
		X_new = X_old
		#6. 그 ix랑 X 리턴
	return X_new, ix_new# return the index 'ix'
# model 5 (generate two sentences using two previous steps)
def generate_text_three_to_three(model, doc2vec_model, ix_to_char, char_to_ix, model3, doc2vec_model3, ix_to_char3, X_old, ix_list, i):
	if(i<2): #  since we cannot put two inputs at first(X_-1, X_0), we need to use previous model to generate the first input X_1. after this step, we can use two X_i's
		X_new, ix_new = generate_text_all_previous_step(model, doc2vec_model, X_old, ix_list[i], i)
	else:
		#1. ix_list받은거를 통해서 이제 문장 concatenate 시켜서 한개로 만들고
		sent_temp = ix_to_char[ix_list[i-2]] + ". " + ix_to_char[ix_list[i-1]] + ". " + ix_to_char[ix_list[i]]
		#2. 그 다음에 infer_vector통해서 벡터 찾고
		sent_token = tokenize(sent_temp)
		x = doc2vec_model3.infer_vector(sent_token) # is there a way to exclude dot?? (not solved yet)
		similar_doc = doc2vec_model3.docvecs.most_similar([x], topn=1)  # pick the most similar one
		ix_temp = similar_doc[0][0]
		#3. 그거 넣어서 다음 벡터 예측하고
		X_old[0, i, :] = doc2vec_model3.docvecs[ix_temp]  # new input x
		X_temp = X_old[0, i, :].reshape(1, 1, len(X_old[0, i, :]))  # reshape (50,) -> (1,1,50)
		y_vec = model3.predict(X_temp)[0, 0]
		#4. 그 벡터로 또 most_similar해서 리스트에서 문장 1개 뽑고
		y = doc2vec_model3.docvecs.most_similar([y_vec], topn=1)
		next_sent = ix_to_char3[y[0][0]]
		#5. 그 뽑은거 분리시켜서 앞에꺼만 쓰고 그거의 ix찾고 그 ix 해당하는 X벡터 찾아서 집어넣음
		first_sent, _, _ = next_sent.split(".")
		ix_new = char_to_ix[first_sent]
		X_old[0, i, :] = doc2vec_model.docvecs[ix_new]  # new input x
		X_new = X_old
		#6. 그 ix랑 X 리턴
	return X_new, ix_new# return the index 'ix'

# show the candidates made by five different models and pick one of them
def pick_ix_X(ix_set, X_set, ix_to_char, ix_list):
	'''
	:param ix_set: set of ix's that are the candidate of next sentences'
	:param X_set: set of X's that will be used as a next X
	:param ix_to_char: convert index to sentence
	:return: picked ix and X
	'''
	for i in range(len(ix_set)): # show
		print("sentence_{}: ".format(i + 1) + ix_to_char[ix_set[i]])
	number = input("please choose one sentece: ") # pick
	print("\n")
	ix = ix_set[int(number) - 1]  # next index
	X = X_set[int(number) - 1]
	ix_list.append(ix)
	return ix, X, ix_list