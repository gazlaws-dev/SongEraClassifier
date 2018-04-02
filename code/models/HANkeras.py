import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

data_train = pd.read_csv('lyrics.csv')
print data_train.shape

from nltk import tokenize

lyrics = []
labels = []
texts = []

#to get the decade
data_train.Year = np.divide(data_train.Year,10)
data_train.Year = np.multiply(data_train.Year,10)


for idx in range(data_train.Lyrics.shape[0]):
	text = BeautifulSoup(data_train.Lyrics[idx])
	year =data_train.Year[idx]
	if(year<1980 or year >=2010):
		continue
	year = (year-1980)/10;	#1980,90,2000
	text = text.getText()
	text = text.encode('ascii','ignore')
	texts.append(text)
	sentences = tokenize.sent_tokenize(text)
	lyrics.append(sentences)	
	labels.append(year)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(lyrics):
	for j, sent in enumerate(sentences):
		if j< MAX_SENTS:
			wordTokens = text_to_word_sequence(sent)
			k=0
			for _, word in enumerate(wordTokens):
				if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
					data[i,j,k] = tokenizer.word_index[word]
					k=k+1					
					
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


GLOVE_DIR = "/home/gaz/data/glove/glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

#############################################################
def f1_score(y_true, y_pred):

	# Count positive samples.
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	# If there are no true samples, fix the F1 score at 0.
	if c3 == 0:
		return 0

	# How many selected items are relevant?
	precision = c1 / c2

	# How many relevant items are selected?
	recall = c1 / c3

	# Calculate f1_score
	f1_score = 2 * (precision * recall) / (precision + recall)
	return f1_score


def precision(y_true, y_pred):

	# Count positive samples.
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	# If there are no true samples, fix the F1 score at 0.
	if c3 == 0:
		return 0

	# How many selected items are relevant?
	precision = c1 / c2

	return precision


def recall(y_true, y_pred):

	# Count positive samples.
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	# If there are no true samples, fix the F1 score at 0.
	if c3 == 0:
		return 0

	recall = c1 / c3

	return recall

#for Baseline check
####################################################33


#embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
#for word, i in word_index.items():
#	embedding_vector = embeddings_index.get(word)
#	if embedding_vector is not None:
#		# words not found in embedding index will be all-zeros.
#		embedding_matrix[i] = embedding_vector
#		
#embedding_layer = Embedding(len(word_index) + 1,
#							EMBEDDING_DIM,
#							weights=[embedding_matrix],
#							input_length=MAX_SENT_LENGTH,
#							trainable=True)
#sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
#embedded_sequences = embedding_layer(sentence_input)
#l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
#sentEncoder = Model(sentence_input, l_lstm)

#Lyrics_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
#Lyrics_encoder = TimeDistributed(sentEncoder)(Lyrics_input)
#l_lstm_sent = Bidirectional(LSTM(100))(Lyrics_encoder)
#preds = Dense(3, activation='softmax')(l_lstm_sent)
#model = Model(Lyrics_input, preds)

#model.compile(loss='categorical_crossentropy',
#			  optimizer='rmsprop',
#			  metrics=['accuracy', recall])

#print("model fitting - Hierachical LSTM")
#print model.summary()
#history_callback = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=1, batch_size=200)
#loss_history = history_callback.history["loss"]
#numpy_loss_history = np.array(loss_history)
#np.savetxt("loss_history1.txt", numpy_loss_history, delimiter=",")
#model.save('nlpOutput1.h5')

# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector
		
embedding_layer = Embedding(len(word_index) + 1,
							EMBEDDING_DIM,
							weights=[embedding_matrix],
							input_length=MAX_SENT_LENGTH,
							trainable=True)

class AttLayer(Layer):
	def __init__(self, **kwargs):
		self.init = initializers.get('normal')
		#self.input_spec = [InputSpec(ndim=3)]
		super(AttLayer, self).__init__(**kwargs)
		
	def build(self, input_shape):
		assert len(input_shape)==3
		self.W = self.add_weight(name='kernel', 
									shape=(input_shape[-1],),
									initializer='normal',
									trainable=True)
		#?self.trainable_weights = [self.W]
		super(AttLayer, self).build(input_shape)
		
#	def build(self, input_shape):
#		assert len(input_shape)==3
#		#self.W = self.init((input_shape[-1],1))
#		self.W = self.init((input_shape[-1],))
#		#self.input_spec = [InputSpec(shape=input_shape)]
#		self.trainable_weights = [self.W]
#		super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

	def call(self, x, mask=None):
		eij = K.tanh(K.dot(x, self.W))
		
		ai = K.exp(eij)
		weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
		
		weighted_input = x*weights.dimshuffle(0,1,'x')
		return weighted_input.sum(axis=1)
	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[-1])

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

Lyrics_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
Lyrics_encoder = TimeDistributed(sentEncoder)(Lyrics_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(Lyrics_encoder)	#
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(3, activation='softmax')(l_att_sent)	#change to number of classes
model = Model(Lyrics_input, preds)

model.compile(loss='categorical_crossentropy',
			  optimizer='rmsprop',
			  metrics=['accuracy',  recall])

print("model fitting - Hierachical attention network")
print model.summary()
history_callback = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=10, batch_size=200)
loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history2.txt", numpy_loss_history, delimiter=",")
model.save('nlpOutput2.h5')
