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
from keras.models import load_model



class AttLayer(Layer):
	def __init__(self, **kwargs):
		self.init = initializers.get('normal')
		super(AttLayer, self).__init__(**kwargs)
		
	def build(self, input_shape):
		assert len(input_shape)==3
		self.W = self.add_weight(name='kernel', 
									shape=(input_shape[-1],),
									initializer='normal',
									trainable=True)
		super(AttLayer, self).build(input_shape)

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


print("Loading Model .......................")

model = load_model('nlpOutput2.h5')

lyrics = "If I should stay\nI would only be in your way\nSo I'll go but I know\nI'll think of you every step of the way\n\n\nOoh\nOoh "

print("Loaded Model yay!")


y_pred = model.predict(lyrics)

print(y_pred)
