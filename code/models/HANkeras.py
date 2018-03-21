import tensorflow as tf
import numpy
import json
from keras.engine.topology import Layer
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import TimeDistributed, Bidirectional, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, Dropout
from keras.layers import Dense, Input
from keras import backend as K
from keras import optimizers

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
#The dataset can be loaded directly. Because the output variable contains strings, it is easiest to load the data using pandas. We can then split the attributes (columns) into input variables (X) and output variables (Y)

dataframe = pandas.read_csv("data/billboard_lyrics_1964-2015.csv", header='infer')	#infer may not work, lookup pandas.read_csv
dataset = dataframe.values	#Numpy representation of NDFrame
print(dataset[0,3])
#exit();
X = dataset[:,4]	#lyrics as a string
Y = dataset[:,3].astype(int)		#astype(int)?	#year as a number
						# 1958-2015, should divide into decades for output
Y = numpy.divide(Y, 10)
Y = numpy.multiply(Y, 10)	#1950,60,70,80,90,2000,2010	//actually only 6 values?

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
#vector of integers to a one hot encoding using the Keras function to_categorical().
dummy_y = np_utils.to_categorical(encoded_Y)


print "Finished inputting training data"
#1 input -> [8 hidden nodes] -> 6 outputs


#define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=1, activation='relu'))
	model.add(Dense(6, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


#We can now create our KerasClassifier for use in scikit-learn.

#We can also pass arguments in the construction of the KerasClassifier class that will be passed on to the fit() function internally used to train the neural network. Here, we pass the number of epochs as 2 and batch size as 100 to use when training the model. Debugging is also turned off when training by setting verbose to 0.

estimator = KerasClassifier(build_fn=baseline_model, epochs=2, batch_size=100, verbose=0)

#model evaluation

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

print "Evaluating"	

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




