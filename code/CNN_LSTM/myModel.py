import os
from keras import backend as K

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score

import math
import random
from keras import optimizers
import numpy as np
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score


from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Layer,Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import  GRU, Bidirectional
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, GlobalAveragePooling2D, SpatialDropout2D
from keras.callbacks import History 
from keras.models import Model

from keras.layers.noise import GaussianNoise

from collections import Counter

from sklearn.utils import class_weight


def cnn_block(input_shape):
	input = Input(shape=input_shape)
	dp_conv = 0.4
	act_conv = 'relu'
	x = GaussianNoise(0.0005)(input)
	x = Conv2D(64, (3, 1), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation(act_conv)(x)
	x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
	x = Conv2D(128, (3, 1), strides=(1, 1), padding='same')(input)
	x = BatchNormalization()(x)
	x = Activation(act_conv)(x)
	for i in range(8):
		x = Conv2D(256, (3, 1), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation(act_conv)(x)
		x = MaxPooling2D(padding="same", pool_size=(2, 1))(x)
	flatten1 = Flatten()(x)

	cnn_eeg = Model(inputs=input, outputs=flatten1)
	return cnn_eeg

def build_model(data_dim, n_channels, n_cl):
	eeg_channels = 3
	hidden_units  = 256
	
	init_conv = 'glorot_normal'
	dp = 0.4
	
	input_eeg = Input(shape=( None,data_dim,1, n_channels))
	cnn_eeg = cnn_block(( data_dim,1, n_channels))
	
	print(cnn_eeg.summary())

	x_eeg = TimeDistributed(cnn_eeg)(input_eeg)



	x = BatchNormalization()(x_eeg)
	x = Bidirectional(LSTM(units=32,
               return_sequences=True, activation='tanh',
               recurrent_activation='sigmoid', dropout = dp, recurrent_dropout = dp))(x)
	x = BatchNormalization()(x)
	

	x = Bidirectional(LSTM(units=32,
               return_sequences=True, activation='tanh',
               recurrent_activation='sigmoid', dropout = dp, recurrent_dropout = dp))(x)
	x = BatchNormalization()(x)
	
	predictions = TimeDistributed(Dense(units=n_cl, activation='softmax', kernel_initializer=init_conv))(x)


	model = Model(inputs=[input_eeg] , outputs=[predictions])
	return [ model]
