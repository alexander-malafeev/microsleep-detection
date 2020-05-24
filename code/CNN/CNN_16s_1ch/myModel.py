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

def build_model(data_dim, n_channels, n_cl):
	eeg_channels = 1

	act_conv = 'relu'
	init_conv = 'glorot_normal'
	dp_conv = 0.3
	def cnn_block(input_shape):
		input = Input(shape=input_shape)
		x = GaussianNoise(0.0005)(input)
		x = Conv2D(32, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
		x = BatchNormalization()(x)
		x = Activation(act_conv)(x)
		x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		
		x = Conv2D(64, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
		x = BatchNormalization()(x)
		x = Activation(act_conv)(x)
		x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		for i in range(4):
			x = Conv2D(128, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
			x = BatchNormalization()(x)
			x = Activation(act_conv)(x)
			x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		for i in range(6):
			x = Conv2D(256, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
			x = BatchNormalization()(x)
			x = Activation(act_conv)(x)
			x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		flatten1 = Flatten()(x)
		cnn_eeg = Model(inputs=input, outputs=flatten1)
		return cnn_eeg
		
	hidden_units1  = 256
	dp_dense = 0.5

	eeg_channels = 1
	eog_channels = 2

	input_eeg = Input(shape=( data_dim, 1,  1))
	cnn_eeg = cnn_block(( data_dim, 1, 1))
	x_eeg = cnn_eeg(input_eeg)
	x = BatchNormalization()(x_eeg)
	x = Dropout(dp_dense)(x)
	x =  Dense(units=hidden_units1, activation=act_conv, kernel_initializer=init_conv)(x)
	x = BatchNormalization()(x)
	x = Dropout(dp_dense)(x)


	predictions = Dense(units=n_cl, activation='softmax', kernel_initializer=init_conv)(x)

	model = Model(inputs=[input_eeg] , outputs=[predictions])
	return [cnn_eeg, model]
