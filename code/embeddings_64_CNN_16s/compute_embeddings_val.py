import os

import keras

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score


import math
import random
from keras import optimizers
import numpy as np
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score
np.random.seed(0)

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Layer,Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import  GRU, Bidirectional
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, GlobalAveragePooling2D
from keras.callbacks import History 
from keras.models import Model

from keras.layers.noise import GaussianNoise

from collections import Counter

from sklearn.utils import class_weight

from myModel import build_model



batch_size = 200
n_ep = 2
fs  = 200;
# half_size of the sliding window in samples
w_len = 8*fs;
data_dim = w_len*2
half_prec = 0.5
prec = 1
n_cl = 4

img_dir = './tSNE_val/'

import sys
sys.path.append("..")
from loadData import  *
from utils import *

data_dir = './../../../data/files/'
f_set = './../../../data/file_sets.mat'

create_tmp_dirs([img_dir])

mat = spio.loadmat(f_set)
	
files_train = []
files_val = []
files_test = []
	
	
	
tmp =  mat['files_train']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_train.extend(file)
print(files_train)
tmp =  mat['files_val']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_val.extend(file)
tmp =  mat['files_test']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_test.extend(file)


def my_generator(data_train, targets_train, sample_list, shuffle = True):
	if shuffle:
		random.shuffle(sample_list)
	while True:
		for batch in batch_generator(sample_list, batch_size):
			batch_data1 = []
			batch_data2 = []
			batch_targets = []
			for sample in batch:
				[f, s, b, e, c] = sample
				sample_label = targets_train[f][c][s]
				sample_x1 = data_train[f][c][b:e+1]
				sample_x2 = data_train[f][2][b:e+1]
				sample_x = np.concatenate( ( sample_x1, sample_x2 ), axis = 2 )
				batch_data1.append(sample_x)
				batch_targets.append(sample_label)
			batch_data1 = np.stack(batch_data1, axis=0)
			batch_targets = np.array(batch_targets)
			batch_targets = np_utils.to_categorical(batch_targets, n_cl)
			batch_data1 = (batch_data1 )/100
			batch_data1 = np.clip(batch_data1, -1, 1)
			yield [ batch_data1 ], batch_targets

n_channels = 3



ordering = 'channels_last';
keras.backend.set_image_data_format(ordering)


print("Building model")
[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)
Nadam = optimizers.Nadam( )
model.compile(optimizer=Nadam,  loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode=None)
print(cnn_eeg.summary())
print(model.summary())
model.load_weights('./model.h5')
print("Done")

f_list = files_val

cnn_eeg.compile(optimizer=Nadam,  loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode=None)


model.load_weights('./model.h5')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

p_s = 5

X_g = []
Y_g = []
batch_size_val = 200
f_list = files_val
val_l = []
print('validation set')
print(f_list[0])
print('====================')
for j in range(len(f_list)):
	val_y_ = []
	val_y = []
	f = f_list[j]
	print('reading file ', f)
	data, targets, N_samples  = load_data(data_dir, [f_list[j]], w_len)
	print('Done ')
	print("Building validation sample list")
	sample_list_val = []
	for i in range(len(targets)):
		sample_list_val.append([])
		for j in range(len(targets[i][0])):
			mid = j*prec
			# we add the padding size
			mid += w_len
			wnd_begin = mid-w_len
			wnd_end = mid+w_len-1
			sample_list_val[i].append([i,j,wnd_begin, wnd_end, 0 ])
	print("Done")		
 
 
 
	generator_val = my_generator(data, targets, sample_list_val[0], shuffle = False)
	y_pred = cnn_eeg.predict_generator( generator_val, int(math.ceil((len(sample_list_val[0])+0.0)/batch_size_val)), workers=1)
	val_l.append(len(sample_list_val[0]))
	print(y_pred.shape)
	for k in range(y_pred.shape[0]):
		val_y_.append( y_pred[k])
	generator_val = my_generator(data, targets, sample_list_val[0], shuffle = False)
	for ii in range(int(math.ceil( (len(sample_list_val[0])+0.0)/batch_size_val) )):
		[x, y] = next(generator_val)
		for k in range(y.shape[0]):
			val_y.append( y[k] )
	val_y = val_y[::100]
	val_y_ = val_y_[::100]
	val_y = np.stack(val_y, axis=0)
	val_y_ = np.stack(val_y_, axis=0)

	X = val_y_
	Y = val_y
	X_g = []
	Y_g = []
	for l in range(0, len(X), 10):
		X_g.append(X[l])
		Y_g.append(Y[l])
	embeddings = []
	emb_color = []
	for i in range(len(X)):
		output_class = np.argmax(Y[i,:])
		embedding = X[i,:]
		embeddings.append(embedding[0])
		if (output_class == 0):
			emb_color.append("#0000ff")
		elif (output_class == 1):
			emb_color.append("#ff0000")
		elif (output_class == 2):
			emb_color.append("#00ff00")
		elif (output_class == 3):
			emb_color.append("#ff00ff")


	tsne = TSNE(n_components=2, random_state=0)
	emb_2d = tsne.fit_transform(X)



	plt.figure(figsize=(8, 8))
	plt.scatter(x = emb_2d[:,0], y=emb_2d[:,1], color=emb_color, s=p_s)
	plt.savefig(img_dir+f+'.png')
