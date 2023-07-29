import os

import keras

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score
import scipy.io 
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
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D
from keras.callbacks import History 
from keras.models import Model

from keras.layers.noise import GaussianNoise

from collections import Counter

from sklearn.utils import class_weight

from myModel import build_model

from os import listdir
from os.path import isfile, join

import sys
sys.path.append("..")
from loadData import  *


out_dir = './pred/'


data_dir = './../../../data/files/'
f_set = './../../../data/file_sets.mat'

create_tmp_dirs([out_dir,  out_dir+'/val/'])

files_val = []
files_test = []
files_train = []
	
	
mat = spio.loadmat(f_set)
	
	
	
tmp =  mat['files_val']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_val.extend(file)
tmp =  mat['files_test']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_test.extend(file)



n_rec = 8
batch_size = 128
n_ep = 30
fs  = 200;
w1 = 200;
step = 50; #0.25 s
# half_size of the sliding window in samples
w_len =  (20*50+200)
data_dim = w1
half_prec = 50
prec = 1
n_cl = 2

print("=====================")
print("Reading dataset to predict:")
(  data_val, targets_val, N_samples_val) = load_data(data_dir,files_val, w_len)

(  data_test, targets_test, N_samples_test) = load_data(data_dir,files_test, w_len)


ordering = 'channels_last';
keras.backend.set_image_data_format(ordering)



sample_list_val = []
for i in range(len(targets_val)):
	sample_list_val.append([])
	l= len(targets_val[i][0])
	kk = (len(targets_val[i][0])-w1-w_len)//prec
	wnd_end = kk*prec+w1
	sample_list_val[i].append([i, w_len//2, wnd_end+w_len//2, 0])
	
sample_list_val2 = []
for i in range(len(targets_val)):
	sample_list_val2.append([])
	l= len(targets_val[i][1])
	kk = (len(targets_val[i][1])-w1-w_len)//prec
	wnd_end = kk*prec+w1
	sample_list_val2[i].append([i, w_len//2, wnd_end+w_len//2, 1])	
 
 
 
sample_list_test = []
for i in range(len(targets_test)):
	sample_list_test.append([])
	l= len(targets_test[i][0])
	kk = (len(targets_test[i][0])-w1-w_len)//prec
	wnd_end = kk*prec+w1
	sample_list_test[i].append([i, w_len//2, wnd_end+w_len//2, 0])
	
sample_list_test2 = []
for i in range(len(targets_test)):
	sample_list_test2.append([])
	l= len(targets_test[i][1])
	kk = (len(targets_test[i][1])-w1-w_len)//prec
	wnd_end = kk*prec+w1
	sample_list_test2[i].append([i, w_len//2, wnd_end+w_len//2, 1])	
		

n_channels = 3

def my_generator(data_train, targets_train, sample_list, shuffle = True, batch_size = 200):
	if shuffle:
		random.shuffle(sample_list)
	while True:
		for batch in batch_generator(sample_list, batch_size):
			batch_data = []
			batch_targets = []
			for sample in batch:
				[f, b, e, c] = sample
				sample_xx1 = data_train[f][c][b:e]
				sample_xx2 = data_train[f][2][b:e]
				sample_xx = np.concatenate( ( sample_xx1, sample_xx2 ), axis = 2 )
				sample_yy = targets_train[f][c][b:e]
				sample_x = []
				sample_y = []
				z = 0
				while z<len(sample_xx)-w1:
					sample_x.append(sample_xx[z:z+w1])
					tmp_lbl  =   sample_yy[z+w1//2-20:z+w1//2+20]
					a = Counter(tmp_lbl)
					r  = a.most_common(1)[0][0]
					sample_y.append(r)
					z += step
				trgt =  np.stack(sample_y, axis=0)
				sample_x = np.stack(sample_x, axis=0)
				batch_data.append(sample_x)
				batch_targets.append( np.stack(trgt, axis=0))
			batch_data = np.stack(batch_data, axis=0)
			batch_targets = np.stack(batch_targets, axis=0)
			batch_targets = np.array(batch_targets)
			batch_targets_tmp = np.copy(batch_targets)
			batch_targets[batch_targets==2] = 0
			batch_targets[batch_targets==3] = 0
			batch_targets = np_utils.to_categorical(batch_targets, n_cl)	
			
			batch_data = (batch_data +100)/200 
			batch_data = np.clip(batch_data, 0, 1)
			y = batch_targets
			s_w = np.zeros((y.shape[0], y.shape[1]))
			yield batch_data, batch_targets #, s_w

def val_data_to_batch(data, targets):
	batch_data = []
	batch_targets = []
	for j in range(len(targets)):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		b = wnd_begin
		e = wnd_end
		sample_label = targets[j]
		sample_x = data[b:e+1]
		batch_data.append(sample_x)
		batch_targets.append(sample_label)
	batch_data = np.stack(batch_data, axis=0)
	batch_targets = np_utils.to_categorical(batch_targets, n_cl)
	batch_data = (batch_data )/100 #- np.mean(batch_data, axis=1)
	np.clip(batch_data, -1, 1)
	return batch_data, batch_targets		
			

[ model] = build_model(data_dim, n_channels, n_cl)


Nadam = optimizers.Nadam( )
model.compile(optimizer=Nadam,  loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode=None)

model.load_weights('./model.h5')

y_ = []
y = []

y_p = []


f_list = files_val
for j in range(0,len(f_list)):
	f = f_list[j]
	generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
	scores = model.evaluate_generator( generator_val, 1, workers=1)
	generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
	y_pred = model.predict_generator( generator_val, 1, workers=1)

	val_y_tmp = []
	generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
	for ii in range(int(math.ceil((len(sample_list_val[j],)+0.0)/batch_size))):
		[x, y] = next(generator_val)
		for k in range(y.shape[0]):
			val_y_tmp.append( y[k] )

	val_y_tmp = np.stack(val_y_tmp, axis=0)
	y_ =  np.argmax(y_pred, axis=2).flatten() 
	y = np.argmax(val_y_tmp, axis=2).flatten()
	y_p = scores 
  
	scipy.io.savemat( out_dir+'/val/'+f+'.mat', mdict={ 'y_p':y_p,  'y_': y_, 'y':y}
