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

import sys
sys.path.append("../..")
from loadData import  *
from utils import *

batch_size = 200
n_ep = 3
fs  = 200;
w_len = 1*fs;
data_dim = w_len*2
half_prec = 0.5
prec = 1
n_cl = 4




data_dir = './../../../data/files/'
f_set = './../../../data/file_sets.mat'

create_tmp_dirs(['./models/',  './predictions/'])

mat = spio.loadmat(f_set)
	
files_train = []
files_val = []
files_test = []
	
	
	
tmp =  mat['files_train']
for i in range(len(tmp)):
	file = [str(''.join(l)) for la in tmp[i] for l in la]
	files_train.extend(file)
	
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
			batch_data1 = (batch_data1 )/100 #- np.mean(batch_data, axis=1)
			batch_data1 = np.clip(batch_data1, -1, 1)
			yield [ batch_data1 ], batch_targets


n_channels = 3

st0 = classes_global(data_dir, files_train)
cls = np.arange(n_cl)
cl_w = class_weight.compute_class_weight('balanced', cls, st0)
print(cl_w)

print("=====================")
print("class weights ")
print(cl_w)
print("=====================")
print("=====================")
print("Reading training dataset:")
(  data_train, targets_train, N_samples) = load_data(data_dir,files_train, w_len)
print('N_samples ',N_samples)
N_batches = int(math.ceil((N_samples+0.0)/batch_size))
print('N batches ',N_batches)

print("=====================")
print("Reading validation dataset:")
(  data_val, targets_val, N_samples_val) = load_data(data_dir,files_val, w_len)



# create indexes of samples 
# each element is [file number in data_train, index in its targets, index of the beginning, index of the end of the window]
sample_list = []
for ch in range(2):
	for i in range(len(targets_train)):
		for j in range(len(targets_train[i][0])):
			mid = j*prec
			# we add the padding size
			mid += w_len
			wnd_begin = mid-w_len
			wnd_end = mid+w_len-1
			sample_list.append([i,j,wnd_begin, wnd_end, ch ])

sample_list_val = []

for i in range(len(targets_val)):
	sample_list_val.append([])
	for j in range(len(targets_val[i][0])):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		sample_list_val[i].append([i,j,wnd_begin, wnd_end, 0 ])
		
			
ordering = 'channels_last';


keras.backend.set_image_data_format(ordering)




[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)
Nadam = optimizers.Nadam( )
model.compile(optimizer='Nadam',  loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode=None)
print(cnn_eeg.summary())
print(model.summary())



print(model.metrics_names)

history = History()

K_cv_tmp = []
K_tst_tmp = []
K_val = np.zeros( (n_ep, n_cl) )
K_tst = np.zeros( (n_ep, n_cl) )

acc_val = []
acc_tst = []
acc_tr = []
K_tr = []
loss_tr = []
loss_tst = []
loss_val = []
N_steps = 1000


for i in range(n_ep):
	print("Epoch = " + str(i))
	generator_train = my_generator(data_train, targets_train, sample_list)

	model.fit_generator(generator_train, steps_per_epoch = N_batches, class_weight = cl_w,  epochs = 1, verbose=1,  callbacks=[history], initial_epoch=0 )
	
	acc_tr.append(history.history['acc'])
	loss_tr.append(history.history['loss'])

	val_y_ = []
	val_y = []
	loss_val_tmp = []
	f_list = files_val
	val_l = []
	for j in range(len(data_val)):
		f = f_list[j]
		generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
		scores = model.evaluate_generator( generator_val, int(math.ceil((len(sample_list_val[j],)+0.0)/batch_size)), workers=1)
		generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
		y_pred = model.predict_generator( generator_val, int(math.ceil((len(sample_list_val[j],)+0.0)/batch_size)), workers=1)
		val_l.append(len(sample_list_val[j]))
		print(len(sample_list_val[j]))
		print(len(targets_val[j][0]))
		loss_val_tmp.append(scores[0])
		val_y_.extend( np.argmax(y_pred, axis=1).flatten() )
		val_y.extend( targets_val[j][0])
	
	loss_val.append(np.mean(loss_val_tmp))
	t1 = kappa_metric( val_y, val_y_, n_cl )
	K_val_tmp = t1
	K_val[i,:] = K_val_tmp
	t2 = cohen_kappa_score(val_y, val_y_)
	acc_val.append(t2)
	print( "K val per class = ", t1 )
	print( "K val = ", t2 )
	print( "loss val = ", loss_val[-1] )
			
	model.save('./models/model_ep'+str(i)+'.h5')
	spio.savemat('./predictions/predictions_ep'+str(i)+'.mat', mdict={'val_y': val_y, 'val_y_': val_y_, 'val_l': val_l,  'files_test':files_test, 'files_val':files_val, 'acc_tr': acc_tr, 'loss_tr':loss_tr, 'loss_val':loss_val,  'acc_val':acc_val,  'K_val':K_val}  )
	

model.save('./model.h5')


spio.savemat('./predictions.mat', mdict={'val_y': val_y, 'val_y_': val_y_, 'val_l': val_l,  'files_test':files_test, 'files_val':files_val, 'acc_tr': acc_tr, 'loss_tr':loss_tr, 'loss_val':loss_val,  'acc_val':acc_val,  'K_val':K_val  })


