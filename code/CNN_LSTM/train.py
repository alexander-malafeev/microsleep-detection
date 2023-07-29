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
sys.path.append("..")
from loadData import  *
from utils import *

n_rec = 8
batch_size = 128
n_ep = 8
fs  = 200;
w1 = 200;
step = 50; #0.25 s
# half_size of the sliding window in samples
w_len =  (20*50+200) 
data_dim = w1
half_prec = 50
prec = 1
n_cl = 2


data_dir = './../../../data/files/'
f_set = './../../../data/file_sets.mat'

create_tmp_dirs(['./models/',  './predictions/'])

mat = spio.loadmat(f_set)
	
files_train = []
files_val = []
files_test = []
	
	
skip = 10
	
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

		
def my_generator(data_train, targets_train, sample_list, shuffle = True, batch_size = 200):
	if shuffle:
		random.shuffle(sample_list)
	sample_list = sample_list[::skip]
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
				print(len(sample_x))
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
			y = batch_targets_tmp
			s_w = np.zeros((y.shape[0], y.shape[1]))
			for i in range(y.shape[0]):
				for j in range(y.shape[1]):
					s_w[i, j] = cl_w[y[i,j]]	
			yield batch_data, batch_targets, s_w

			
	

n_channels = 3

st0 = classes_global(data_dir, files_train)
cl_w = [ 0.28157203,  3.57512078, 35.6736615,   7.10387778]
cl_w[2] = 0
cl_w[3] = 0

print("=====================")
print("class weights ")
print(cl_w)
print("=====================")
print("=====================")
print("Reading training dataset:")
(  data_train, targets_train, N_samples) = load_data(data_dir,files_train, w_len)
N_samples = N_samples
print('N_samples ',N_samples)
print('w_len ',w_len)
print('batch_size ',batch_size)

N_batches = int(math.ceil((N_samples+0.0)/(batch_size)))
print('N batches ',N_batches)

print("=====================")
print("Reading validation dataset:")
(  data_val, targets_val, N_samples_val) = load_data(data_dir,files_val, w_len)


# create indexes of samples 
# each element is [file number in data_train, index in its targets, index of the beginning, index of the end of the window]
def create_sample_list(targets_train):
	sample_list = []
	for ch in range(2):
		for i in range(len(targets_train)):
			l= len(targets_train[i])
			for j in range((len(targets_train[i][0])-2*w_len)//prec):
				mid = j*prec
				# we add the padding size
				mid += w_len//2
				wnd_begin = mid-w_len//2
				wnd_end = mid+w_len//2
				sample_list.append([i,wnd_begin, wnd_end, ch ])
	return sample_list



sample_list_val = []
for i in range(len(targets_val)):
	sample_list_val.append([])
	l= len(targets_val[i][0])
	kk = (len(targets_val[i][0])-w1-w_len)//prec
	wnd_end = kk*prec+w1
	sample_list_val[i].append([i, w_len//2, wnd_end+w_len//2, 0])


			
ordering = 'channels_last';
keras.backend.set_image_data_format(ordering)


learning_rate = 0.1
decay_rate = learning_rate / n_ep


[ model] = build_model(data_dim, n_channels, n_cl)
Nadam = optimizers.Nadam( clipnorm=1.)
model.compile(optimizer=Nadam,  loss='categorical_crossentropy', metrics=['accuracy'],  sample_weight_mode="temporal")
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

sample_list = create_sample_list(targets_train)
for i in range(n_ep):
	print("Epoch = " + str(i))
	generator_train = my_generator(data_train, targets_train, sample_list, shuffle = True, batch_size = batch_size)
	model.fit_generator(generator_train, steps_per_epoch = N_batches//skip,  epochs = 1, verbose=1, initial_epoch=0 )

	val_y_ = []
	val_y = []
	loss_val_tmp = []
	f_list = files_val
	val_l = []
	for j in range(len(data_val)):
		f = f_list[j]
		generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
		scores = model.evaluate_generator( generator_val, 1, workers=1)
		print(scores)
		generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
		y_pred = model.predict_generator( generator_val, 1, workers=1)
		val_y_tmp = []
		generator_val = my_generator(data_val, targets_val, sample_list_val[j], shuffle = False)
		for ii in range(int(math.ceil((len(sample_list_val[j],)+0.0)/batch_size))):
			[x, y, _] = next(generator_val)
			for k in range(y.shape[0]):
				val_y_tmp.append( y[k] )
		
		loss_val_tmp.append(scores[0])

		val_y_tmp = np.stack(val_y_tmp, axis=0)
		val_l.append(len(np.argmax(y_pred, axis=2).flatten()))
		val_y_.extend( np.argmax(y_pred, axis=2).flatten() )
		val_y.extend(  np.argmax(val_y_tmp, axis=2).flatten() )
	
	loss_val.append(np.mean(loss_val_tmp))
	t1 = kappa_metric( val_y, val_y_, n_cl )
	K_val_tmp = t1
	K_val[i,:] = K_val_tmp
	t2 = cohen_kappa_score(val_y, val_y_)
	acc_val.append(t2)
	print( "K val per class = ", t1 )
	print( "K val = ", t2 )
	print( "loss val = ", loss_val[-1] )
	spio.savemat('./predictions_'+str(i)+'.mat', mdict={'val_y': val_y, 'val_y_': val_y_, 'val_l': val_l,  'files_test':files_test, 'files_val':files_val, 'acc_tr': acc_tr, 'loss_tr':loss_tr, 'loss_val':loss_val,  'acc_val':acc_val,  'K_val':K_val  })
	model.save('./models/model_ep'+str(i)+'.h5')
	

model.save('./model.h5')
spio.savemat('./predictions.mat', mdict={'val_y': val_y, 'val_y_': val_y_, 'val_l': val_l,  'files_test':files_test, 'files_val':files_val, 'acc_tr': acc_tr, 'loss_tr':loss_tr, 'loss_val':loss_val,  'acc_val':acc_val,  'K_val':K_val  })
