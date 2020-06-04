import math
import random
import numpy as np
from sklearn.metrics import cohen_kappa_score

def create_tmp_dirs(tmp_dirs):
	for tmp_dir in tmp_dirs:
		if not os.path.isdir(tmp_dir):
			os.mkdir(tmp_dir) 

def kappa_metric(y_true, y_pred, n_cl = 4):
	# computes Cohen kappa per class
	y =  np.array(y_true) 
	y_ = np.array(y_pred) 
	res = []
	for c in range(n_cl):
		res.append(cohen_kappa_score(y==c, y_==c))
	return np.array(res)


def batch_generator(samples, batch_size):
	batch = []
	k = 0 # k is the number of processed samples
	for sample in samples:
		if k % batch_size == 0:
			batch  = [sample]
			k = 1
		else:
			batch.append(sample)
			k += 1
		if k % batch_size == 0:
			yield batch
	if len(batch)>0:
		yield batch
