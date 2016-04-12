import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

class hella_rfs():
	def __init__(self, use_weights = False):
		self.rfs = []
		self.use_weights = use_weights
	def fit(self,train_X,train_Y):
		#split set into ones and zeros
		zeros = train_X[train_Y == 0,:]
		ones = train_X[train_Y == 1,:]
		num_ones = ones.shape[0]
		# compute number of chunks to split
		num_chunks = int(zeros.shape[0]/num_ones)
		chunks = np.array_split(zeros,num_chunks)
		i = 0
		for chunk in chunks:
			print('training random forest %s of %s' %(i,num_chunks))
			chunk_rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
			chunk_train_X = np.concatenate([chunk,ones])
			chunk_train_Y = np.concatenate([np.zeros([chunk.shape[0],1]),np.ones([num_ones,1])]).ravel()
			chunk_rf.fit(chunk_train_X,chunk_train_Y)
			self.rfs.append(chunk_rf)
			i+=1
	def predict(self,test_df):
		test_X = np.array(test_df)
		predictions = []
		for rf_i in self.rfs:
			predictions.append(rf_i.predict(test_X))
		predictions = np.array(predictions).T
		predictions = np.array(mode(predictions,axis = 1).mode).reshape([test_X.shape[0],1])
		return(predictions)


