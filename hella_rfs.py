import numpy as np
#import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix
from scipy.stats import mode
from sklearn import cross_validation

class hella_rfs():
	def __init__(self, use_weights = False):
		self.rfs = []
		#use_weights if true will hold a small set aside for cross validation, which will determine how much each random forest contributes to the committee
		if use_weights:
			self.weights = []
		else:
			self.weights = None
	def fit(self,train_X,train_Y):
		#split set into ones and zeros
		zeros = train_X[train_Y == 0,:]
		ones = train_X[train_Y == 1,:]
		num_ones = ones.shape[0]
		# compute number of chunks to split
		num_chunks = int(zeros.shape[0]/num_ones)
		chunks = np.array_split(zeros,num_chunks)
		#train rfs
		i = 0
		for chunk in chunks:
			
			print('training random forest %s of %s' %(i,num_chunks))
			chunk_rf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
			print(chunk_rf.get_params())
			chunk_train_X = np.concatenate([chunk,ones])
			chunk_train_Y = np.concatenate([np.zeros([chunk.shape[0],1]),np.ones([num_ones,1])]).ravel()
			#cross_validation
			if self.weights is not None:
				print('cross_validation')
				scores = cross_validation.cross_val_score(chunk_rf, chunk_train_X, chunk_train_Y, cv = 10, n_jobs = -1)
				print(scores.mean())
				self.weights.append(scores.mean())
			#train
			chunk_rf.fit(chunk_train_X,chunk_train_Y)
			self.rfs.append(chunk_rf)
			i+=1
	def predict(self,test_df):
		test_X = np.array(test_df)
		predictions = []
		for rf_i in self.rfs:
			predictions.append(rf_i.predict(test_X))
		predictions = np.array(predictions).T
		if self.weights is None:
			predictions = np.array(mode(predictions,axis = 1).mode).reshape([test_X.shape[0],1])
		else:
			self.weights /= sum(self.weights)
			#convex combination of predictions
			predictions *= self.weights
			predictions = predictions.sum(axis = 1) 
			#threshold
			predictions = [1 if p >= 0.5 else 0 for p in predictions]
		return(predictions)


