from __future__ import print_function, division
import numpy as np
import pandas as pd
import santander_preprocess

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, StratifiedKFold



class SantanderCV(object):
	def __init__(self, model, X, y, resample=False): 
		self.model = model
		self.X = X
		self.y = y
		self.resample = resample


	def grid_search(self, param_grid, cv=None):
		'''
		Grid search to find optimal hyperparameters.
		'''

		gscv = GridSearchCV(self.model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
		gscv.fit(self.X, self.y)
		print('Best estimator was %s' % gscv.best_estimator_)
		return gscv.best_estimator_


	def cross_validation(self, K=10):
		'''
		Performs K-fold cross validation on a given X and y with a 
		given model.

		If resampling was not used, then stratified K-fold CV is 
		performed.  (This retains class ratio in the folds.)

		:param K:
			The number of folds.  

		:return:
			Mean accuracy after K-fold CV.  
		'''

		if not self.resample:
			skf = StratifiedKFold(y=self.y, n_folds=K)
			acc = []
			print('Performing %s-fold stratified CV with model %s' % (K, self.model))
			for train_index, test_index in skf:
				X_train, X_test = self.X[train_index], self.X[test_index]
				y_train, y_test = self.y[train_index], self.y[test_index]
				self.model.fit(X_train, y_train)
				acc.append(self.model.score(X_test, y_test))
			mean_acc = sum(acc) / len(acc)
			print('Mean accuracy is %f' % mean_acc)


		else:
			kf = KFold(n=len(self.X), n_folds=K)
			acc = []
			print('Performing %s-fold CV with model %s' % (K, self.model))
			for train_index, test_index in kf:
				X_train, X_test = self.X[train_index], self.X[test_index]
				y_train, y_test = self.y[train_index], self.y[test_index]
				self.model.fit(X_train, y_train)
				acc.append(self.model.score(X_test, y_test))
			mean_acc = sum(acc) / len(acc)
			print('Mean accuracy is %f' % mean_acc)

