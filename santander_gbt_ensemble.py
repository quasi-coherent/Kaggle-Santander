#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import santander_preprocess as spp

from scipy.stats import mode
from sklearn.ensemble import GradientBoostingClassifier



class GBTEnsemble(object):
	'''
	Trains an ensemble of gradient boosted trees, where each 
	committee member sees the training set with a different ratio
	of over-sampling.
	''' 
	def __init__(self, k_best=250, 
		sampling_ratios=np.arange(1, 24, dtype='float'), 
		learning_rate=0.1, 
		n_estimators=100, 
		subsample=1.0, 
		max_depth=3):
		self.k_best = k_best
		self.sampling_ratios = sampling_ratios
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.subsample = subsample
		self.max_depth = max_depth
		self.ensemble = []

	def fit(self):
		sampler = spp.Santander(k_best=self.k_best)
		for ratio in self.sampling_ratios:
			print('Fitting GBT with ratio %f...' % ratio)
			X_train, y_train, _ = \
			sampler.preprocess(resample_method='SMOTE', ratio=ratio)
			GBT = GradientBoostingClassifier(learning_rate=self.learning_rate, 
					n_estimators=self.n_estimators, subsample=self.subsample,
					max_depth=self.max_depth, verbose=1)
			GBT.fit(X_train, y_train)
			self.ensemble.append(GBT)
		print('Done fitting...')
		return self.ensemble

	def predict(self):
		X_test = spp.Santander(k_best=self.k_best).X_test

		preds = []
		n = 1
		for gbt in self.ensemble:
			print('Making predictions with GBT %d' % n)
			preds.append(gbt.predict(X_test))
			n += 1

		preds = np.array(preds).T
		preds = np.array(mode(preds, axis = 1).mode)\
				.reshape([X_test.shape[0], 1])

		print('Done predicting...')
		return preds



if __name__ == '__main__':
	import pandas as pd

	gbte = GBTEnsemble(max_depth=5)
	gbte.fit()
	preds = gbte.predict()
	sub = pd.read_csv('data/sample_submission.csv')
	print('Making submission...')
	sub.TARGET = preds
	sub.to_csv('submission/GBTEnsemble-submission.csv', index=False)






