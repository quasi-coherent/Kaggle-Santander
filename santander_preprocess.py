from __future__ import print_function

import numpy as np
import pandas as pd

import unbalanced_dataset
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA


class Santander(object):
	def __init__(self, k_best=False):
		self.k_best = k_best
		train = pd.read_csv('data/train.csv')
		test = pd.read_csv('data/test.csv')
		# Some rows have zero variance
		train = train.loc[:, train.std() > 0] 
		test = test.loc[:, test.std() > 0]

		# Treating -999999 as missing; impute with knn
		train['var3'] = train['var3'].replace(-999999, 2)
		test['var3'] = test['var3'].replace(-999999, 2)
		X_train = train.ix[:, :-1].values
		y_train = train.ix[:, -1].values.ravel()
		X_test = test.values

		if self.k_best:
			# Select k best features by F-score
			kb = SelectKBest(f_classif, k=self.k_best)
			X_train = kb.fit_transform(X_train, y_train)
			X_test = kb.transform(X_test)
			
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test


	def preprocess(self, pca_components=None, resample_method=None, ratio=1.0):
		'''
		First, uses principal component analysis to reduce 
		dimensionality of the dataset.  

		Then, uses the Python package UnbalancedDataset, which 
		offers a number of re-sampling techniques to address
		the issue of class imbalance in the Santander dataset.  

		:param pca_components:
			Number of components to keep.  If None, all components
			are kept.  If n_components='mle', Minka's MLE is used to 
			guess the dimension.  If 0<n_components<1, select the number 
			of components such that the amount of variance that needs to 
			be explained is greater than the percentage specified by 
			n_components.

		:param resample_method: 
			Re-sampling method to use.  Can be 'UnderSampler',
			'OverSampler', 'SMOTE', or 'BalancedCascade'.  

		:return:
			Preprocessed train and test sets. 
		'''

		pca = PCA(n_components=pca_components)
		
		if resample_method == 'UnderSampler':
			US = unbalanced_dataset.under_sampling\
						.UnderSampler(verbose=True, ratio=ratio)
			usX, usy = US.fit_transform(self.X_train, self.y_train)
			return pca.fit_transform(usX), usy, pca.fit_transform(self.X_test)

		elif resample_method == 'OverSampler':
			OS = unbalanced_dataset.over_sampling\
						.OverSampler(verbose=True, ratio=ratio)
			osX, osy = OS.fit_transform(self.X_train, self.y_train)
			return pca.fit_transform(osX), osy, pca.fit_transform(self.X_test)

		elif resample_method == 'SMOTE':
			SM = unbalanced_dataset.over_sampling\
						.SMOTE(kind='regular', verbose=True, ratio=ratio)
			smX, smy = SM.fit_transform(self.X_train, self.y_train)
			return pca.fit_transform(smX), smy, pca.fit_transform(self.X_test)

		elif resample_method == 'BalancedCascade':
			BC = unbalanced_dataset.ensemble_sampling\
						.BalancedCascade(verbose=True, ratio=ratio)
			bcX, bcy = BC.fit_transform(self.X_train, self.y_train)
			return pca.fit_transform(bcX), bcy, pca.fit_transform(self.X_test)

		else:
			return pca.fit_transform(self.X_train), self.y_train, pca.fit_transform(self.X_test)

