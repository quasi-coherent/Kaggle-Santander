from __future__ import print_function

import numpy as np
import pandas as pd

import unbalanced_dataset
from sklearn.decomposition import PCA


class Santander(object):
	def __init__(self, train=True):
		self.train = train
		if self.train:
			df = pd.read_csv('data/train.csv')
			df = df.loc[:, df.std() > 0]  # Some rows have zero variance

			# Treating -999999 as missing; impute with knn
			df['var3'] = df['var3'].replace(-999999, 2)
			self.X = df.ix[:, :-1].values
			self.y = df.ix[:, -1].values
		else:
			df = pd.read_csv('data/test.csv')
			df = df.loc[:, df.std() > 0]
			df['var3'] = df['var3'].replace(-999999, 2)
			self.X = df.values


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
			Only preprocessed X if train=False; else y and 
			preprocessed X.
		'''

		pca = PCA(n_components=pca_components)

		if not self.train:
			return pca.fit_transform(self.X)
		
		if resample_method == 'UnderSampler':
			US = unbalanced_dataset.under_sampling\
						.UnderSampler(verbose=True, ratio=ratio)
			usX, usy = US.fit_transform(self.X, self.y)
			return pca.fit_transform(usX), usy

		elif resample_method == 'OverSampler':
			OS = unbalanced_dataset.over_sampling\
						.OverSampler(verbose=True, ratio=ratio)
			osX, osy = OS.fit_transform(self.X, self.y)
			return pca.fit_transform(osX), osy

		elif resample_method == 'SMOTE':
			SM = unbalanced_dataset.over_sampling\
						.SMOTE(kind='regular', verbose=True, ratio=ratio)
			smX, smy = SM.fit_transform(self.X, self.y)
			return pca.fit_transform(smX), smy

		elif resample_method == 'BalancedCascade':
			BC = unbalanced_dataset.ensemble_sampling\
						.BalancedCascade(verbose=True, ratio=ratio)
			bcX, bcy = BC.fit_transform(self.X, self.y)
			return pca.fit_transform(bcX), bcy

		else:
			return pca.fit_transform(self.X), self.y

