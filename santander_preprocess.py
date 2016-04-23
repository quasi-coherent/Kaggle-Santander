from __future__ import print_function

import numpy as np
import pandas as pd

import unbalanced_dataset
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA, KernelPCA


class Santander(object):
	def __init__(self, pca_components=None, whiten=True, k_best=False):
		train = pd.read_csv('data/train.csv')
		test = pd.read_csv('data/test.csv')
		# Some rows have zero variance
		# train = train.loc[:, train.std() > 0] 
		# test = test.loc[:, test.std() > 0]

		# # Treating -999999 as missing; impute with knn
		train['var3'] = train['var3'].replace(-999999, 2)
		test['var3'] = test['var3'].replace(-999999, 2)
		X_train = train.ix[:, :-1].values
		y_train = train.ix[:, -1].values
		X_test = test.values

		# Perform PCA
		pca = PCA(n_components=pca_components, whiten=whiten)
		X_train = pca.fit_transform(X_train, y_train)
		X_test = pca.fit_transform(X_test)

		if k_best:
			if k_best > pca_components:
				k_best='all'
			# Select k best features by F-score
			kb = SelectKBest(f_classif, k=k_best)
			X_train = kb.fit_transform(X_train, y_train)
			X_test = kb.transform(X_test)
			
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
	def dim_reduce(self, reduce_method = None, n_components = None):
		''' 
		Only dimensionality reduction. 
		
		:param reduce_method: 
			The method for dimensionality reduction. Can be ...
			
		:n_components:
			Dimension
		'''
		if reduce_method is None:
			return self.X_train
		elif reduce_method == 'pca':
			print('performing pca dimensionality reduction.')
			pca = PCA(n_components = n_components, whiten = False)
			self.X_train = pca.fit_transform(self.X_train)
		elif reduce_method =='kpca':
			print('performing kpca dimensionality reduction.')
			kpca = KernelPCA(n_components = n_components, kernel = 'rbf', eigen_solver = 'arpack')
			self.X_train = kpca.fit_transform(self.X_train)

	def preprocess(self, resample_method=None, ratio=1.0):
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
			'OverSampler', 'SMOTE_SVM', or 'BalancedCascade'.  

		:return:
			Preprocessed train and test sets. 
		'''
		
		if resample_method == 'UnderSampler':
			US = unbalanced_dataset.under_sampling\
						.UnderSampler(verbose=True, ratio=ratio)
			self.X_train, self.y_train = US.fit_transform(self.X_train, self.y_train)
			return self.X_train, self.y_train, self.X_test

		elif resample_method == 'OverSampler':
			OS = unbalanced_dataset.over_sampling\
						.OverSampler(verbose=True, ratio=ratio)
			self.X_train, self.y_train = OS.fit_transform(self.X_train, self.y_train)
			return self.X_train, self.y_train, self.X_test

		elif resample_method == 'SMOTE':
			SM = unbalanced_dataset.over_sampling\
						.SMOTE(kind='regular', verbose=True, ratio=ratio)
			self.X_train, self.y_train = SM.fit_transform(self.X_train, self.y_train)
			return self.X_train, self.y_train, self.X_test

		elif resample_method == 'SMOTE_SVM':
			svm_args = {'class_weight': 'auto'}
			SMSVM = unbalanced_dataset.over_sampling\
						.SMOTE(kind='svm', verbose=True, ratio=ratio, **svm_args)
			self.X_train, self.y_train = SMSVM.fit_transform(self.X_train, self.y_train)
			return self.X_train, self.y_train, self.X_test

		elif resample_method == 'BalancedCascade':
			BC = unbalanced_dataset.ensemble_sampling\
						.BalancedCascade(verbose=True, ratio=ratio)
			self.X_train, self.y_train = BC.fit_transform(self.X_train, self.y_train)
			self.X_train, self.y_train, self.X_test

		else:
			return self.X_train, self.y_train, self.X_test

