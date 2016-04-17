#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
import santander_preprocess as spp
import santander_cv as scv

# VotingClassifer needs sklearn v.0.17.1
from sklearn.ensemble import (RandomForestClassifier,
								GradientBoostingClassifier,
								VotingClassifier)


S = spp.Santander(k_best=300)
X_train, y_train, X_test = S.preprocess(resample_method='SMOTE_SVM', ratio=23.0)


print('Doing random grid search with RF...')
rf_param = {'n_estimators': np.arange(50, 201, 10),
			'criterion': ['gini', 'entropy'], 
			'max_features': ['auto', 'log2', None],
			'max_depth':[5, None],
			'boostrap': [True, False]}
rfc = RandomForestClassifier(verbose=1)
rf_best = scv.SantanderCV(model=rfc, X=X_train, y=y_train)\
		 .random_search(param_distributions=rf_param)


print('Doing random grid search with GBT...')
gbt_param = {'n_estimators': np.arange(100, 1001, 50),
			 'loss': ['deviance', 'exponential'],
			 'learning_rate': np.linspace(0.1, 1.0, 10),
			 'max_depth': np.arange(1, 8), 
			 'max_features': ['auto', 'log2', None]}
gbt = GradientBoostingClassifier(verbose=1)
gbt_best = scv.SantanderCV(model=gbt, X=X_train, y=y_train)\
		  .random_search(param_distributions=gbt_param)


print('Making predictions...')
preds = VotingClassifer([rf_best, gbt_best], voting='soft')\
		  .predict(X_test)


print('Preparing submission...')
sub = pd.read_csv('data/sample_submission.csv')
sub.TARGET = preds
sub.to_csv('submission/eclf-submission.csv', index=False)
print('Done!')
