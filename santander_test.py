from __future__ import print_function
import numpy as np
import santander_preprocess as spp
import santander_cv as scv
import santander_predict_submit as sps

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier



print('Performing preprocessing...')
X_train, y_train, X_test = spp.Santander(k_best=100)\
			.preprocess(resample_method=None, ratio=23.0)


#bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
						#algorithm='SAMME',
						#n_estimators=1000,
						#learning_rate=0.2)

from hella_rfs import hella_rfs
hrfs = hella_rfs(use_weights = True)

# This would take a while, i.e., forever.  
# print('Doing grid search...')
# grid = [{'n_estimators': np.arange(50, 500, 20), 
# 			'learning_rate': np.linspace(.1, 2, 10)}]
# train_cv = scv.SantanderCV(bdt, X_train, y_train, resample=True)
# bdt_best = train_cv.grid_search(param_grid=grid)
# print('Best model obtained is %s' % abc_best)


# print('Performing 10-fold CV...')
# scv.SantanderCV(bdt, X_train, y_train)\
# 			.cross_validation()


print('Now making submission...')
sps.kaggle_submit(hrfs, X_train, y_train, X_test, 'hella_rfs')
