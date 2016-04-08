from __future__ import print_function
import numpy as np
import santander_preprocess as spp
import santander_cv as scv
import santander_predict_submit as sps

from sklearn.ensemble import AdaBoostClassifier



print('Performing preprocessing...')
X_train, y_train = spp.Santander()\
			.preprocess(pca_components=0.6)


abc_best = AdaBoostClassifier(learning_rate=0.5, n_estimators=500)

 
# print('Doing grid search...')
# grid = [{'n_estimators': np.arange(50, 500, 10), 
# 			'learning_rate': np.linspace(.1, 2, 20)}]
# train_cv = scv.SantanderCV(abc, X_train, y_train, resample=True)
# abc_best = train_cv.grid_search(param_grid=grid)
# print('Best model obtained is %s' % abc_best)


print('Performing 10-fold CV...')
scv.SantanderCV(abc_best, X_train, y_train)\
			.cross_validation()


print('Now making submission...')
X_test = spp.Santander(train=False).preprocess(pca_components=0.6)
sps.kaggle_submit(abc_best, X_train, y_train, X_test, 'test')