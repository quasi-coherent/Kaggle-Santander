from __future__ import print_function
import numpy as np
import santander_preprocess as spp
import santander_cv as scv
import santander_predict_submit as sps

from sklearn.svm import SVC



print('Performing preprocessing...')
X_train, y_train = spp.Santander()\
			.preprocess(pca_components=0.6, resample_method='SMOTE')


svc = SVC()


# Warning: This part would take quite a while, i.e., forever.  
# print('Doing grid search...')
# grid = [{'kernel':['rbf', 'linear'],
# 				'gamma': np.logspace(-9, 3, 13),
# 				'C':np.logspace(-2, 10, 13)}]
# train_cv = scv.SantanderCV(svc, X_train, y_train, resample=True)
# svc_best = train_cv.grid_search(param_grid=grid)
# print('Best model obtained is %s' % svc_best)


print('Performing 10-fold CV...')
scv.SantanderCV(svc, X_train, y_train, resample=True)\
			.cross_validation()


print('Now making submission...')
X_test = spp.Santander(train=False).preprocess(pca_components=0.6)
sps.kaggle_submit(svc, X_train, y_train, X_test, 'test')