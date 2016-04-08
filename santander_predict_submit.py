from __future__ import print_function
import pandas as pd



def kaggle_submit(model, X_train, y_train, X_test, name):
	'''
	Prepares submission file for Kaggle.
	'''
	
	print('Fitting model %s' % model)
	model.fit(X_train, y_train)

	print('Making predictions...')
	pred = model.predict(X_test)

	print('Preparing submission file...')
	sub = pd.read_csv('data/sample_submission.csv')
	sub.TARGET = pred
	sub.to_csv('submission/%s-submission.csv' % name, index=False)
	print('Done!')