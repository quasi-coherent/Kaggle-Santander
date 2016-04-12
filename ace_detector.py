import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from spectral import ace

""" A second take on this problem, """

def cross_validate():
    #for tinkering with the model
    #read data
    print('loading data')
    all_df = pd.read_csv('./data/train.csv',index_col = 'ID')

    #feature selection 
    #feature_selection = feature_importance(all_df) > 0.0001
    #feature_selection = list(feature_selection.ravel())
    #feature_selection.append(True)
    #all_df = all_df[all_df.columns[feature_selection]]
    
    #split data into training and cross validation set
    zeros_df = all_df[all_df.TARGET == 0]
    ones_df = all_df[all_df.TARGET == 1]
    
    num_ones = ones_df.shape[0]
    
    msk = np.random.rand(len(zeros_df)) < 0.1*num_ones/len(zeros_df)
    zeros_train_df = zeros_df[~msk]
    zeros_test_df = zeros_df[msk]
    
    msk = np.random.rand(num_ones) < 0.1
    ones_train_df = ones_df[~msk]
    ones_test_df = ones_df[msk]
    
    train_df = pd.concat([zeros_train_df,ones_train_df,zeros_test_df])
    targets_df = ones_test_df
    #test_df has approx 50% 1s and 50% 0s
    
    #oversample with smote
    train_X = np.array(train_df.drop('TARGET',axis = 1))
    train_Y = np.array(train_df.TARGET)

    #predict on smoted data
    print('predicting')
    targets = np.array(targets_df.drop('TARGET',axis = 1))
    
    output = ace(train_X,targets)
    print(output)
    
    
    #score
    print('scoring')
    conf_matrix = confusion_matrix(test_Y,predictions)
    print('confusion matrix:')
    print(pd.DataFrame(conf_matrix,columns = [0,1]))
    
    print('accuracy:')
    print(sum(test_Y.reshape(predictions.shape) == predictions)/len(test_Y))
    #return(test_Y,predictions)
    #return(test_df,train_df)
    
def run_for_kaggle():
    #this is for kaggle
    #produces predictions.csv
    #read data
    train_df = pd.read_csv('./data/train.csv',index_col = 'ID')
    test_df = pd.read_csv('./data/test.csv',index_col = 'ID')

    #format to numpy
    #train_X = np.array(train_df.drop('TARGET',axis = 1))
    #train_Y = np.array(train_df.TARGET)
    #test_X = np.array(test_df)
    
    #train svm
    my_hella_rfs = hella_rfs()
    my_hella_rfs.fit(train_df)
    
    #predict
    predictions = my_hella_rfs.predict(test_df).astype(int).ravel()
    
    #output
    out_df = pd.DataFrame(np.array([test_df.index,predictions]).T,columns = ['ID','TARGET'])
    out_df.to_csv('predictions.csv',index = False)

def feature_importance(all_df):
    train_X = np.array(all_df.drop('TARGET', axis = 1))
    train_Y = np.array(all_df.TARGET)
    rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, class_weight = 'balanced')
    rf.fit(train_X,train_Y)
    return(rf.feature_importances_)

if __name__ == '__main__':
    cross_validate()
    #run_for_kaggle()
    
    
