import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

""" A second take on this problem, """

class hella_rfs():
    def __init__(self):
        self.rfs = []
    def fit(self,train_df):
        #split df into zeros and ones dfs
        zeros = np.array(train_df[train_df.TARGET == 0].drop('TARGET',axis = 1))
        ones = np.array(train_df[train_df.TARGET == 1].drop('TARGET',axis = 1))
        # compute number of chunks to split
        # this should be done differently i think
        num_chunks = int(2*zeros.shape[0]/ones.shape[0])
        chunks = np.array_split(zeros,num_chunks)
        i = 0
        for chunk in chunks:
            print('training svm %s of %s' %(i,num_chunks))
            chunk_rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
            chunk_ones = ones[np.random.rand(ones.shape[0])>0.5,:]
            chunk_train_X = np.concatenate([chunk,chunk_ones])
            chunk_train_Y = np.concatenate([np.zeros([chunk.shape[0],1]),np.ones([chunk_ones.shape[0],1])]).ravel()
            chunk_rf.fit(chunk_train_X,chunk_train_Y)
            self.rfs.append(chunk_rf)
            i+=1
    def predict(self,test_df):
        test_X = np.array(test_df)
        predictions = []
        for rf_i in self.rfs:
            predictions.append(rf_i.predict(test_X))
        predictions = np.array(predictions).T
        predictions = np.array(mode(predictions,axis = 1).mode).reshape([test_X.shape[0],1])
        return(predictions)

def cross_validate():
    #for tinkering with the model
    #read data
    all_df = pd.read_csv('./data/train.csv',index_col = 'ID')

    #feature selection 
    feature_selection = feature_importance(all_df) > 0.0001
    feature_selection = list(feature_selection.ravel())
    feature_selection.append(True)
    all_df = all_df[all_df.columns[feature_selection]]
    
    #split data
    #need to be more intelligent about this
    zeros_df = all_df[all_df.TARGET == 0]
    ones_df = all_df[all_df.TARGET == 1]
    
    num_ones = ones_df.shape[0]
    msk = np.random.rand(len(zeros_df)) < 0.1*num_ones/len(zeros_df)
    
    zeros_train_df = zeros_df[~msk]
    zeros_test_df = zeros_df[msk]
    msk = np.random.rand(num_ones) < 0.1
    ones_train_df = ones_df[~msk]
    ones_test_df = ones_df[msk]
    
    train_df = pd.concat([zeros_train_df,ones_train_df])
    test_df = pd.concat([zeros_test_df,ones_test_df])
    #return(test_df,train_df)
    
    
    
    #train
    my_hella_rfs = hella_rfs()
    my_hella_rfs.fit(train_df)
    
    
    #predict
    predictions = my_hella_rfs.predict(test_df.drop('TARGET',axis = 1))
    test_Y = np.array(test_df.TARGET) #true target values
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
    
    
