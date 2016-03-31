import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

""" A second take on this problem, """

class hella_svms():
    def __init__(self):
        self.svms = []
    def fit(self,train_df):
        #split df into zeros and ones dfs
        zeros = np.array(train_df[train_df.TARGET == 0].drop('TARGET',axis = 1))
        ones = np.array(train_df[train_df.TARGET == 1].drop('TARGET',axis = 1))
        # compute number of chunks to split
        # this should be done differently i think
        num_chunks = int(zeros.shape[0]/ones.shape[0])
        chunks = np.array_split(zeros,num_chunks)
        i = 0
        for chunk in chunks:
            print('training svm %s of %s' %(i,num_chunks))
            chunk_svm = svm.LinearSVC()
            chunk_train_X = np.concatenate([chunk,ones])
            chunk_train_Y = np.concatenate([np.zeros([chunk.shape[0],1]),np.ones([ones.shape[0],1])]).ravel()
            chunk_svm.fit(chunk_train_X,chunk_train_Y)
            self.svms.append(chunk_svm)
            i+=1
        #remove the last one, since it wont be as good as the rest
        #probably should just not train it in the first place
        self.svms = self.svms[:-1]
    def predict(self,test_df):
        test_X = np.array(test_df)
        predictions = []
        for svm_i in self.svms:
            predictions.append(svm_i.predict(test_X))
        predictions = np.array(predictions).T
        predictions = np.array(mode(predictions,axis = 1).mode).reshape([test_X.shape[0],1])
        return(predictions)

def cross_validate():
    #for tinkering with the model
    #read data
    all_df = pd.read_csv('train.csv',index_col = 'ID')

    #split data
    #need to be more intelligent about this
    num_obs = all_df.shape[0]
    rp = np.random.permutation(num_obs)
    train_df = all_df[rp>0.5*num_obs]
    test_df = all_df[rp<=0.5*num_obs]
    
    #train
    my_hella_svms = hella_svms()
    my_hella_svms.fit(train_df)
    
    
    #predict
    predictions = my_hella_svms.predict(test_df.drop('TARGET',axis = 1))
    test_Y = np.array(test_df.TARGET) #true target values
    conf_matrix = confusion_matrix(test_Y,predictions)
    print('confusion matrix:')
    print(pd.DataFrame(conf_matrix,columns = [0,1]))
    #print('accuracy:')
    #print(my_svm.score(test_X,test_Y))

def run_for_kaggle():
    #this is for kaggle
    #produces predictions.csv
    #read data
    train_df = pd.read_csv('train.csv',index_col = 'ID')
    test_df = pd.read_csv('test.csv',index_col = 'ID')

    #format to numpy
    #train_X = np.array(train_df.drop('TARGET',axis = 1))
    #train_Y = np.array(train_df.TARGET)
    #test_X = np.array(test_df)
    
    #train svm
    my_hella_svms = hella_svms()
    my_hella_svms.fit(train_df)
    
    #predict
    predictions = my_hella_svms.predict(test_df).astype(int).ravel()
    
    #output
    out_df = pd.DataFrame(np.array([test_df.index,predictions]).T,columns = ['ID','TARGET'])
    out_df.to_csv('predictions.csv',index = False)
    
if __name__ == '__main__':
    #cross_validate()
    run_for_kaggle()
    
    
