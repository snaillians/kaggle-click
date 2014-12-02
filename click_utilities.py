import numpy as np
import pandas as pd
import scipy as sp
#from collections import Counter
from sklearn import metrics
from sklearn.cluster import KMeans
import sys,os,time,pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from scipy.optimize import minimize

#Use this function to preprocess training data and testing data.
def preprocess_data(dataframe,training=True,data_map=None,bias=0.05):
    #pd.set_option('chained_assignment',None)
    #change the position of each column
    if training==True:
        dataframe.loc[:]=dataframe.iloc[:,np.r_[1,0,2:24]]
    else:
        dataframe.insert(0,"click",0)

    #subtract the hour information from the column "hour", map values of some columns into 0-1 float number, change id to str
    dataframe.loc[:,"hour"]=dataframe["hour"].astype(str).map(lambda x: x[-2:]).astype(float)
    dataframe.loc[:,"id"]=dataframe["id"].astype(str)
    dataframe.loc[:,["device_id","device_ip"]]=dataframe.loc[:,["device_id","device_ip"]].apply(lambda t: t.map(lambda x: (abs(hash(x))%10**6)/float(10**6)))

    '''#how much of the data is missing?
    print "Proportion of missing data: %f."%( float( pd.isnull( dataframe ).sum().sum() )/(dataframe.shape[0]*dataframe.shape[1]) )
    #do a naive filling of missing data for col_name in dataframe.columns:
    dataframe[col_name].fillna(value=dataframe[col_name].mean(), inplace=True )'''

    # map values of some columns based on its conditional probability
    mapcolumn=[x for x in dataframe.columns if x not in ["device_id","device_ip"]]
    if training==True:
        data_map=generate_map(dataframe[mapcolumn])
        dataframe=remap_data(dataframe,data_map)
        return dataframe,data_map
    else:
        dataframe=remap_data(dataframe,data_map,bias)
        return dataframe

#This is the evaluation function using logloss as criterion
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

#The objective function for optimization of coefficients
def llfun_combine(w,act,X,alpha):
    return llfun(act,np.dot(X,w))+alpha*np.dot(w,w)

#The first order derivative function of objective function
def llfun_der(w,act,pred):
    der=np.zeros(4)
    for i in range(len(act)):
        x=pred[i,:]
        y=act[i]
        der=der+(y-np.dot(w,x))/np.dot(w,x)/(1-np.dot(w,x))*x
    llder=der*-1.0/len(act)+np.ones(4)
    return llder

#Entropy: measure the disorder that a variable holds
def entropy(s):
    #p, lns = Counter(s), float(len(s))
    #return -sum( count/lns * np.log(count/lns) for count in p.values())
    return None

#generate the mapping dictionaries for different variables
def generate_map(dataframe):
    map={}
    prior=dataframe["click"].mean()
    for name in dataframe.columns:
        if name in ["click","id"]:
            pass
        else:
            freq=dataframe.groupby(name).apply(lambda subset: sum(subset["click"]==1)/float(len(subset)))
            #freq.sort()
            #priorfreq=dataframe.groupby(name).apply(lambda subset: len(subset)/float(len(dataframe)))
            #freq[:]=range(len(freq))
            #freq[:]=freq[:]*len(freq)
            freq["other"]=prior
            map[name]=freq
    return map

#remapping the data into integers according to its conditional probability
def remap_data(dataframe,map,bias):
    for name in map.keys():
        submap=map[name]
        maplevel=submap.keys()
        levels,indices=np.unique(dataframe[name],return_inverse=True)
        newlevel=np.repeat(0.0,len(levels)+1)
        for idx,item in enumerate(levels):
            if item in maplevel:
                newlevel[idx]=submap[item]
            else:
                newlevel[idx]=submap["other"]-bias
        dataframe[name]=newlevel[indices]
    return dataframe

#observe the correlations between two discrete distributed random variables
def observe_correlations(nrows=1000000):
    training_file = "data/train.csv"
    dataframe = pd.read_csv(training_file,nrows=nrows,header=0)
    info_variable=pd.DataFrame(index=dataframe.columns[2:])
    info_variable["levels"]=dataframe.iloc[:,2:].apply(lambda t: np.unique(t).shape[0])
    info_variable["Entropy"]=dataframe.iloc[:,2:].apply(lambda t: entropy(t))
    info_variable["MutualInformation"]=dataframe.iloc[:,2:].apply(lambda t: metrics.normalized_mutual_info_score(dataframe["click"],t))
    info_variable["clustering"]=cluster_var(dataframe)
    info_variable=info_variable.sort("clustering")
    return info_variable

#cluster the original variables based on the normalized mutual information
def cluster_var(dataframe):
    pd.set_option('expand_frame_repr', False)
    names=[x for x in dataframe.columns if x not in ["click","id","device_id","device_ip"]]
    m=len(names)
    MI=pd.DataFrame(np.zeros(shape=(m,m)),index=names,columns=names)
    for i in range(m):
        for j in range(i+1,m):
            MI.iloc[i,j]=metrics.normalized_mutual_info_score(dataframe.loc[:,names[i]],dataframe.loc[:,names[j]])
    for i in range(m):
        for j in range(i+1):
            if j==i:
                MI.iloc[i,j]=1
            else:
                MI.iloc[i,j]=MI.iloc[j,i]
    clustering=KMeans(n_clusters=10).fit_predict(MI)
    clustering=pd.Series(clustering,index=names)
    return clustering

def reduce_PCA(dataframe):
    PCA_file="data/pca_structure.pickle"
    group=[["site_id","site_domain","site_category"],
           ["device_model"],
           ["C14","C17","C18","C19","C21"],
           ["C15","C16"],
           ["app_id","app_domain","app_category"]]
    PCA_models=[]
    T=np.empty(shape=[len(dataframe),len(group)])
    if os.path.exists(PCA_file):
        PCA_models=pickle.load(open(PCA_file,"rb"))
        print "PCA loading matrix is loaded."
        for index,cluster in enumerate(group):
            subX=np.array(dataframe[cluster])
            subPCA=PCA_models[index]
            T[:,index]=subPCA.transform(subX)[:,0]
    else:
        for index,cluster in enumerate(group):
            subX=np.array(dataframe[cluster])
            subPCA=PCA(n_components=1).fit(subX)
            PCA_models.append(subPCA)
            T[:,index]=subPCA.transform(subX)[:,0]
        pickle.dump(PCA_models,open(PCA_file,"wb"))
        print "PCA loading matrix is stored."
    print "PCA transformation is performed."
    return T

def reduce_PLS(dataframe):
    PLS_file="data/pls_structure.pickle"
    selectedcolumn=[x for x in dataframe.columns if x not in ["id","click","device_id","device_ip"]]
    X=np.array(dataframe[selectedcolumn])
    y=np.array(dataframe["click"])
    if os.path.exists(PLS_file):
        stand_PLS=pickle.load(open(PLS_file,'rb'))
        print "PLS structure is loaded."
    else:
        stand_PLS=PLSRegression(n_components=10,scale=True)
        stand_PLS.fit(X, y[:,np.newaxis])
        stand_PLS.y_scores_=None
        stand_PLS.x_scores_=None
        pickle.dump(stand_PLS,open(PLS_file,"wb"))
        print "PLS transform structure is stored."
    T=stand_PLS.transform(X)
    print "PLS transformation is performed."
    return T

def reduce_manually(dataframe):
    #selectedcolumn=[x for x in dataframe.columns if x not in ["id","click"]]
    #selectedcolumn=["site_id","site_domain","site_category","device_model","app_id","app_domain","app_category","C14","C15","C16","C17","C18","C19","C20","C21","device_id","device_ip"]
    selectedcolumn=["site_id","site_domain","device_model","device_id","device_ip","app_id","C14","C17","C19","C21"]
    X=dataframe[selectedcolumn]
    print "Important variables are selected manually."
    return np.array(X)

def read_training(flag=0,flag2=0):
    scale=["100M","1000M","all"]
    training_file = "data/train.csv"
    intermediate_file="data/intermediate_"+scale[flag]+".csv"
    map_file="data/map_"+scale[flag]+".pickle"

    if os.path.exists(intermediate_file) and os.path.exists(map_file):
        train = pd.read_csv(intermediate_file,header=0)
        data_map=pickle.load(open(map_file,"rb"))
        print "Data in. Preprocessed data has %d records." %(train.shape[0])
    else:
        if flag==0:
            train = pd.read_csv(training_file,nrows=1000000,header=0)
        if flag==1:
            train = pd.read_csv(training_file,nrows=10000000,header=0)
        print "Data in. Training data has %d records." %(train.shape[0])
        #Transform the data into the convenient form
        train,data_map=preprocess_data(train)
        print "Data has been preprocessed."
        train.to_csv(intermediate_file,header=True,comments='',index=False,float_format="%5.4f")
        pickle.dump(data_map,open(map_file,"wb"))
        print "Preprocessed Data is recorded as a file."

    #reduce the dimension of the data
    if flag2==0:
        X=reduce_PCA(train)
    elif flag2==1:
        X=reduce_PLS(train)
    elif flag2==2:
        X=reduce_manually(train)
    else:
        X=np.array(train.iloc[:,2:])
    y=train["click"]
    seq=np.random.permutation(np.arange(y.shape[0]))
    index_train=seq[1:int(0.5*y.shape[0])]
    index_test=seq[int(0.5*y.shape[0]):]
    print index_train.shape,index_test.shape
    X_train=X[index_train,:]
    X_validate=X[index_test,:]
    y_train=y[index_train]
    y_validate=y[index_test]
    #X_train, X_validate, y_train, y_validate = train_test_split(X,y, test_size=0.5) #split data into training and testing data
    return X_train,X_validate,y_train,y_validate,data_map

class log_combiner(object):
    def __init__(self,alpha=1e-4,maxiter=20):
        self.alpha=alpha
        self.maxiter=maxiter

    def fit(self,X,y):
        w0=np.ones(shape=[1,X.shape[1]])*1/float(X.shape[1])
        result = minimize(llfun_combine,w0,args=(y,X,self.alpha),method="Nelder-Mead",options={"maxiter":self.maxiter})
        self.w=result.x

    def predict_proba(self,X):
        return np.dot(X,self.w)

    def predict(self,X):
        return np.dot(X,self.w)