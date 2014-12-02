#Kaggle click competition

import sys,os,time,pickle,subprocess
import numpy as np
import pandas as pd
import click_utilities as util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import BernoulliRBM
#from nolearn.dbn import DBN

from sklearn.pipeline import make_pipeline

start = time.time() # record time
log_file=time.strftime('log/%b-%d_ %H:%M:%S.txt',time.localtime()) #log into file
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
tee = subprocess.Popen(["tee", log_file], stdin=subprocess.PIPE)
os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

#calculate the Entropy and mutual information to select the variables
#print util.observe_correlations(10000000)

clf_squence=["GBT"]
clf = {
    "GBT":         GradientBoostingClassifier(n_estimators=200,max_leaf_nodes=50),
#    "DBN":       DBN([10, 10, 10, 10 , 10],learn_rates=0.3,learn_rate_decays=0.9,epochs=10,verbose=1),
#    "NLR":  make_pipeline(PolynomialFeatures(2),SGDClassifier(loss="log",alpha=5e-5,penalty="l1",n_jobs=4,warm_start=True)),
#     "MNB":         MultinomialNB()
#    "LR_SGD":      SGDClassifier(loss="log",alpha=5e-5,penalty="l1",n_jobs=4,warm_start=True)
#    "LR":          LogisticRegression(C=1) can be replaced by LR_SGD, and slower than later.
#    "LDA":        LDA(),  cannot adjust model at all
#    "QDA":        QDA(),  too high loss
#    "NB":         GaussianNB(), too high loss
#    "AdaBoost":   AdaBoostClassifier(n_estimators=20), higher loss and lower efficiency
#    "RF":        RandomForestClassifier(n_estimators=30, max_features=1), not good and slow
}#construct classifiers
mixture=Ridge(alpha=1e-2)
#mixture=util.log_combiner(alpha=1e-3)
#mixture=LogisticRegression()

flag=2 # 0 for 100M, 1 for 1000M, 2 for all data
flag2=2 # 0 for PCA, 1 for PLS, 2 for selected variables, -1 for original variables

clf_file="data/clf.pickle"
mix_file="data/mix.pickle"
if os.path.exists(clf_file) and os.path.exists(mix_file) and False: # read in the trained models
    clf=pickle.load(open(clf_file,"rb"))
    scale=["100M","1000M","all"]
    mixture=pickle.load(open(mix_file,"rb"))
    map_file="data/map_"+scale[flag]+".pickle"
    data_map=pickle.load(open(map_file,"rb"))
    print "trained models are loaded already."
else:  # train models from training data
    X_train, X_validate, y_train, y_validate, data_map = util.read_training(flag=flag,flag2=flag2)# read training data
#    prior_probs=np.repeat(np.mean(y_train),len(y_validate))#null model
#    print "Logloss of null model: %f, overall probability is %f" %(util.llfun(y_validate,prior_probs),prior_probs[:,np.newaxis].mean(axis=0))

    output=pd.DataFrame();input_prob=pd.DataFrame();
    for name in clf_squence:#train submodels
        print "begin to train %s model" %(name)
        fit_start=time.time()
        clf[name].fit(X_train, y_train)
        output[name]=clf[name].predict_proba(X_train)[:,1]
        input_prob[name]=clf[name].predict_proba(X_validate)[:,1]
        print "Time used = %s, Logloss: %f, overall probability: %f" %(time.strftime('%H:%M:%S',time.gmtime(time.time()-fit_start)),util.llfun(y_validate,input_prob[name]),input_prob[name].mean(axis=0))

    output=np.array(output);input_prob=np.array(input_prob);
    mixture.fit(output,y_train)#train mixture models
    merge=mixture.predict(input_prob)
    print "Logloss of Ensemble model: %f, overall probability is %f" %(util.llfun(y_validate,merge),merge.mean(axis=0))

    pickle.dump(clf,open(clf_file,"wb"))
    pickle.dump(mixture,open(mix_file,"wb"))

#read the test data in
test_file = "data/test.csv"
test = pd.read_csv(test_file,header=0)
print "Data in. Test data has %d records." %(test.shape[0])
test = util.preprocess_data(test,training=False,data_map=data_map,bias=0.06)
print "Test data has been preprocessed."

if flag2==0:
    X_test=util.reduce_PCA(test)
elif flag2==1:
    X_test=util.reduce_PLS(test)
else:
    X_test=util.reduce_manually(test)

#calculate the prediction of test data
test_input=pd.DataFrame()
for name in clf_squence:
    test_input[name]=clf[name].predict_proba(np.array(X_test))[:,1]
test_input=np.array(test_input)
test_merge= mixture.predict(test_input)

#check the overall click probability for each prediction
pmean=np.hstack((test_input,test_merge[:,np.newaxis])).mean(axis=0)
print pd.Series(pmean,index=clf_squence+["test_combined"])

#generate the submission file
pd.DataFrame(zip(test["id"],test_merge),columns=["id","click"]).to_csv("data/submission_combined.csv",header=True,comments='',index=False)
for i,name in enumerate(clf_squence):
    outputfile="data/submission_"+name+".csv"
    pd.DataFrame(zip(test["id"],test_input[:,i]),columns=["id","click"]).to_csv(outputfile,header=True,comments='',index=False)

print 'complete. Total time used = %s' %(time.strftime('%H:%M:%S',time.gmtime(time.time()-start)))











