#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np

def trainRandomForest(X,y,config):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


    print("训练集样本规格：", X_train.shape)
    print("训练集标签规格：", X_test.shape)
    print("测试集样本规格：", y_train.shape)
    print("测试集样本规格：", y_test.shape)
    
    
    rf=RandomForestClassifier(n_estimators=100,max_depth=None,max_features='auto',random_state=0)
    rf.fit(X_train,y_train)
    
    
    n_num = 100
    #acc_list=[]

    rf=RandomForestClassifier(n_estimators=n_num,max_depth=None,max_features='auto',random_state=0)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    acc=accuracy_score(y_pred,y_test)
    print("模型准确度为： ",acc)
    
    #寻找特征值最大的1500
    feat_importances=pd.Series(rf.feature_importances_,index=pd.DataFrame(X).columns)
    feature_1000=feat_importances.nlargest(config['feature_num'])
    #print(feature_1000)
    
    
    feature_1000_name=feature_1000.keys().tolist()
    #print(feature_1000_name)
    f_1000=[]
    for i in feature_1000_name:
        f_1000.append(i)
    #print("f_1000:",f_1000)
    
    
    return acc,f_1000


