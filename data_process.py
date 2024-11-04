#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load data_process.py
#!/usr/bin/env python

# In[1]:


from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#df1=pd.read_csv("data/guan/AAV.csv",sep=',',header=None)
#df.tail()
#df1.tail()


# In[2]:
def convert_to_df(df1,featureNames,file_name):
    X_1000=pd.DataFrame(df1,columns=featureNames)
    #print('X_1000:',X_1000)
    #b=X_1000.isnull().any()
    #print(b.all())
    X_1000.to_csv(file_name)
    return X_1000
    

def getDF(f1,config):
    df1=pd.read_csv(f1,sep=',',header=None)
    df1.tail()
    df1=df1.drop(config['del_col'],axis=1)
    #df1.rename(columns={10152:10151},inplace=True)
    #df1.tail()
    return df1

def preData(f1,f2):

    df1=pd.read_csv(f1,sep=',')
    df2=pd.read_csv(f2,sep=',')


    rows,cols=df1.shape
    print(df1.shape)
    df1=df1.drop('name',axis=1)
    #print(df1)
    df2=df2.drop('name',axis=1)

    df2=df2.iloc[1:,1:]


    df=pd.concat([df1,df2],axis=0)
    print(df.shape)

    X=df.iloc[1:,1:-1]
    print(X.shape)
    y=df.iloc[1:,-1]

    X=np.array(X,dtype=float)
    y=np.array(y)
    
    return df1,df, X,y


    # In[ ]:


def  getFeaturesData(df,featureNames):
    X_1000=pd.DataFrame(df,columns=featureNames)
    print('X_1000:',X_1000)
    #b=X_1000.isnull().any()
    #print(b.all())
    #X_1000.to_csv("X_1000.csv")
    df_np=np.array(df)
    print(df_np)


    # In[17]:


    df_np=df_np[1:,1:].astype(np.float32)
    print(df_np)


    # In[18]:

    
   # X=df_np[:,:-1]
    #print(X)


    # In[23]:


    #y=df_np[:,-1].astype(np.int)
    #print(y)
    
    return X

    


# In[ ]:




