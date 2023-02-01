#!/usr/bin/env python
# coding: utf-8

# In[189]:


import math
import string
import pandas as pd
import numpy as np
import sklearn as sl


# In[190]:


nucleotids = pd.read_csv("hw01_data_points.csv", header=None)
labels = pd.read_csv("hw01_class_labels.csv", header=None).astype(int)


# In[191]:


trainData = nucleotids[:300]
testData = nucleotids[300:]
trainLabel = labels[:300]
testLabel = labels[300:]
testLabel = labels[300:]


testLabel = np.array(testLabel)
trainLabel = np.array(trainLabel)
trainData = np.array(trainData)
testData = np.array(testData)
labels = np.array(labels)


# In[192]:


K = np.max(labels)
print(K)


# In[193]:


def meanfunc (x):
     return [np.mean(np.logical_and(trainData == x , trainLabel==np.array([c+1])), axis=0)/(1/2) for c in range(K)]

pAcd= meanfunc('A')
pTcd = meanfunc('T')
pCcd = meanfunc('C')
pGcd = meanfunc('G')

class_priors = [np.mean(labels==np.array([c+1])) for c in range(K)]
#arraying
pAcd = np.array(pAcd)
pTcd = np.array(pTcd)
pGcd = np.array(pGcd)
pCcd = np.array(pCcd)
CP = np.array(class_priors)


# In[194]:


a = np.log(pAcd)
g = np.log(pGcd)
t = np.log(pTcd)
c = np.log(pCcd)


datA = ((trainData == 'A').astype(int))
datG = ((trainData == 'G').astype(int))
datC = ((trainData == 'C').astype(int))
datT = ((trainData == 'T').astype(int))

def shape1 (x,y):
    return np.matmul(x, np.transpose(y))

datA = shape1(datA, a)
datG = shape1(datG, g)
datT = shape1(datT, t)
datC = shape1(datC, c)


# In[195]:


trainingScore = np.sum((datA, datG, datT, datC, CP), axis = 0)  
trainingScore = np.array(trainingScore)

predicted1 = np.argmax(trainingScore, axis = 1) + 1
trainLabel = np.reshape(trainLabel, (300,))

confusionTrain = pd.crosstab(predicted1, trainLabel, rownames = ["y-pred"], colnames = ["y-truth"])


# In[196]:


print(confusionTrain)


# In[197]:


testA = ((testData == 'A').astype(int))
testG = ((testData == 'G').astype(int))
testT = ((testData == 'T').astype(int))
testC = ((testData == 'C').astype(int))

testA = shape1(testA,a)
testG = shape1(testG,g)
testT = shape1(testT,t)
testC = shape1(testC,c)

testScore = np.sum((testA, testG, testT, testC, CP), axis = 0) + 1
testScore = np.array(testScore)
predicted2 = (np.argmax(testScore, axis = 1) + 1)
testLabel = np.reshape(testLabel, (100,))
confusionTest = pd.crosstab(predicted2, testLabel, rownames = ["y-pred"], colnames = ["y-truth"])


# In[198]:


print(confusionTest)


# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




