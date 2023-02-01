#!/usr/bin/env python
# coding: utf-8

# In[44]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# In[56]:


data_set= np.genfromtxt("hw02_data_points.csv", delimiter = ",")
class_labels = np.genfromtxt("hw02_class_labels.csv", dtype=int , delimiter = ",")

w0 = np.genfromtxt('hw02_w0_initial.csv', delimiter=",")
W = np.genfromtxt('hw02_W_initial.csv', delimiter=",")


training_data = data_set[:10000]
test_data = data_set[10000:]

training_label = class_labels[:10000]
test_label = class_labels[10000:]



K = np.max(training_label)
N = data_set.shape[0]
D = data_set.shape[1] 


#X = data_set[0:10000, 0:(D)]
#Y_truth = data_set[:10000, D:(D)].astype(int)


Y_test = np.zeros((test_label.shape[0], K)).astype(int)
Y_test[range(test_label.shape[0]), test_label - 1] = 1
Y_truth = np.stack([1*(training_label == (i+1)) for i in range(K)], axis = 1)


# In[57]:


print(D)


# In[58]:


#GRADIENT AND SIGMOID FUNCTIONS 
def sigmoid(X, W, w0):
    return (1 / (1 + np.exp(-np.matmul(X, W) + w0)))

def gradient_W(X, Y_truth, Y_predicted):
    return(np.asarray([-np.sum(np.repeat(((Y_truth[:,c] - Y_predicted[:,c])* Y_predicted[:,c]*(1 - Y_predicted[:,c]))[:, None]
                                         ,X.shape[1], axis = 1) * X, axis = 0) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum((Y_truth - Y_predicted)*Y_predicted*(1 - Y_predicted), axis = 0))


# In[59]:


# LEARNING PARAMETERS
eta = 0.00001 #step sizes


# In[76]:


iteration_count = 1000

objective_values = []
i = 1
while True:
       
    Y_predicted = sigmoid(training_data, W, w0)
    objective_values = np.append(objective_values, (0.5)*np.sum(np.sum((Y_truth - Y_predicted)**2, axis=1), axis=0))
    W = W - eta * gradient_W(training_data, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)
    
    if (i == iteration_count + 1):
        break
    
    i = i + 1
    print(i)


# In[77]:


print(W)
print(w0)


# In[66]:


plt.figure(figsize=(6, 6))
plt.plot(range(1, iteration_count + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[ ]:


Y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(Y_predicted, Y_truth, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)


# In[ ]:


Y_predicted_test = np.argmax(sigmoid(test_data, W, w0), axis = 1) + 1
confusion_test = pd.crosstab(Y_predicted_test, Y_test, rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_test)

