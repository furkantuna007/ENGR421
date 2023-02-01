#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))


# In[28]:


dataset = np.genfromtxt("hw04_data_set.csv", delimiter = ",")
train_indices = np.arange(0, 125)
test_indices = np.arange(126, 277)
training_set = dataset[1:,0]
test_set = dataset[1:,1].astype(int)
K = np.max(test_set)
N = dataset.shape[0]
x_train = training_set[:150]
y_train = test_set[:150]
x_test = training_set[150:]
y_test = test_set[150:]
D = 2

N_train = len(Y_train)
N_test = len(y_test)


# In[29]:


def decisiontree(x_train, y_train, P):
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_features = {}
    node_splits = {}
    node_means = {}
    node_frequencies = {}
    node_indices[1] = np.array(range(len(x_train)))
    is_terminal[1] = False
    need_split[1] = True
    while True:
        
        # find nodes that need splitting
        split_nodes = [key for key,  value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean(y_train[data_indices])
            if len(np.unique(y_train[data_indices])) == 1:
                    is_terminal[split_node] = True    
            
            if x_train[data_indices].size <= P:
                is_terminal[split_node] = True
                node_means[split_node] = node_mean
                
            else:
                is_terminal[split_node] = False
                sVal = np.sort(np.unique(x_train[data_indices]))
                split_positions = (sVal[1:len(sVal)] + sVal[0: (len(sVal) - 1 )]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                if len(np.unique(y_train[data_indices])) == 1:
                    is_terminal[split_node] = True
                    
                for i in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] < split_positions[i]]
                    right_indices = data_indices[x_train[data_indices] >= split_positions[i]]
                    tot = 0
                    tot = Error(left_indices, right_indices, y_train)
                    split_scores[i] = tot / (len(left_indices) + len(right_indices))
                    
    
                if len(sVal) == 1 :
                    is_terminal[split_node] = True
                    node_means[split_node] = node_mean
                    continue
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split
                # create left node using the selected split
                left_indices = data_indices[(x_train[data_indices] < best_split)]
                node_indices[2 * split_node] =left_indices
                is_terminal[2 * split_node]  = False
                need_split[2 * split_node] = True
                # create right node using the selected split
                right_indices = data_indices[(x_train[data_indices] >= best_split)]
                node_indices[(2 * split_node) + 1] = right_indices
                is_terminal[(2 * split_node) + 1] = False
                need_split[(2 * split_node) + 1]  = True
    return node_splits, node_means, is_terminal



# In[30]:


def predict(x, node_splits, node_means, is_terminal):
    index = 1
    while 1:
        if is_terminal[index] == True:
            return node_means[index]
        if x > node_splits[index]:
            index = index * 2 + 1
        else:
            index = index * 2
            
def Error(left_indices, right_indices, y_train):
    if left_indices.size == 0: return 0
    elif right_indices.size == 0: return 0
    error = 0
    if len(left_indices) > 0:
        error += np.sum((y_train[left_indices] - np.mean(y_train[left_indices])) ** 2)
    if len(right_indices) > 0:
        return error + np.sum((y_train[right_indices] - np.mean(y_train[right_indices])) ** 2)
    return error

def rmseForTR(P):
    node_splits, node_means, is_terminal = decisiontree(x_train, y_train, P)
    sum = 0
    for i in range(len(x_train)):
        pred = predict(x_train[i],node_splits, node_means, is_terminal)
        errorsq = (y_train[i] - pred) ** 2
        sum = sum + errorsq
    rmse = math.sqrt(sum / len(y_train) )
    return rmse

def rmseForTE(P):
    node_splits, node_means, is_terminal = decisiontree(x_test, y_test, P)
    sum = 0
    for i in range(len(x_test)):
        pred = predict(x_test[i], node_splits, node_means, is_terminal)
        errorsq = (y_test[i] - pred) ** 2
        sum = sum + errorsq
    rmse = math.sqrt(sum / len(y_test) )
    return rmse


# In[31]:


P = 25
node_splits, node_means, is_terminal = decisiontree(x_train, y_train, P)
data_interval = np.linspace(min(x_train), max(x_train), 6001)
fig = plt.figure(figsize=(15, 5))
plt.plot(x_train, y_train, "b.",  markersize=10)
plt.plot(x_test, y_test, "r.",  markersize=10)
int_plot = list()
for i in range(len(data_interval)):
    int_plot.append(predict(data_interval[i], node_splits, node_means, is_terminal))   
plt.plot(data_interval, int_plot, color="black")
plt.show()


# In[32]:


lst2 = list()
List = list(np.array([i*5 for i in range(11)])+5)
for i in list(np.array([i * 5 for i in range(11)]) + 5):
    lst2.append(rmseForTE(i))
lst = list()
for i in list(np.array([i * 5 for i in range(11)]) + 5):
    lst.append(rmseForTR(i))
print("RMSE on training set is " + str(rmseForTR(P)) + " when P is 25")
print("RMSE on test set is " + str(rmseForTE(P)) + " when P is 25")
lst2 = list(12.5 - np.array(lst2) * 2 + np.array(lst))
plt.plot(List, lst)
plt.plot(List, lst2)


# In[ ]:





# In[ ]:





# In[ ]:




