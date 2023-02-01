#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[3]:


dataset = np.genfromtxt("hw03_data_set.csv", delimiter=",", skip_header=1)
training_data = dataset[0:150]
test_data = dataset[150:272]


# In[14]:


training_data = np.array(training_data)
test_data = np.array(test_data)
trainingX = training_data[:, 0]
testX = test_data[:, 0]
trainingY = training_data[:, 1]
testY = test_data[:, 1].astype(int)
N = testX.shape[0]
minn = min(trainingX)
maxx = max(trainingX)


# In[15]:


bin_width = 0.37
origin = 1.5
leftBorder = np.arange(origin, maxx, bin_width)
rightBorder = np.arange(origin + bin_width, maxx + bin_width, bin_width)
data_interval = np.arange(origin, maxx, step=0.0001)


# In[33]:


p_hat = np.array([np.sum(((trainingX > leftBorder[i]) & (trainingX <= rightBorder[i])) * trainingY) / np.sum((trainingX > leftBorder[i]) & (trainingX <= rightBorder[i])) for i in range(len(leftBorder))])


# In[34]:


plt.figure(figsize=(10, 6))
plt.plot(trainingX , trainingY, "b.", markersize=10)
plt.plot(testX, testY, "r.", markersize=10)
for i in range(len(leftBorder)):
    plt.plot([leftBorder[i], rightBorder[i]], [p_hat[i], p_hat[i]], "k-")
for i in range(len(leftBorder) - 1):
    plt.plot([rightBorder[i], rightBorder[i]], [p_hat[i], p_hat[i + 1]], "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.show()


# In[31]:


y_hat = np.zeros(N_test)
for x in range(len(leftBorder)):
    for y in range(N_test):
        if leftBorder[x] < testX[y] <= rightBorder[x]:
            y_hat[y] = p_hat[int((testX[y] - origin) / bin_width)]

rmse = np.sqrt(np.sum((testY - y_hat) ** 2 / N_test))
print("Regressogram => RMSE is", rmse, "when h is ", bin_width)


# In[26]:


p_hat = np.array([np.sum((np.abs((x - trainingX) / bin_width) < 1) * trainingY)  / np.sum(np.abs((x - trainingX) / bin_width) < 1) for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(trainingX, trainingY, "b.", markersize=10)
plt.plot(testX, testY, "r.", markersize=10)
plt.plot(data_interval, p_hat, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.show()
for i in range(N):
    y_hat[i] = p_hat[int((testX[i] - origin) / 0.0001)]
rmse = np.sqrt(np.sum((testY - y_hat) ** 2 / N))
print("Running Mean Smoother => RMSE is", rmse, "when h is ", bin_width)


# In[28]:


def Kernel(x):
    return 1 / np.sqrt(math.pi * 2) * np.exp(- x ** 2 / 2)
p_hat = np.array([np.sum(Kernel((x - trainingX) / bin_width) * trainingY) / np.sum(Kernel((x - trainingX) / bin_width)) for x in data_interval])
plt.figure(figsize=(10, 6))
plt.plot(trainingX, trainingY, "b.", markersize=10)
plt.plot(testX, testY, "r.", markersize=10)
plt.plot(data_interval, p_hat, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.show()
for i in range(N):
    y_hat[i] = p_hat[int((testX[i] - origin) / 0.0001)]    
print(N)
rmse = np.sqrt(np.sum((testY - y_hat) ** 2 / N_test))
print("Kernel Smoother => RMSE is", rmse, "when h is ", bin_width)


# In[ ]:





# In[ ]:




