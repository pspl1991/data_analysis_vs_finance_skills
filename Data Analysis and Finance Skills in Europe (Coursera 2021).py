#!/usr/bin/env python
# coding: utf-8

# ## FIRST REGRESSION IN PYTHON 

# ##### Context: In this dataset, I will analyse the principal skills in Europe according to Coursera (2021), with special focus in Finance and Data Analysis. The reason? Is two of my main profissional areas :)

# ### Simple Linear Regression

# 1st: Import the relevant libraries

# In[64]:


import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib as plt
import seaborn as sns
sns.set()
import sklearn
import os


# 2nd: Load the data

# In[65]:


data = pd.read_csv('Europe_Skills.csv')


# In[66]:


data


# In[67]:


data.describe()


# ### Create the first regression

# 3.1: Define the dependent and the independent variables

# In[68]:


y = data['Finance']

x1 = data['Data Analysis']


# 3.2: Explore the data

# In[69]:


from matplotlib import pyplot as plt


# In[70]:


plt.scatter(x1,y)
plt.xlabel('Data Analysis',fontsize=20)
plt.ylabel('Finance',fontsize=20)
plt.show()


# ### Regression Itself

# In[71]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[72]:


plt.scatter(x1,y)
yhat = 0.5885*x1+0.4075
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('Data Analysis', fontsize = 20)
plt.ylabel('Finance', fontsize = 20)
plt.show()


# In[73]:


plt.scatter(data['Data Analysis'],y)
yhat_no = 0.5885+0.4075*data['Data Analysis']
yhat_yes = 0.8665 + 0.0014*data['Data Analysis']
fig = plt.plot(data['Data Analysis'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['Data Analysis'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('Data Analysis', fontsize = 20)
plt.ylabel('Finance', fontsize = 20)
plt.show()

