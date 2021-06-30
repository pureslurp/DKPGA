#!/usr/bin/env python
# coding: utf-8

# In[172]:


#import numpy as np
import pandas as pd
import random


# In[173]:


df = pd.read_csv('DKSalaries.csv')


# In[174]:


df.head()


# In[190]:


def objective(x):
    p0 = []
    for iden in x:
        p0.append(float(df.loc[int(iden)]['AvgPointsPerGame']))
    return sum(p0)

def constraint(x):
    s0 = []
    for iden in x:
        s0.append(float(df.loc[int(iden)]['Salary']))
    return 50000 - sum(s0)

def genIter():
    r0 = []
    while len(r0) < 6:
        r = np.random.randint(0,high=len(df))
        if r not in r0:
            r0.append(r)
    return r0

def getNames(lineup):
    n0 = []
    for iden in lineup:
        n0.append(df.loc[int(iden)]['Name'])
    return n0


# In[191]:


maxIter = 0
i = 0
print(genIter())


# In[192]:


while i < 100:
    lineup = genIter()
    currentIter = objective(lineup)
    if currentIter > maxIter and constraint(lineup) > 0:
        maxIter = currentIter
        maxLineup = lineup
    i = i + 1    


# In[193]:


print(maxIter)
print(getNames(maxLineup))


# In[ ]:




