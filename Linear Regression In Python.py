#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[3]:


iris = pd.read_csv('iris.csv')
iris.head()


# In[8]:


iris[iris['sepal_width'] > 4] 


# In[9]:


iris[iris['petal_width'] > 1]


# In[15]:


#RELATION BETWEEN PETAL LENGTH AND SEPAL LENGTH USING SCATTERPLOT
sns.scatterplot(x = 'sepal_length', y = 'petal_length', data = iris, hue = "species")
plt.show()


# In[16]:


iris.head()


# In[20]:


y = iris[['sepal_length']]
x = iris[['sepal_width']]
#SKLEARN HAS MODULES LIKE LINEAR REGRESSION, LOGISTIC REGRESSION, RANDOM FOREST

from sklearn.model_selection import train_test_split
#0.3 --> 30% of the entire data set will used as a test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3) 


# In[24]:


#x_train.head()
y_train.head()


# In[28]:


from sklearn.linear_model import LinearRegression


# In[30]:


#CREATING AN INSTANCE FOR THE LINEAR REGRESSION
LR = LinearRegression()
LR.fit(x_train, y_train)


# In[31]:


y_predicted = LR.predict(x_test)


# In[33]:


y_predicted[0:10]


# In[34]:


from sklearn.metrics import mean_squared_error


# In[35]:


mean_squared_error(y_test, y_predicted)


# In[37]:


#MODEL - 2. MODEL WITH MORE INDEPENDENT VARIABLES


# In[36]:


y = iris[['sepal_length']]
x = iris[['sepal_width', 'petal_length', 'petal_width']]


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
LR2 = linear_regression()
LR2.fit(x__test, y_test)
LR2_predict = LR2.predict(x_test)


# In[40]:


mean_squared_error(y_test, y_predicted)


# In[ ]:


#THUS AS WE HAVE USED MULTIPLE VARIABLES, THE MEAN SQUARED ERROR IS LESS, THAT IS THE ACCURACY HAS INCREASED

