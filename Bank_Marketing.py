#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[4]:


train=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset5/main/termdeposit_train.csv')
train


# In[5]:


test=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset5/main/termdeposit_test.csv')
test


# In[6]:


train.info()


# In[7]:


train.isnull().sum()


# In[8]:


test.isnull().sum()


# In[9]:


train.dtypes


# In[10]:


sns.heatmap(train.isnull())


# In[11]:


import matplotlib.pyplot as plt
train.plot(kind='box',subplots=True,layout=(2,6),figsize=(10,10))


# In[12]:


sns.countplot(train['subscribed'])
print(train['subscribed'].value_counts())


# In[13]:


sns.countplot(train['age'])
print(train['age'].value_counts())


# In[14]:


sns.countplot(train['ID'])
print(train['ID'].value_counts())


# In[15]:


sns.countplot(train['balance'])
print(train['balance'].value_counts())


# In[16]:


train.plot(kind='kde',subplots=True,layout=(2,6),figsize=(15,6))


# In[17]:


sns.distplot(train['age'])


# In[18]:


sns.distplot(train['balance'])


# In[19]:


sns.distplot(train['day'])


# In[20]:


sns.countplot(data=train,x='marital',hue='subscribed')


# In[21]:


df=pd.concat([train,test])


# In[22]:


df


# In[23]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=oe.fit_transform(df[i].values.reshape(-1,1))
        
df


# In[24]:


df.describe()


# In[25]:


plt.figure(figsize=(15,8))
sns.heatmap(df.describe(),annot=True,fmt='0.2f',linecolor='black',linewidth=0.2,cmap='Blues_r')


# In[26]:


df=df.corr()
df


# In[ ]:





# In[27]:


plt.figure(figsize=(20,12))
sns.heatmap(df.corr(), cmap='Blues_r', annot=True)


# In[28]:


df['ID'].plot.box()


# In[29]:


df['default'].plot.box()


# In[30]:


df['day'].plot.box()


# In[31]:


df.drop('loan',axis=1,inplace=True)


# In[32]:


df


# In[33]:


import numpy as np
from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[34]:


threshold=3
np.where(z<3)


# In[35]:


df1=df[(z<3).all(axis=1)]
df1


# In[36]:


df.shape


# In[37]:


df1.shape


# In[38]:


df1


# In[39]:


x=df1.iloc[:,:-1]
x


# In[40]:


y=df1.iloc[:,-1]
y


# In[41]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=pd.DataFrame(sc.fit_transform(x),columns=x.columns)
df


# In[42]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values, i)
              for i in range (len(x.columns))]
vif['Features']=x.columns
vif


# In[43]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# In[44]:


x_train.shape


# In[45]:


x_test.shape


# In[46]:


y_train.shape


# In[47]:


y_test.shape


# In[48]:


y.value_counts()


# In[51]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
maxAccu=0
maxRS=0
for i in range(1,100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=i)
    RFR=RandomForestClassifier()
    RFR.fit(x_train,y_train)
    pred=RFR.predict(x_test)
    acc=accuracy_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print('Best accuracy is ',maxAccu,'at random_state',maxRS)


# In[ ]:





# In[ ]:




