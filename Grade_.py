#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# import nessesary lybreries for loading dataset

# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset4/main/Grades.csv')
df


# grade dataset having 571 rows 43 columns

# In[3]:


df.isnull().sum()


# null values presnt in allmost 38 columns

# In[4]:


df.info()


# In[5]:


df.dropna(inplace=True)


# drop method used to remove the null values

# In[6]:


df.info()


# In[7]:


df.dtypes


# two type of data in this data set
# 1 object
# 2 float

# In[8]:


df.isnull().sum()


# no null values after use drop method

# In[9]:


sns.heatmap(df.isnull())


# In[11]:


df.corr()
df


# corelation is clear showes how many columns are corelate with target

# In[12]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()

for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=oe.fit_transform(df[i].values.reshape(-1,1))
        
        
df


# In[13]:


df.describe()


# In[14]:


sns.countplot(df['CS-105'])


# In[15]:


plt.figure(figsize=(30,20))
sns.heatmap(df.corr(),annot=True,fmt='0.2f')


# its showes datawill corelate with target

# In[16]:


import seaborn as sns
sns.distplot(df)


# distplot is shoews skewness of data 

# In[17]:


df['CS-106'].plot.box()


# plotbox cs-106 is no outliers

# In[18]:


df['Seat No.'].plot.box()


# seatno also no outliers

# In[19]:


df['PH-121'].plot.box()


# in ph their is no  outliers

# In[20]:


df['CS-317'].plot.box()


# In[21]:


import numpy as np
from scipy.stats import zscore
z=np.abs(zscore(df))
z


# zscore method use to remove outliers present in dataset 

# In[22]:


threshold=3
print(np.where(z>3))


# setting threshold to 3 because 99% of the data is present in the range of 12 standard deviation

# In[ ]:


df1=df[(z<3).all(axis=1)]


# In[24]:


df1.shape


# data is saved in df1 and also completly remove outliers in data

# In[25]:


df.shape


# In[26]:


print('persentage of data los',(df.shape[0]-df1.shape[0]/df.shape[0])*100)


# In[27]:


df1.describe()


# In[28]:


plt.figure(figsize=(30,15))
sns.heatmap(df1.corr(),annot=True)


# In[29]:


x=df1.iloc[:,:-1]
x


# separeting dataset into x train and test data 

# In[30]:


y=df1.iloc[:,-1]
y


# separet dataset into y train test data 

# In[31]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=pd.DataFrame(sc.fit_transform(x),columns=x.columns)
df


# use standardscaler method to stadardize the data

# In[32]:


x.shape


# In[33]:


y.shape


# In[34]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values, i)
              for i in range (len(x.columns))]
vif['Features']=x.columns
vif


# In[ ]:





# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# importing the train and test model to separate the train,test data

# In[36]:


x_train.shape


# In[37]:


x_test.shape


# In[38]:


y_train.shape


# In[39]:


y_test.shape


# In[40]:


y.value_counts()


# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
maxAccu=0
maxRs=0
for i in range(1,1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    acc=r2_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRs=i
print("Best Accuracy is", maxAccu,"at random_state ",maxRs)


# since my target is having contineous data, so i have to choose Regression model

# In[42]:


from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.linear_model import Lasso,Ridge,ElasticNet


# importing the nessasary libraries for Regression model

# In[43]:


print('RandomForestRegressor')
Rd=RandomForestRegressor()
Rd.fit(x_train,y_train)
predrd=Rd.predict(x_test)
print('R2 score of test data',r2_score(y_test,predrd)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predrd))
print('Mean squared error ',mean_squared_error(y_test,predrd))


# In[65]:


print('LinearRegression')
lr=LinearRegression()
lr.fit(x_train,y_train)
predlr=lr.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predlr)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predlr) )
print('mean squared error : ',mean_squared_error(y_test,predlr))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predlr)))


# linear regression r2 score is 99%

# In[52]:


print('ExtraTreesRegressor')
et=ExtraTreesRegressor()
et.fit(x_train,y_train)
predet=et.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predet)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predet) )
print('mean squared error : ',mean_squared_error(y_test,predet))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predet)))


# In[56]:


print('GradientBoostingRegressor')
gb=GradientBoostingRegressor()
gb.fit(x_train,y_train)
predgb=gb.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predgb)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predgb) )
print('mean squared error : ',mean_squared_error(y_test,predgb))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predgb)))


# GradiantBosstingRegression r2 score 97%

# In[55]:


print('AdaBoostRegressor')
ad=AdaBoostRegressor()
ad.fit(x_train,y_train)
predad=ad.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predad)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predad) )
print('mean squared error : ',mean_squared_error(y_test,predad))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predad)))


# In[57]:


print('BaggingRegressor')
bg=BaggingRegressor()
bg.fit(x_train,y_train)
predbg=bg.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predbg)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predbg) )
print('mean squared error : ',mean_squared_error(y_test,predbg))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predbg)))


# In[58]:


print('DecisionTreeRegressor')
print('BaggingRegressor')
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
preddt=dt.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,preddt)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,preddt) )
print('mean squared error : ',mean_squared_error(y_test,preddt))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,preddt)))


# In[59]:


ls=Lasso()
ls.fit(x_train,y_train)
predls=ls.predict(x_test)
predls_train=ls.predict(x_train)
print('R2 score ',r2_score(y_test,predls)*100)
print('R2 score on train ', r2_score(y_train,predls_train)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predls))
print('Mean squared error ',mean_squared_error(y_test,predls))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predls)))
print('\n')

rd=Ridge()
rd.fit(x_train,y_train)
predrd=rd.predict(x_test)
predrd_train=rd.predict(x_train)
print('R2 score ', r2_score(y_test,predrd)*100)
print('R2 score on train data ', r2_score(y_train,predrd_train)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predrd))
print('Mean squared error ', mean_squared_error(y_test,predrd))
print('Root mean squared error ',np.sqrt(mean_squared_error(y_test,predrd)))
print('\n')

el=ElasticNet()
el.fit(x_train,y_train)
predel=el.predict(x_test)
predel_train=el.predict(x_train)
print('R2 score ',r2_score(y_test,predel)*100)
print('R2 score on train data',r2_score(y_train,predel_train)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predel))
print('Mean squared error ',mean_squared_error(y_test,predel))
print('Root mean sqaured error ', np.sqrt(mean_squared_error(y_test,predel)))


# In[60]:


from sklearn.model_selection import cross_val_score


# In[66]:


print('GradientBoosting')
score=cross_val_score(gb,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w R2 score and cross validation score',(r2_score(y_test,predgb))-(score.mean())*100)
print('\n')

print('BaggingRegressor')
score=cross_val_score(bg,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w R2 score and cross validation score', (r2_score(y_test,predbg)-(score.mean()))*100)
print('\n')

print('ExtraTree')
score=cross_val_score(et,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and cross validation score is ',(r2_score(y_test,predet)-(score.mean()))*100)
print('\n')

print('AdaBoost')
score=cross_val_score(ad,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and cross validation score is ',(r2_score(y_test,predad)-(score.mean()))*100)
print('\n')

print('LinearRegression')
score=cross_val_score(lr,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and cross validation score is ',(r2_score(y_test,predlr)-(score.mean()))*100)
print('\n')


print('DecisionTree')
score=cross_val_score(dt,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and Cross validation score ',(r2_score(y_test,preddt)-(score.mean()))*100)
print('\n')

print('Lasso')
score=cross_val_score(ls,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and Cross validation score ',(r2_score(y_test,predls)-(score.mean()))*100)
print('\n')

print('Ridge')
score=cross_val_score(rd,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and Cross validation score ',(r2_score(y_test,predrd)-(score.mean()))*100)
print('\n')

print('ElasticNet')
score=cross_val_score(el,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and Cross validation score ',(r2_score(y_test,predel)-(score.mean()))*100)
print('\n')


# In[67]:


from sklearn.model_selection import GridSearchCV


# In[70]:


param = { 'loss' : ['ls', 'lad', 'huber', 'quantile'],
              'learning_rate' : (0.05,0.25,0.50,1),
              'criterion' : ['friedman_mse', 'mse', 'mae'],
              'max_features' : ['auto', 'sqrt', 'log2']
             }
gscv=GridSearchCV(GradientBoostingRegressor(),param,cv=5)
gscv.fit(x_train,y_train)


# In[71]:


gscv.best_params_


# In[72]:


model=GradientBoostingRegressor(criterion='mse',learning_rate=0.05,loss='ls',max_features='auto')


# In[ ]:





# In[73]:


model.fit(x_train,y_train)
pred_model=model.predict(x_test)
print('R2 score ', r2_score(y_test,pred_model)*100)
print('Mean absolute error ',mean_absolute_error(y_test,pred_model))
print('Mean squared error ', mean_squared_error(y_test,pred_model))
print('Root mean squared error ',np.sqrt(mean_squared_error(y_test,pred_model)))      


# In[77]:


import pickle
filename='Grade_'
pickle.dump(model,open(filename,'wb'))


# In[78]:


import pickle
new_model=pickle.load(open('Grade_','rb'))


# In[79]:


score=new_model.score(x_test,y_test)
print(score*100)


# GradientBoostingRegressor is best model to predict my grade 

# In[ ]:




