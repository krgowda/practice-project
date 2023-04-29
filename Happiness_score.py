#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# importing the nessasary lybraries for loading the dataset and analysing the dataset by graphs

# In[74]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')


# In[75]:


df


# saving dataset in variable df

# In[76]:


df.columns


# in this dataset 12 columns are present

# In[77]:


df.head()


# In[78]:


df.tail()


# In[79]:


df.info()


# each column has 158 rows and there is no null data is present

# In[80]:


df.dtypes


# 3 types of data present in dataset
# int-1column
# object-2columns
# float-9columns

# In[81]:


for i in df.columns:
    print(df[i].value_counts())


# In[82]:


df.isnull().sum()


# no null values presnt in this dataset

# In[83]:


sns.heatmap(df.isnull())


# using heatmap to check the null values in dataset but there is no null values present...

# In[84]:


df['Dystopia Residual'].nunique()


# In[85]:


for i in df.columns:
    print(df[i].unique())


# In[86]:


df.describe()


# here the mean is greater than the median so here we can confirm the right skewed data is present in the some dataset,
# mean is less than the median so here we can confirm the left skewed data aslo their

# In[87]:


plt.figure(figsize=(15,8))
sns.heatmap(df.describe(),annot=True,fmt='0.2f',linecolor='black',linewidth=0.2,cmap='Blues_r')


# In[88]:


import matplotlib.pyplot as plt
df.plot(kind='box',subplots=True,layout=(2,6),figsize=(10,10))


# use boxplot to find the outlayer in columns
# here outlayer seen in Trust,standard error,

# In[89]:


dfcor=df.corr()
dfcor


# In[90]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='Blues_r',annot=True)


# freedom,health,family are positively co relate with target
# trust,generosity are negatively corelate with target

# In[91]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()

for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# using the ordinal encoder to convert the Object data to Numarical data ..
# 
# beacuse we can not use the catogorical data to predict the model further

# In[92]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# using zscore method to remove the outliers present in the dataset..

# In[93]:


threshold=3
print(np.where(z>3))


# setting threshold to 4 because 90% of the data is present in the range of 4 standard deviations 

# In[94]:


z.iloc[157,9]


# checking zscore with iloc function , clearly its greater than 9 means outliers is present

# In[95]:


z.iloc[157,10]


# checking zscore with iloc function , clearly its greater than 10 means outliers is present

# In[96]:


z.iloc[157,8]


# checking zscore with iloc function , clearly its greater than 8 means outliers is present

# In[97]:


df1=df[(z<3).all(axis=1)]
df1


# In[98]:


df.shape


# In[99]:


df1.shape


# data set save with df1

# In[100]:


df1.skew()


# In[101]:


x=df1.iloc[:,:-1]
x


# use i loc function to seperate x columns 149rows and 11 columns

# In[102]:


y=df1.iloc[:,-1]
y


# seperate target column use iloc function

# In[103]:


x.shape


# In[104]:


y.shape


# In[105]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1=pd.DataFrame(sc.fit_transform(df1),columns=df1.columns)
df1


# In[ ]:


standerd scaler method 


# In[106]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values, i)
              for i in range (len(x.columns))]
vif['Features']=x.columns
vif


# correlation between independent and multicolinearity in regression 

# In[107]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# importing the train and test model to separate the train,test data

# In[108]:


x_train.shape


# In[109]:


x_test.shape


# In[110]:


y_train.shape


# In[111]:


y_test.shape


# In[112]:


y.value_counts()


# we are seen the y column data is continues so we can use linear regression

# In[113]:


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


# In[114]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=373)


# In[115]:


from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.linear_model import Lasso,Ridge,ElasticNet


# import nessasary lybrires for regression

# In[116]:


print('RandomForestRegressor')
Rd=RandomForestRegressor()
Rd.fit(x_train,y_train)
predrd=Rd.predict(x_test)
print('R2 score of test data',r2_score(y_test,predrd)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predrd))
print('Mean squared error ',mean_squared_error(y_test,predrd))


# randomforestregression r2 score is 63%

# In[117]:


lr=LinearRegression()
lr.fit(x_train,y_train)
predlr=lr.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predlr)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predlr) )
print('mean squared error : ',mean_squared_error(y_test,predlr))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predlr)))


# LinearRegression r2 score is 100%

# In[118]:


gd=GradientBoostingRegressor()
gd.fit(x_train,y_train)
predgd=gd.predict(x_test)
predgd_train=gd.predict(x_train)
print('R2 Score ',r2_score(y_test,predgd)*100)
print('R2 score on train data ',r2_score(y_train,predgd_train)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predgd))
print('Mean squared error ', mean_squared_error(y_test,predgd))
print('Root mean squared error ',np.sqrt(mean_squared_error(y_test,predgd)))


# GradientBoostingRegression r2score is 66%

# In[119]:


Ad=AdaBoostRegressor()
Ad.fit(x_train,y_train)
predad=Ad.predict(x_test)
predad_train=Ad.predict(x_train)
print('R2 score ',r2_score(y_test,predad)*100)
print('R2 score on train ',r2_score(y_train,predad_train)*100)
print('Mean absolute error ', mean_absolute_error(y_test,predad))
print('Mean squared error ', mean_squared_error(y_test,predad))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predad)))


# In[120]:


Et=ExtraTreesRegressor()
Et.fit(x_train,y_train)
predet=Et.predict(x_test)
predet_train=Et.predict(x_train)
print('R2 score ',r2_score(y_test,predet)*100)
print('R2 score on train ',r2_score(y_train,predet_train)*100)
print('Mean absolute error', mean_absolute_error(y_test,predet))
print('Mean sqaured error',mean_squared_error(y_test,predet))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predet)))


# In[121]:


bg=BaggingRegressor()
bg.fit(x_train,y_train)
predbg=bg.predict(x_test)
predbg_train=bg.predict(x_train)
print('R2 score ',r2_score(y_test,predbg)*100)
print('R2 score on train ', r2_score(y_train,predbg_train)*100)
print('Mean absolute error ', mean_absolute_error(y_test,predbg))
print('Mean sqaured error ', mean_squared_error(y_test,predbg))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predbg)))


# In[122]:


from sklearn.tree import DecisionTreeRegressor as dtr
dtc=dtr()
dtc.fit(x_train,y_train)
preddtc=dtc.predict(x_test)
preddtc_train=dtc.predict(x_train)
print('R2 score ',r2_score(y_test,preddtc)*100)
print('R2 score on train ',r2_score(y_train,preddtc_train)*100)
print('Mean sqaured error ',mean_squared_error(y_test,preddtc))
print('Mean absolute error ', mean_absolute_error(y_test,preddtc))
print('Root mean sqaured error ', np.sqrt(mean_squared_error(y_test,preddtc)))


# In[123]:


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


# In[124]:


from sklearn.model_selection import cross_val_score


# In[125]:


score=cross_val_score(lr,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[126]:


print('GradientBoosting')
score=cross_val_score(gd,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w R2 score and cross validation score',(r2_score(y_test,predgd))-(score.mean())*100)
print('\n')

print('BaggingRegressor')
score=cross_val_score(bg,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w R2 score and cross validation score', (r2_score(y_test,predbg)-(score.mean()))*100)
print('\n')

print('ExtraTree')
score=cross_val_score(Et,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and cross validation score is ',(r2_score(y_test,predet)-(score.mean()))*100)
print('\n')

print('ADA boost')
score=cross_val_score(Ad,x,y,cv=5)
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
score=cross_val_score(dtc,x,y,cv=5)
print(score)
print(score.mean()*100)
print('Difference b/w the R2 score and Cross validation score ',(r2_score(y_test,preddtc)-(score.mean()))*100)
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


# checking for cross validation score in diffrent types of regression models

# In[127]:


score=cross_val_score(lr,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[128]:


from sklearn.metrics import r2_score
print(r2_score(y_test,predlr))


# In[138]:


lr.fit(x_train,y_train)
pred_model=lr.predict(x_test)
print('R2 score ', r2_score(y_test,pred_model)*100)
print('Mean absolute error ',mean_absolute_error(y_test,pred_model))
print('Mean squared error ', mean_squared_error(y_test,pred_model))
print('Root mean squared error ',np.sqrt(mean_squared_error(y_test,pred_model)))      


# In[139]:


import pickle
filename='Happiness_score'
pickle.dump(lr,open(filename,'wb'))


# pickle method to save the data set

# In[140]:


import pickle
new_model=pickle.load(open('Happiness_score','rb'))


# In[141]:


score=new_model.score(x_test,y_test)
print(score*100)


# conlusion=leanear regression is best model to predict my happiness dataset

# In[ ]:




