#!/usr/bin/env python
# coding: utf-8

# In[860]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# import all the nessary lybraries such as pandas,numpy,matolot and seaborn

# In[861]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv')
df


# loading the dataset from my jupyter notebook and naming as df

# In[862]:


df.info()


# basaball dataset haveing two types of data  integer and float

# In[863]:


df.head()


# In[864]:


df.tail()


# In[865]:


df.isnull().sum()


# no null values are in data set

# In[866]:


sns.heatmap(df.isnull())


# In[867]:


df.describe()


# * The Mean is greater than the median , means there is a right skewed data present
# *the difrence between 75th percentile and the maximum values are high,it indicates there are some outliers present in these columns
# * We need to remove the remove the outliers from the dataset to get better prediction of the model
# 

# In[868]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()

for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=oe.fit_transform(df[i].values.reshape(-1,1))
        
df


# Using The OrdinalEncoder to convert the string data into numarical data, because some of columns has string data types

# In[869]:


plt.figure(figsize=(20,8))
sns.heatmap(df.describe(),annot=True,cmap='Blues_r',linewidth=0.2)


# ##### 

# In[870]:


df['W'].plot.box()


# In[871]:


df['R'].plot.box()


# In[872]:


df['AB'].plot.box()


# In[873]:


df['H'].plot.box()


# In[874]:


df['2B'].plot.box()


# In[875]:


df['3B'].plot.box()


# In[876]:


df['HR'].plot.box()


# In[877]:


df['BB'].plot.box()


# In[878]:


df['SO'].plot.box()


# In[879]:


df['SB'].plot.box()


# In[880]:


df['RA'].plot.box()


# In[881]:


df['ER'].plot.box()


# In[882]:


df['ERA'].plot.box()


# In[883]:


df['CG'].plot.box()


# In[884]:


df['SHO'].plot.box()


# In[885]:


df['SV'].plot.box()


# In[886]:


df['E'].plot.box()


# In[887]:


sns.countplot(df['E'])


# countplot E column are outliers presnt

# In[888]:


sns.distplot(df['E'])


# In[889]:


df.corr()
df


# In[ ]:





# In[890]:


plt.figure(figsize=(8,12))
sns.heatmap(df.corr(),annot=True,fmt='0.2f')


# In[891]:


import numpy as np
from scipy.stats import zscore
z=np.abs(zscore(df))
z


# To remove the outliers using the ZSCORE method , because if the outliers increses then there is chance of wrong prediction

# In[892]:


threshold=3
print(np.where(z>3))


# In[893]:


df1=df[(z<3).all(axis=1)]


# In[894]:


df1.shape


# we can see that outliers are removed in df1 dataset

# In[895]:


df.shape


# In[896]:


df1.skew()


# In[897]:


plt.figure(figsize=(20,25))
p=1

for i in df1.columns:
    if p<30:
        plt.subplot(6,6,p)
        sns.boxplot(data=df1[i])
        plt.xlabel(i)
        plt.ylabel('-')
    p+=1
plt.show()


# useboxpolt methed to find the outliers

# In[898]:


df1.drop(['SV'],axis=1,inplace=True)


# drop sv column becasue of remove the outlier

# In[900]:


x=df1.iloc[:,:-1]
x


# separetion of x columns

# In[901]:


y=df1.iloc[:,-1]
y


# separetion of y column

# In[902]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# import train, test& split to sepretion x train x test and ytrain y test 

# In[903]:


x_train.shape


# In[904]:


x_test.shape


# In[905]:


y_train.shape


# In[906]:


y_test.shape


# In[ ]:





# In[ ]:





# In[907]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1=pd.DataFrame(sc.fit_transform(x),columns=x.columns)
df1


# using the StandardScaler to standardize the all values in the columns with mean equal to zero and Stanadard deviation -0 to +2

# In[908]:


x.shape


# In[909]:


y.shape


# In[910]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values, i)
              for i in range (len(x.columns))]
vif['Features']=x.columns
vif


# to find the multi colinearity of columns using the Vif fuction...
# the vif value of all the columns are less than the 10 it means there is  multi co linearity present in the any columns

# In[899]:


df1.describe()


# In[911]:


y.value_counts()


# In[912]:


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


# In[913]:


from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.linear_model import Lasso,Ridge,ElasticNet


# importing LinearRegression model and also importing r2_score

# In[914]:


print('RandomForestRegressor')
Rd=RandomForestRegressor()
Rd.fit(x_train,y_train)
predrd=Rd.predict(x_test)
print('R2 score of test data',r2_score(y_test,predrd)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predrd))
print('Mean squared error ',mean_squared_error(y_test,predrd))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predrd)))


# In[ ]:





# In[915]:


lr=LinearRegression()
lr.fit(x_train,y_train)
predlr=lr.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predlr)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predlr) )
print('mean squared error : ',mean_squared_error(y_test,predlr))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predlr)))


# in LInearRegression r2sore is -288

# In[916]:


gd=GradientBoostingRegressor()
gd.fit(x_train,y_train)
predgd=gd.predict(x_test)
predgd_train=gd.predict(x_train)
print('R2 Score ',r2_score(y_test,predgd)*100)
print('R2 score on train data ',r2_score(y_train,predgd_train)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predgd))
print('Mean squared error ', mean_squared_error(y_test,predgd))
print('Root mean squared error ',np.sqrt(mean_squared_error(y_test,predgd)))


# GradientBoostingRegressor r2score is -88.05

# In[917]:


Ad=AdaBoostRegressor()
Ad.fit(x_train,y_train)
predad=Ad.predict(x_test)
predad_train=Ad.predict(x_train)
print('R2 score ',r2_score(y_test,predad)*100)
print('R2 score on train ',r2_score(y_train,predad_train)*100)
print('Mean absolute error ', mean_absolute_error(y_test,predad))
print('Mean squared error ', mean_squared_error(y_test,predad))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predad)))


# AdaBoostRegressor r2score-15%

# In[918]:


Et=ExtraTreesRegressor()
Et.fit(x_train,y_train)
predet=Et.predict(x_test)
predet_train=Et.predict(x_train)
print('R2 score ',r2_score(y_test,predet)*100)
print('R2 score on train ',r2_score(y_train,predet_train)*100)
print('Mean absolute error', mean_absolute_error(y_test,predet))
print('Mean sqaured error',mean_squared_error(y_test,predet))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predet)))


# ExtraTreesRegressor r2score -28%

# In[919]:


bg=BaggingRegressor()
bg.fit(x_train,y_train)
predbg=bg.predict(x_test)
predbg_train=bg.predict(x_train)
print('R2 score ',r2_score(y_test,predbg)*100)
print('R2 score on train ', r2_score(y_train,predbg_train)*100)
print('Mean absolute error ', mean_absolute_error(y_test,predbg))
print('Mean sqaured error ', mean_squared_error(y_test,predbg))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predbg)))


# In[920]:


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


# DecisionTreeRegressor r2score -135%

# In[921]:


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


#  we are useing lasso,Ridge,ElasticNet r2score is -45%,-90% and -48%

# In[922]:


from sklearn.model_selection import cross_val_score


# import cross_val_score method

# In[923]:


score=cross_val_score(lr,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[944]:


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


# In[925]:


score=cross_val_score(lr,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[949]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred ))


# In[950]:


from sklearn.model_selection import GridSearchCV


# import GridSearchCV  for the Regresstion model

# In[928]:


parameters= { 
            "n_estimators"      : [10,20,30],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }


# In[939]:


gscv=GridSearchCV(ExtraTreesRegressor(),parameters,cv=5)
gscv.fit(x_train,y_train)
gscv.best_params_


# In[942]:


model=ExtraTreesRegressor(n_estimators=30,max_features='sqrt',min_samples_split=4,bootstrap=True)


# In[943]:


model.fit(x_train,y_train)
pred=model.predict(x_test)
print(r2_score(y_test,pred)*100)


# checking r2score with best model

# In[951]:


import pickle
filename='Bassball'
pickle.dump(model,open(filename,'wb'))


# importing pickle and saving the model

# In[952]:


new_model=pickle.load(open('Bassball','rb'))
new_model.score(x_test,y_test)*100


# Loading the model from jupyternote book and checkin r2score is 18.44%

# In[ ]:




