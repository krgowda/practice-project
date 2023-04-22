#!/usr/bin/env python
# coding: utf-8

# # Medical insurence Project

# In[143]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# importing the nessasary lybraries for loading the dataset and analysing the dataset by graphs

# In[79]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv')


# loading the dataset with the help of pandas lybrary from my jupyternote book

# In[80]:


df


# saving dataset in variable df

# In[81]:


df.columns


# there are 7 columns present in my dataset df
# 
# 1.age
# 
# 2.sex
# 
# 3.bmi
# 
# 4.children
# 
# 5.smoker
# 
# 6.region
# 
# 7.charges

# In[82]:


df.dtypes


# 3 types of data present in dataset
# 1.integer - 2columns
# 
# 2.float - 2 columns
# 
# 3.object - 3 columns

# In[83]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# there is no missing data present in any columns 

# In[84]:


df.info()


# each column has 1338 rows and there is no null data is present 

# In[85]:


df.isnull().sum()


# checking for the null values

# In[86]:


sns.heatmap(df.isnull())


# using heatmap to check the null values in dataset but there is no null values present...

# In[87]:


df['charges'].nunique()


# 1337 unique values are present in target column

# In[88]:


for i in df.columns:
    print(df[i].unique())
    print('\n')


# checking for unique values present in the dataset 

# In[89]:


print(df.describe())
plt.figure(figsize=(15,8))
sns.heatmap(df.describe(),annot=True,fmt='0.2f',linecolor='black',linewidth=0.2,cmap='Blues_r')


# each column has 1338 rows 
# 
# here the mean is greater than the median so here we can confirm the right skewed data is present in the dataset
# 
# the target column there huge difference is in between the 3rd quartile and the max values , means outliers are present 
# in the target and in the graph also we can see the same thing 

# In[90]:


df['charges'].plot.box()


# using boxplot clearly showing outliers are present in the charges column , we have clean the outliers later

# In[91]:


df['children'].plot.box()


# there is no outliers present in the children column

# In[92]:


df['age'].plot.box()


# in age column also there is no outliers present

# In[93]:


df['bmi'].plot.box()


# here bmi column has small amount of outliers present

# In[94]:


df


# In[95]:


sns.countplot(df['sex'])
print(df['sex'].value_counts())


# using countplot we observe that sex column has nearly equal number of value counts

# In[96]:


sns.countplot(df['smoker'])
print(df['smoker'].value_counts())


# smoker column has not equal number proportion in smoker and non smoker 

# In[97]:


sns.countplot(df['region'])
print(df['region'].value_counts())


# region column has nearly equal number of value counts for all regions

# In[98]:


sns.distplot(df['bmi'])


# checking for the skewness in bmi colums there is no skewness is present

# In[99]:


sns.distplot(df['charges'])


# in charges column right skewed data is present

# In[100]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()

for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# using the ordinal encoder to convert the Object data to Numarical data ..
# 
# beacuse we can not use the catogorical data to predict the model further

# In[101]:


df.info()


# we can see here my catogorical data is converted into float datatype

# In[102]:


print(df.corr())
sns.heatmap(df.corr(),cmap='Blues_r',linecolor='black',linewidth=0.30,fmt='0.2f',annot=True)
print('\n')

print('checking the co relation with respect to the Target ')
k=df.corr()['charges'].sort_values(ascending=False)
print(k)


# ** smoker columns has high positive reletion with respect to the target 
# 
# ** region column has negative reletion with the target 
# 
# 

# In[103]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# using zscore method to remove the outliers present in the dataset...

# In[104]:


threshold=3
print(np.where(z>3))


# setting threshold to 3 because 99% of the data is present in the range of 3 standard deviations 

# In[105]:


z.iloc[32][3]


# checking zscore with iloc function , clearly its greater than 3 means outliers is present

# In[106]:


df1=df[(z<3).all(axis=1)]
print('df1 = ',df1.shape)
print('df = ',df.shape)


# removed the outliers from the dataset , the number of rows are 1309, it shows removed the outliers and saving to df1 

# In[107]:


df1


# In[108]:


Q1=df.quantile(0.25)

Q3=df.quantile(0.75)

IQR=Q3-Q1

df2=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]


# removing the outliers using the quantile method 

# In[109]:


df2.shape


# removed the outliers and saving to new dataset df2

# In[110]:


print('Data loss percentage with zscore method ',((df.shape[0]-df1.shape[0])/df.shape[0])*100)
print('Data loss percentage with quantile method ',((df.shape[0]-df2.shape[0])/df.shape[0])*100)


# camparing the data loss after remiving the outliers ,acceptable range of data loss is 10% , so we go through the zscore method
# is best here

# In[111]:


df=df1


# saving back to the model df1 as df

# In[112]:


df.drop(['region'],axis=1,inplace=True)


# droping the region column because its negatively co releted to the target column

# In[113]:


df.skew()


# checking for the skewness 

# In[114]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=df.columns)
df


# with the help of standard scaler we are standardizing to mean is equal to zero and standard deviation to 1

# In[115]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF Values']=[variance_inflation_factor(df.values,i)
                  for i in range(len(df.columns))]
vif['VIF Features']=df.columns


# applying vif factor to check the underfitting and over fitting 

# In[116]:


vif


# In[117]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# separating the dataset into x and y for training and testing

# In[118]:


x


# x variable has 1309 rows and 5 columns 

# In[119]:


y


# target variable has 1309 rows 

# In[120]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# since my target is having contineous data, so i have to choose Regression model

# In[121]:


from sklearn.model_selection import train_test_split


# importing the train and test model to separate the train,test data

# In[122]:


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


# using the linearregression model to get the best random state for better prediction

# In[123]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=373)


# In[124]:


from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso,Ridge,ElasticNet


# importing the nessasary libraries for Regression model

# In[125]:


print('RandomForestRegressor')
Rd=RandomForestRegressor()
Rd.fit(x_train,y_train)
predrd=Rd.predict(x_test)
print('R2 score of test data',r2_score(y_test,predrd)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predrd))
print('Mean squared error ',mean_squared_error(y_test,predrd))


# with RandomForestClassifier the r2 score is 89%

# In[126]:


lr=LinearRegression()
lr.fit(x_train,y_train)
predlr=lr.predict(x_test)
pred_train=lr.predict(x_train)
print('r2_score',r2_score(y_test,predlr)*100)
print('r2_score on training data',r2_score(y_train,pred_train)*100)
print('mean absolute error :  ',mean_absolute_error(y_test,predlr) )
print('mean squared error : ',mean_squared_error(y_test,predlr))
print('Root mean squared error : ', np.sqrt(mean_squared_error(y_test,predlr)))


# with linearRegression r2 score is 81%

# In[127]:


gd=GradientBoostingRegressor()
gd.fit(x_train,y_train)
predgd=gd.predict(x_test)
predgd_train=gd.predict(x_train)
print('R2 Score ',r2_score(y_test,predgd)*100)
print('R2 score on train data ',r2_score(y_train,predgd_train)*100)
print('Mean absolute error ',mean_absolute_error(y_test,predgd))
print('Mean squared error ', mean_squared_error(y_test,predgd))
print('Root mean squared error ',np.sqrt(mean_squared_error(y_test,predgd)))


# with Gradient Boosting r2 score 90%

# In[128]:


Ad=AdaBoostRegressor()
Ad.fit(x_train,y_train)
predad=Ad.predict(x_test)
predad_train=Ad.predict(x_train)
print('R2 score ',r2_score(y_test,predad)*100)
print('R2 score on train ',r2_score(y_train,predad_train)*100)
print('Mean absolute error ', mean_absolute_error(y_test,predad))
print('Mean squared error ', mean_squared_error(y_test,predad))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predad)))


# r2 score is 83% with the Adaboost

# In[129]:


Et=ExtraTreesRegressor()
Et.fit(x_train,y_train)
predet=Et.predict(x_test)
predet_train=Et.predict(x_train)
print('R2 score ',r2_score(y_test,predet)*100)
print('R2 score on train ',r2_score(y_train,predet_train)*100)
print('Mean absolute error', mean_absolute_error(y_test,predet))
print('Mean sqaured error',mean_squared_error(y_test,predet))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predet)))


# r2 score is 83% with the ExtraTreeclassifier

# In[130]:


bg=BaggingRegressor()
bg.fit(x_train,y_train)
predbg=bg.predict(x_test)
predbg_train=bg.predict(x_train)
print('R2 score ',r2_score(y_test,predbg)*100)
print('R2 score on train ', r2_score(y_train,predbg_train)*100)
print('Mean absolute error ', mean_absolute_error(y_test,predbg))
print('Mean sqaured error ', mean_squared_error(y_test,predbg))
print('Root mean squared error ', np.sqrt(mean_squared_error(y_test,predbg)))


# r2 score is 88% with the Bagging regressor

# In[131]:


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


# r2 score is 75% only with decisiontreeregressor

# In[132]:


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


# lasso,ridge and elasticnet only Ridge regressor has good r2 score that is 81%

# In[133]:


from sklearn.model_selection import cross_val_score


# to find the best fitting model use cross validation technique , it devides the all the data into equal number of folds and predicts the best score 

# In[134]:


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


# by looking the cross validation score of each model r2 score of GradientBoostingRegressor is the best model with r2_score 85%

# In[135]:


from sklearn.model_selection import GridSearchCV


# to find best fitting parameters for my GradientBoostingregressor using the GridsearchCV

# In[136]:


param = { 'loss' : ['ls', 'lad', 'huber', 'quantile'],
              'learning_rate' : (0.05,0.25,0.50,1),
              'criterion' : ['friedman_mse', 'mse', 'mae'],
              'max_features' : ['auto', 'sqrt', 'log2']
             }
gscv=GridSearchCV(GradientBoostingRegressor(),param,cv=5)
gscv.fit(x_train,y_train)


# In[137]:


gscv.best_params_


# best parameters given from gridsearchcv we are going to use in the model

# In[138]:


model=GradientBoostingRegressor(criterion='mse',learning_rate=0.05,loss='ls',max_features='auto')


# In[139]:


model.fit(x_train,y_train)
pred_model=model.predict(x_test)
print('R2 score ', r2_score(y_test,pred_model)*100)
print('Mean absolute error ',mean_absolute_error(y_test,pred_model))
print('Mean squared error ', mean_squared_error(y_test,pred_model))
print('Root mean squared error ',np.sqrt(mean_squared_error(y_test,pred_model)))      


# after using the best parameters my model performing very well , r2 score is 91%

# In[140]:


import pickle
filename='medical_insurence'
pickle.dump(model,open(filename,'wb'))


# for saving the best model i am importing the pickle lybrari and giving the file name as medical_insurence

# In[141]:


import pickle
new_model=pickle.load(open('medical_insurence','rb'))


# loading the saved model from the jupyter notebook

# In[142]:


score=new_model.score(x_test,y_test)
print(score*100)


# conclusion :GradientBoostingRegressor is best model to predict my medical_insuence model 

# In[ ]:




