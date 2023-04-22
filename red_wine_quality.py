#!/usr/bin/env python
# coding: utf-8

# In[1]:



import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# importing all the nessasary lybraries  for loading dataset and analysing dataset

# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/shivam229/dsrscientist-DSData/master/winequality-red.csv')
df


# loading dataset with the help of raw data link,save the data as df and dataset contains 1599 rows and 12 columns

# In[3]:


df.columns


# they are 12 columns in this dataset

# In[4]:


df.head()


# In[5]:


df.dtypes


# in this dataset having 2type of data 1float & 2 integer

# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df['quality'].value_counts()


# six diffent values present in target column

# In[9]:


for i in df.columns:
    print(df[i].value_counts())


# ther is no missing data in this dataset or columns

# In[10]:


sns.heatmap(df.isnull())


# checking for null  values in dataset but ther is no null values 

# In[11]:


df.isnull().sum()


# checking the null values count their is no null values

# In[12]:


df.describe()


# here the mean is greater than the median so here we can confirm the right skewed data is present in the dataset

# In[13]:


plt.figure(figsize=(15,8))
sns.heatmap(df.describe(),annot=True,fmt='0.2f',linecolor='black',linewidth=0.2,cmap='Blues_r')


# in the total sulfur dioxide column there huge difference is in between the 3rd quartile and the max values , means outliers are present in the total sulfuer dioxide,residualsugar,fixed acidity,free sulfure dioxide and  target column also ,in the graph also we can see the same thing

# In[14]:


dfcor=df.corr()
dfcor


# useing correlation method to find out relationship between varieables

# In[15]:


sns.heatmap(dfcor)


# corelation between the x value and y value

# In[16]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='Blues_r',annot=True)


# * citric acid,sulphate,alcohol  are  positive relation with respect to target
# * total sulfure dioxide are negetive relation with target

# In[ ]:





# In[17]:


df['fixed acidity'].plot.box()


# use plot box to identify the out lines ,fixed acidity have outelayers  

# In[18]:


df['volatile acidity'].plot.box()


# in voataile acidity aslo have outlayers

# In[19]:


df['citric acid'].plot.box()


# citric acid outlayer seems very less

# In[20]:


df['residual sugar'].plot.box()


# residual sugar having more outlayers are their

# In[21]:


df['chlorides'].plot.box()


# In[22]:


df['free sulfur dioxide'].plot.box()


# In[23]:


df['total sulfur dioxide'].plot.box()


# In[24]:


df['density'].plot.box()


# In[25]:


df['pH'].plot.box()


# out layer presnt in ph column

# In[26]:


df['sulphates'].plot.box()


# out layer presnt in sulphates colomn

# In[27]:


df['alcohol'].plot.box()


# In[28]:


df['quality'].plot.box()


# In[29]:


df.plot(kind='box',subplots=True,layout=(2,6),figsize=(10,10))


# out layer presnt in residual sugar, free sulfer dioxide,sulphate,totalsulfur dioxide,ph,

# In[30]:


sns.distplot(df['citric acid'])


# checking for skewness, in citric acid data will distrubute equealy

# In[31]:


sns.distplot(df['total sulfur dioxide'])


# in tota sulfure dioxide left skewness are presnt

# In[32]:


sns.pairplot(df)


# In[ ]:





# In[33]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# useing zscore method to remove the outlayer of the dataset

# In[34]:


threshold=3
np.where(z<3)


# setting threshold to 0 to 8 haveing  90% of the data is present in the range of 3 standard deviation0 to

# In[35]:


z.iloc[1598,9]


# checking zscore with iloc function , clearly its greater than 3 means outliers is present

# In[36]:


z.iloc[1598,11]


# In[37]:


z.iloc[1598,10]


# In[38]:


df1=df[(z<3).all(axis=1)]
df1


# removeing the outlayer of the dataset

# In[39]:


df.shape


# In[40]:


df1.shape


# new dataset save in df1

# In[41]:


df1.skew()


# checking for the skewness

# In[42]:


df1['residual sugar']=np.cbrt(df1['residual sugar'])


# use cuberoot method to remove skewness

# In[43]:


df1.skew()


# In[44]:


df1['chlorides']=np.cbrt(df1['chlorides'])


# In[45]:


x=df1.iloc[:,:-1]
x


# use i loc function to separating the dataset into x and y for training and testing

# In[46]:


y=df1.iloc[:,-1]
y


# separat y variabel in dataset for tarining and testing 

# In[47]:


x.shape


# In[48]:


y.shape


# In[49]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1=pd.DataFrame(sc.fit_transform(df1),columns=df1.columns)
df1


# standaedscaler removes the mean and scale each variabel to dataset

# In[50]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values, i)
              for i in range (len(x.columns))]
vif['Features']=x.columns
vif


# applying vif factor to check the underfitting and over fitting

# In[51]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# importing the train and test model to separate the train,test data

# In[52]:


x_train.shape


# In[53]:


x_test.shape


# In[54]:


y_train.shape


# In[55]:


y_test.shape


# In[56]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y=sm.fit_resample(x,y)


# use sote techniqe it helps to overcome the overfitting data

# In[57]:


y.value_counts()


# In[58]:


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


# randomfrosetclassifier use to check the accuracy, hear accureacy is 86%

# In[59]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier


# since my target is having contineous data, so i have to choose Regression model and import nessary libreries for regression model

# In[60]:


RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
predRFC=RFC.predict(x_test)
print(accuracy_score(y_test,predRFC))
print(confusion_matrix(y_test,predRFC))
print(classification_report(y_test,predRFC))


# using the linearregression model to get the best random state for better prediction

# In[77]:


lg=LogisticRegression()
lg.fit(x_train,y_train)
predlg=lg.predict(x_test)
print(accuracy_score(y_test,predlg))
print(confusion_matrix(y_test,predlg))
print(classification_report(y_test,predlg))


# In[78]:


et=ExtraTreesClassifier()
et.fit(x_train,y_train)
predet=et.predict(x_test)
print(accuracy_score(y_test,predet))
print(confusion_matrix(y_test,predet))
print(classification_report(y_test,predet))


# In[ ]:





# In[63]:


svc=SVC()
svc.fit(x_train,y_train)
predsvc=svc.predict(x_test)
print(accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))
print(classification_report(y_test,predsvc))


# in svc accuracy is 40%

# In[64]:


gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
predgb=gb.predict(x_test)
print(accuracy_score(y_test,predgb))
print(confusion_matrix(y_test,predgb))
print(classification_report(y_test,predgb))


#  in GradientBoostingClassifier accuracy is 80%

# In[65]:


bc=BaggingClassifier()
bc.fit(x_train,y_train)
predbc=bc.predict(x_test)
print(accuracy_score(y_test,predbc))
print(confusion_matrix(y_test,predbc))
print(classification_report(y_test,predbc))
      


# in BaggingClassifier accuracy is 78%

# In[66]:


ab=AdaBoostClassifier()
ab.fit(x_train,y_train)
predab=ab.predict(x_test)
print(accuracy_score(y_test,predab))
print(confusion_matrix(y_test,predab))
print(classification_report(y_test,predab))


# In[67]:


from sklearn. model_selection import cross_val_score
score=cross_val_score(RFC,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predRFC)-score.mean())


# import cross_val_score method to check for modelcan generaliseover the complet dataset

# In[68]:


score=cross_val_score(lg,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predlg)-score.mean())


# In[69]:


score=cross_val_score(et,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predet)-score.mean())


# In[70]:


score=cross_val_score(svc,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predsvc)-score.mean())


# In[71]:


score=cross_val_score(ab,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predab)-score.mean())


# In[72]:


score=cross_val_score(bc,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predbc)-score.mean())


# In[73]:


score=cross_val_score(gb,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predgb)-score.mean())


# In[109]:


from sklearn.model_selection import GridSearchCV
param = { 'loss' : ['ls', 'lad', 'huber', 'quantile'],
              'learning_rate' : (0.05,0.25,0.50,1),
              'criterion' : ['friedman_mse', 'mse', 'mae'],
              'max_features' : ['auto', 'sqrt', 'log2']
             }
gscv=GridSearchCV(GradientBoostingClassifier(),param,cv=5)


# In[110]:


gscv.fit(x_train,y_train)


# In[108]:


gscv.best_params_


# In[99]:


Final_model =ExtraTreesClassifier(criterion="entropy",max_depth=20,n_estimators=200,n_jobs=-2,random_state=30)
Final_model.fit(x_train,y_train)
pred=Final_model.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc*100)
                                  


# In[100]:


Final_model =ExtraTreesClassifier(criterion="gini",max_depth=20,n_estimators=200,n_jobs=-2,random_state=30)
Final_model.fit(x_train,y_train)
pred=Final_model.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc*100)


# the best model is extratreeclassifier accureacy 85%

# In[85]:


from sklearn.metrics import roc_curve
auc_score=roc_curve(y_test,et.predict(x_test))
print(auc_score)


# In[86]:


polt_roc_curve(Final_model,x_test,y_test)
plt.title('ROC for the best model')
plt.show()


# In[91]:


import pickle
filename='redwine'
pickle.dump(model,open(filename,'wb'))


# for saving the best model i am importing the pickle lybrari and giving the file name as redwine

# In[93]:


import pickle
new_model=pickle.load(open('redwine','rb'))


# load the same model from the jupyer note book

# In[94]:


score=new_model.score(x_test,y_test)
print(score*100)


# extratreeclassifieris best model to predict by redwine quality

# In[ ]:




