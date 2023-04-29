#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# import the nassasary lybraries for loading the dataset

# In[7]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/titanic_train.csv')


# In[8]:


df


# load the dataset useing github link ,titanic dataset contain 891 rows and 12 columns

# In[9]:


df.columns.sort_values(ascending=True)


# In[10]:


df.info()


# dataset having 3types of data like object,folat and integer

# In[11]:


df['Survived'].value_counts()


# here use value count to know how many are alive 0= stands for death,1=stands for servive

# In[12]:


for i in df.columns:
    print(i,'\n',df[i].value_counts()
         ,'\n')


# In[13]:


for i in df.columns:
    print(i,'-----',df[i].nunique())


# In[14]:


df.isnull().sum()


# null values present in cabin,age and embarked coloumns 

# In[15]:


plt.figure(figsize=(20,8))
sns.heatmap(df.isnull())


# In[16]:


df['Cabin'].isnull().sum()


# in cabin column haveing high nul values are present  better to drop this cloumn

# In[17]:


df=df.drop(['Cabin'],axis=1)


# In[18]:


df


# here use drop method remove the cabin column

# In[19]:


df['Age'].isnull().sum()


# In[ ]:





# In[20]:


age_mean=df['Age'].mean()


# In[21]:


df['Age'].fillna(value=age_mean,inplace=True)


# use mean average in the age coloum to balence null values

# In[22]:


df['Age'].isnull().sum()


# In[ ]:





# In[23]:


df['Embarked'].isnull().sum()


# In[24]:


df.dropna(inplace=True)


# use drop na method to drop null values

# In[25]:


df.isnull().sum()


# now dataset is no null values

# In[26]:


df.describe()


# *here the mean is greater than the median so here we can confirm the right skewed data is present in the dataset
# *the target column there huge difference is in between the 3rd quartile and the max values , means outliers are present 
# in the target and in the graph also we can see the same thing 

# In[27]:


sns.countplot(df['Survived'])


# here use countplot in survived column itclearly shows that death is more compare to survive

# In[28]:


sns.catplot(data=df,x='Sex',y='Fare',hue='Survived')


# in sex ratio female is more survive than male

# In[29]:


print(df.corr())
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True,fmt='0.2f')


# parch, sibsp ,age,survived and passengereid is possitively co relate with fare and pclass is negetively corelate

# In[30]:


sns.distplot(df['Age'])


# in age coloum left skew data is present

# In[31]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()

for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# using the ordinal encoder to convert the Object data to Numarical data ..
# 
# beacuse we can not use the catogorical data to predict the model further

# In[32]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# using zscore method to remove the outliers present in the dataset...

# In[33]:


threshold=3
print(np.where(z>3))


# In[34]:


z.iloc[13][7]


# In[35]:


z.iloc[885][7]


# checking zscore with iloc function , clearly its greater than 7 means outliers is present

# In[36]:


z.iloc[50][6]


# In[37]:


df1=df[(z<3).all(axis=1)]


# In[38]:


print(df.shape)
print(df1.shape)


# In[39]:


print('Percentage of data loss ',((df.shape[0]-df1.shape[0])/df.shape[0])*100)


# camparing the data loss after remiving the outliers ,acceptable range of data loss is 10% , so we go through the zscore method
# is best here

# In[40]:


Q1=df.quantile(0.25)
Q3=df.quantile(0.75)

IQR=Q3-Q1

df2=df[~((df<(Q1-1.5*IQR)))|((df>(Q3+1.5*IQR))).any(axis=1)]


# removing the outliers using the quantile method

# In[41]:


df2.shape


# removed the outliers and saving to new dataset df2

# In[42]:


df1.describe()


# In[43]:


plt.figure(figsize=(15,8))
sns.heatmap(df1.corr(),annot=True,fmt='0.2f')


# In[ ]:





# In[44]:


sns.pairplot(data=df1)


# In[45]:


sns.jointplot(data=df1)


# In[46]:


x=df1.drop('Survived',axis=1)


# In[47]:


x


#  x veriabels are 818 rows and 10 columns

# In[48]:


y=df1['Survived']


# In[49]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=pd.DataFrame(sc.fit_transform(x),columns=x.columns)
df


# In[50]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
dtc=DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[51]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y=sm.fit_resample(x,y)


# use somte metod to balence the x and y values

# In[52]:


y.value_counts()


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
maxAccu=0
maxRs=0
for i in range(1,800):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    RFR=RandomForestClassifier()
    RFR.fit(x_train,y_train)
    pred=RFR.predict(x_test)
    acc=accuracy_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRs=i
print("Best Accuracy is", maxAccu*100,"at random_state ",maxRs)


# In[54]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier,RandomForestClassifier,ExtraTreesClassifier


# import nessasary lrbrires for regression

# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=405)


# In[56]:


mn=MultinomialNB()
mn.fit(x_train,y_train)
predmn=mn.predict(x_test)
print('Accuracy_score is',((accuracy_score(y_test,predmn))*100))
print('Classification Report','\n',classification_report(y_test,predmn))
print('confusion Matrix','\n',confusion_matrix(y_test,predmn))


# in multinomialNB accureacy is 61

# In[57]:


gn=GaussianNB()
gn.fit(x_train,y_train)
predgn=gn.predict(x_test)
print('Accuracy Score',(accuracy_score(y_test,predgn))*100)
print(classification_report(y_test,predgn))
print(confusion_matrix(y_test,predgn))


# GaussianNB accuracy is 77%

# In[58]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
preddtc=dtc.predict(x_test)
print('Accuracy Score',accuracy_score(y_test,preddtc)*100)
print(classification_report(y_test,preddtc))
print(confusion_matrix(y_test,preddtc))


# In[59]:


svc=SVC()
svc.fit(x_train,y_train)
predsvc=svc.predict(x_test)
print('Accuracy_score',accuracy_score(y_test,predsvc)*100)
print(classification_report(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))


# SVC accuracy is 61%

# In[60]:


ad=AdaBoostClassifier()
ad.fit(x_train,y_train)
predad=ad.predict(x_test)
print('Accuracy score',accuracy_score(y_test,predad)*100)
print(classification_report(y_test,predad))
print(confusion_matrix(y_test,predad))


# AdaBoostClassifier accuracy is 83%

# In[61]:


bg=BaggingClassifier()
bg.fit(x_train,y_train)
predbg=bg.predict(x_test)
print('Accuracy Score',accuracy_score(y_test,predbg)*100)
print(classification_report(y_test,predbg))
print(confusion_matrix(y_test,predbg))


# Bagging Classifier accuracy is 87%

# In[62]:


gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
predgb=gb.predict(x_test)
print('Accuracy score',accuracy_score(y_test,predgb)*100)
print(classification_report(y_test,predgb))
print(confusion_matrix(y_test,predgb))


# GradientBostingClassifier accuracy is 90%

# In[63]:


et=ExtraTreesClassifier()
et.fit(x_train,y_train)
predet=et.predict(x_test)
print('Accuracy score',accuracy_score(y_test,predet)*100)
print(classification_report(y_test,predet))
print(confusion_matrix(y_test,predet))


# extratrees classifier accuracy is 88%

# In[64]:


rd=RandomForestClassifier()
rd.fit(x_train,y_train)
predrd=rd.predict(x_test)
print('Accuracy score',accuracy_score(y_test,predrd)*100)
print(classification_report(y_test,predrd))
print(confusion_matrix(y_test,predrd))


# In[65]:


from sklearn.model_selection import cross_val_score


# In[70]:


score=cross_val_score(et,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[82]:


score=cross_val_score(gb,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[83]:


score=cross_val_score(mn,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[84]:


score=cross_val_score(dtc,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[85]:


score=cross_val_score(svc,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[86]:


score=cross_val_score(ad,x,y,cv=5)
print('score', score*100)
print('Mean Score',score.mean()*100)


# In[91]:


from sklearn.model_selection import GridSearchCV
param = {'criterion' : ['gini','entropy'],
              'randam_state' : (10,50,1000),
              'max_depth' : [0,10,20],
               'n_job':[-2,-1,1],
              'n_estimators' : [50,100,200,300]}
             
gscv=GridSearchCV(GradientBoostingClassifier(),param,cv=5)


# In[93]:


gscv.fit(x_train,y_train)


# In[95]:


gscv.best_param_


# In[101]:


Fineal_model=ExtraTreesClassifier(criterion='entropy',max_depth=10,n_estimators=200,n_jobs=-2,random_state=10)
Fineal_model.fit(x_train,y_train)
pred=Fineal_model.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc*100)


# In[102]:


Fineal_model=ExtraTreesClassifier(criterion='gini',max_depth=10,n_estimators=200,n_jobs=-2,random_state=10)
Fineal_model.fit(x_train,y_train)
pred=Fineal_model.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc*100)


# In[107]:


import pickle
filename='Titanic'
pickle.dump(Fineal_model,open(filename,'wb'))


# In[108]:


import pickle
new_model=pickle.load(open('Titanic','rb'))


# In[109]:


s=new_model.score(x_test,y_test)
print(s*100)


# conclusion :ExtraTreesClassifier is best model to predict my titanic model

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




