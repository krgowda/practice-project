#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# importing all the nessasary lybraries for loading dataset and analysing dataset

# In[34]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset3/main/glass.csv')
df


# loading dataset with the help of raw data link,save the data as df and dataset contains 213 rows and 11 columns

# In[35]:


df.dtypes


# two types of data are present
# 1= integer
# 2=float
# dtype is object

# In[36]:


df.columns


# in this data columns names is number its convert in given name

# In[37]:


df.rename(columns={'1':'id_number','1.52101':'ri','13.64':'na','4.49':'mg','1.10':'al','71.78':'si','0.06':'k','8.75':'ca','0.00':'ba','0.00.1':'fe','1.1':'type of glass'},inplace=True)


# useing rename method to change columns names

# In[38]:


df


# In[39]:


df.isnull().sum()


# no null values present in dataset

# In[40]:


sns.heatmap(df.isnull())


# use heatmap  to check the null values in dataset their no null values

# In[41]:


print(df.describe())
plt.figure(figsize=(15,8))
sns.heatmap(df.describe(),annot=True,fmt='0.2f',linecolor='black',linewidth=0.2,cmap='Blues_r')


# In[42]:


df.corr()
df


# In[43]:


df.describe()


# hear both diffrences are seen some columns are mean is greater than meadin inthis right skew data are presnt
# and median greater than mean in this data left skew data are present
# huge difference in between the 3rd quartile and maximum values ,it means outliers present in target columns

# In[44]:


df['id_number'].plot.box()


# In[45]:


df['ri'].plot.box()


# In[46]:


df['na'].plot.box()


# In[47]:


df['mg'].plot.box()


# In[48]:


df['al'].plot.box()


# In[49]:


df['si'].plot.box()


# In[50]:


import matplotlib.pyplot as plt
df.plot(kind='box',subplots=True,layout=(2,6),figsize=(10,10))


# use boxplot to find the outliers here ri,na,al,si,k,ca,ba,fe and type of glass also 

# In[51]:


print(df.corr())
sns.heatmap(df.corr(),cmap='Blues_r',linecolor='black',linewidth=0.30,fmt='0.2f',annot=True)
print('\n')


# mg, is negetively corelate with target

# In[52]:


import numpy as np
from scipy.stats import zscore
z=np.abs(zscore(df))
z


# use zscore method to remove outliers

# In[53]:


threshold=3
print(np.where(z>3))


# In[54]:


df1=df[(z<3).all(axis=1)]


# In[55]:


df1.shape


# data saved in df1

# In[56]:


df.shape


# In[58]:


x=df1.iloc[:,:-1]
x


# separet data into x train &test data

# In[59]:


y=df1.iloc[:,-1]
y


# y values separet into train test and  data

# In[60]:


df1.skew()


# check for outliers, most outlier are present in ba columns ,i will drop this column

# In[64]:


df1['ba']=np.cbrt(df1['ba'])


# In[65]:


df1.skew()


# In[66]:


df1['ba']=np.cbrt(df1['ba'])


# In[67]:


df1.skew()


# In[68]:


x.shape


# In[69]:


y.shape


# In[70]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1=pd.DataFrame(sc.fit_transform(df1),columns=df1.columns)
df1


# use the standaredscaler method to standard the data

# In[71]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values, i)
              for i in range (len(x.columns))]
vif['Features']=x.columns
vif


# In[72]:


y.value_counts()


# y values are imblence datawill balence to find the accuracy of the target

# In[73]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y=sm.fit_resample(x,y)


# In[74]:


y.value_counts()


# use smote method to balence the data

# In[75]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)


# importing the train and test model to separate the train,test data

# In[76]:


x_train.shape


# In[77]:


x_test.shape


# In[78]:


y_train.shape


# In[79]:


y_test.shape


# In[80]:


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


# In[81]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier


# import nessasary libraries for classifier

# In[82]:


RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
predRFC=RFC.predict(x_test)
print(accuracy_score(y_test,predRFC))
print(confusion_matrix(y_test,predRFC))
print(classification_report(y_test,predRFC))


# in RandamForestClassifier is 99%

# In[83]:


lg=LogisticRegression()
lg.fit(x_train,y_train)
predlg=lg.predict(x_test)
print(accuracy_score(y_test,predlg))
print(confusion_matrix(y_test,predlg))
print(classification_report(y_test,predlg))


# LogisticRegression is 100%

# In[84]:


et=ExtraTreesClassifier()
et.fit(x_train,y_train)
predet=et.predict(x_test)
print(accuracy_score(y_test,predet))
print(confusion_matrix(y_test,predet))
print(classification_report(y_test,predet))


# ExtraTreeClassifier is 100%

# In[85]:


svc=SVC()
svc.fit(x_train,y_train)
predsvc=svc.predict(x_test)
print(accuracy_score(y_test,predsvc))
print(confusion_matrix(y_test,predsvc))
print(classification_report(y_test,predsvc))


# svc accuracy is 71%

# In[86]:


gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
predgb=gb.predict(x_test)
print(accuracy_score(y_test,predgb))
print(confusion_matrix(y_test,predgb))
print(classification_report(y_test,predgb))


# GradientBoostingClassifier  accuracy is 98%

# In[87]:


bc=BaggingClassifier()
bc.fit(x_train,y_train)
predbc=bc.predict(x_test)
print(accuracy_score(y_test,predbc))
print(confusion_matrix(y_test,predbc))
print(classification_report(y_test,predbc))
      


# In[88]:


ab=AdaBoostClassifier()
ab.fit(x_train,y_train)
predab=ab.predict(x_test)
print(accuracy_score(y_test,predab))
print(confusion_matrix(y_test,predab))
print(classification_report(y_test,predab))


# In[89]:


from sklearn. model_selection import cross_val_score
score=cross_val_score(RFC,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predRFC)-score.mean())
score=cross_val_score(lg,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predlg)-score.mean())
score=cross_val_score(et,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predet)-score.mean())
score=cross_val_score(svc,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predsvc)-score.mean())
score=cross_val_score(ab,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predab)-score.mean())
score=cross_val_score(gb,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predgb)-score.mean())
score=cross_val_score(bc,x,y)
print(score)
print(score.mean())
print('Diffrence between Accuracy score and cross validation score is -',accuracy_score(y_test,predbc)-score.mean())


# In[90]:


from sklearn.model_selection import GridSearchCV


# In[91]:


Final_model =ExtraTreesClassifier(criterion="entropy",max_depth=20,n_estimators=200,n_jobs=-2,random_state=30)
Final_model.fit(x_train,y_train)
pred=Final_model.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc*100)
                                  


# ExtraTreeClassifier  model is the best accuracy , 

# In[93]:


import pickle
filename='Glass_'
pickle.dump(Final_model,open(filename,'wb'))


# In[94]:


import pickle
new_model=pickle.load(open('Glass_','rb'))


# In[95]:


score=new_model.score(x_test,y_test)
print(score*100)


# In[ ]:





# In[ ]:





# In[ ]:




