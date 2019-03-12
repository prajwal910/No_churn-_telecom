#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter


# In[2]:


date_parser = lambda c: pd.to_datetime(c, format='%d/%m/%Y %H:%M:%S', errors='coerce')
data = pd.read_csv('ITSM_data.csv', parse_dates=['Open_Time','Reopen_Time','Close_Time','Resolved_Time'],)


# In[3]:


data.head(2)


# In[4]:


data.Close_Time[3]


# In[93]:


data.info()


# In[6]:


Counter(data.Category).most_common(5)


# In[7]:


data.loc[30:35,['Open_Time','Reopen_Time','Close_Time','Resolved_Time','Handle_Time_hrs','Priority']]


# In[8]:


data[data.Priority.isnull()].Incident_ID.count()


# In[9]:


data[data.Impact.isnull()].Incident_ID.count()


# In[10]:


data1 = data[data.Priority.isnull()==False]


# In[11]:


data1.shape


# In[12]:


#SMOTE = Synthetic Minority Oversampling Technique


# In[13]:


Counter(data1.Priority)


# In[14]:


#find Missing Priorities.  For you we need to find Missing Impact then using Impact and Urgency, derive Priority using Matrix


# In[15]:


import datetime as dt


# In[16]:


duration = []
for index,row in data.iterrows():
    duration.append(row.Close_Time-row.Open_Time)


# In[17]:


data['duration'] = duration


# In[18]:


data.duration.describe()


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
data[data['duration']> dt.timedelta(days=30)].duration.describe()


# In[20]:


data[data['duration']< dt.timedelta(days=30)].duration.describe()


# #### Ticket statuses
# It's broken (Open, reopened and in-progress) <br>
# I think I've fixed it (Resolved)<br>
# Yes, it's fixed, never look at it again (Closed)

# In[21]:


data[data.Close_Time<data.Open_Time].loc[:,['Close_Time','Open_Time','Resolved_Time']]


# In[22]:


#Close Time can't be before Open Time. This takes care of Invalid close times
datav = data[data.Close_Time>data.Open_Time]


# In[23]:


datav.info()


# In[24]:


datav[datav['duration']> dt.timedelta(days=30)].duration.describe()


# In[25]:


Counter(datav.CI_Cat).most_common(10)


# In[26]:


Counter(datav.CI_Subcat).most_common(10)


# In[27]:


Counter(datav.WBS).most_common(10)


# In[28]:


Counter(datav.Category).most_common(5)


# In[29]:


selected = data.loc[:,['CI_Subcat','WBS','Category','Priority']]
selected.dropna(inplace=True)
Counter(selected.Priority)


# In[30]:


selected.info()


# In[31]:


selected = selected[selected.Priority!=1]


# In[32]:


selected.info()


# In[88]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
selected.CI_Subcat = enc.fit_transform(X.CI_Subcat)
selected.Category = enc.fit_transform(X.Category)
selected.WBS = selected.WBS.str[4:].astype('int')


# In[89]:


Counter(selected.Category)


# In[90]:


X = selected.loc[:,['CI_Subcat','WBS','Category']]
y = selected.Priority


# In[37]:


pd.DataFrame(enc.classes_)


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=13)


# In[39]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[40]:


model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
accuracy_score(y_test,y_pred)


# In[41]:


model=RandomForestClassifier(max_depth=15)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
accuracy_score(y_test,y_pred)


# In[42]:


from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
accuracy_score(y_test,y_pred)


# In[ ]:


grid_model.best_params_


# In[ ]:


from sklearn.svm import SVC

model=SVC(kernel='rbf',C=10,gamma=0.1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
pd.crosstab(y_test,y_pred)


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = { 'kernel':['rbf'],
                'C':[1,100],
                'gamma' : [0.01,1.0]}
grid_model = GridSearchCV(SVC(),parameters)
grid_model.fit(X_train, y_train)
grid_model.best_score_


# In[55]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))


# In[56]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_sample(X_train,y_train)


# In[75]:


model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test.values)
print(accuracy_score(y_test,y_pred))
pd.crosstab(y_test,y_pred)


# In[76]:


print(classification_report(y_test,y_pred))


# In[84]:


Counter(y)


# In[87]:


selected.head()


# In[94]:


from sklearn.externals import joblib
joblib.dump(model,'ML_perdict_high_priority.ml')


# In[ ]:




