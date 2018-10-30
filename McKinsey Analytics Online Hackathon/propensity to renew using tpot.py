
# coding: utf-8

# In[1]:



# Import required libraries# Impor 
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np


# In[2]:



# Load the data# Load  
train_df = pd.read_csv('./train.csv')
train_df.head(5)


# In[3]:


train_df.groupby('residence_area_type').renewal.value_counts()


# In[4]:


train_df.groupby(['sourcing_channel','perc_premium_paid_by_cash_credit']).renewal.value_counts()


# In[5]:


id = pd.crosstab([train_df.sourcing_channel, train_df.perc_premium_paid_by_cash_credit], train_df.renewal.astype(float))
id.div(id.sum(1).astype(float), 0)


# In[6]:


#Data Munging

train_df.rename(columns={'renewal': 'class'}, inplace=True)


# In[7]:


train_df.dtypes


# In[8]:


for cat in ['sourcing_channel', 'residence_area_type']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, train_df[cat].unique().size))


# In[9]:


for cat in ['sourcing_channel', 'residence_area_type']:
    print("Levels for catgeory '{0}': {1}".format(cat, train_df[cat].unique()))


# In[10]:


train_df['sourcing_channel'] = train_df['sourcing_channel'].map({'A':0,'B':1,'C':2,'D':3})
train_df['residence_area_type'] = train_df['residence_area_type'].map({'Rural':0,'Urban':1})


# In[11]:


train_df = train_df.fillna(-999)
pd.isnull(train_df).any()


# In[12]:


# #Since Name and Ticket have so many levels, we drop them from our analysis for the sake of simplicity. For Cabin, we encode the levels as digits using Scikit-learn's MultiLabelBinarizer and treat them as new features.


# fromfrom  sklearn.preprocessingsklearn.  import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])

# CabinTrans

# titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)

# assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done

# #We then add the encoded features to form the final dataset to be used with TPOT.

# titanic_new = np.hstack((titanic_new.values,CabinTrans))


# In[13]:


np.isnan(train_df).any()


# In[14]:


train_df_n = train_df.loc[:, train_df.columns !='class']
target = train_df.loc[:, train_df.columns =='class']

train_matrix = train_df_n.as_matrix()

train_matrix[0].size


# In[15]:


renewal_class = target.as_matrix()


# In[16]:


training_indices, validation_indices = training_indices, testing_indices = train_test_split(train_df.index, stratify = renewal_class, train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size


# In[17]:


tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)
tpot.fit(train_matrix[training_indices], renewal_class[training_indices])


# In[18]:


tpot.score(train_matrix[validation_indices], train_df.loc[validation_indices, 'class'].values)


# In[19]:


tpot.export('tpot_prem_renew_pipeline.py')


# In[20]:


#Make predictions on the submission data


# Read in the submission dataset# Read i 
test = pd.read_csv('./test.csv')
test.describe()
test['id'].head()


# In[21]:


pd.isnull(test).any()


# In[22]:


for var in ['Count_3-6_months_late']: #,'Name','Ticket']:
    new = list(set(test[var]) - set(train_df[var]))
    test.ix[test[var].isin(new), var] = -999


# In[23]:


test['sourcing_channel'] = test['sourcing_channel'].map({'A':0,'B':1,'C':2,'D':3})
test['residence_area_type'] = test['residence_area_type'].map({'Rural':0,'Urban':1})


# In[24]:


test = test.fillna(-999)
pd.isnull(test).any()


# In[25]:



# fromfrom  sklearn.preprocessingsklearn.  import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# SubCabinTrans = mlb.fit([{str(val)} for val in titanic['Cabin'].values]).transform([{str(val)} for val in titanic_sub['Cabin'].values])
# titanic_sub = titanic_sub.drop(['Name','Ticket','Cabin'], axis=1)


# # Form the new submission data set# Form t 
# titanic_sub_new = np.hstack((titanic_sub.values,SubCabinTrans))

# np.any(np.isnan(titanic_sub_new))

# # Ensure equal number of features in both the final training and submission dataset
# assert (titanic_new.shape[1] == titanic_sub_new.shape[1]), "Not Equal"

test.head()


# In[56]:


# Generate the predictions
submission = tpot.predict(test)


# In[ ]:



# Create the submission file# Create 

import math

assumed_pbenchmark = 0.5
assumed_incentive_for_benchmark = 1650

effort_in_hrs = 10*(1-math.exp(-assumed_incentive_for_benchmark/400))

perc_improv_in_renew_prob = 20*(1-math.exp(-effort_in_hrs/5))


final = pd.DataFrame({'id': test['id'], 'renewal': submission })

#, 
#                       'incentive': 
#                                   if submission = 1: 
#                                       (assumed_incentive_for_benchmark*submission)/assumed_pbenchmark 
#                                   else: 
#                                       assumed_incentive_for_benchmark 

# # 'revenue': ((submission+(perc_improv_in_renew_prob*submission))*test['premium']-assumed_incentive_for_benchmark*submission)/assumed_pbenchmark 

# #final

final.to_csv('submission_03.csv', index = False)

