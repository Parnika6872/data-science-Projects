#!/usr/bin/env python
# coding: utf-8

# 

# ### Data Dictionary

# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | AGE	          proportion of owner-occupied units built prior to 1940 |
# | DIS             weighted distances to five Boston employment centres |
# | RAD	          index of accessibility to radial highways |
# | TAX         	  full-value property-tax rate per 10,000  |
# | PTRATIO 	      pupil-teacher ratio by town|
# | LSTAT           lower status of the population|
# | MEDV            Median value of owner-occupied homes in 1000s|
# |CRIM	          per capita crime rate by town|
# | ZN	          proportion of residential land zoned for lots over 25,000 sq.ft.|
# | INDUS           proportion of non-retail business acres per town.	|
# | CHAS            Charles River dummy variable (1 if tract bounds river; 0 otherwise)	|
# | NOX	          nitric oxides concentration (parts per 10 million)|
# | RM	          average number of rooms per dwelling|
# | AGE	          proportion of owner-occupied units built prior to 1940 |

# 

# 

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime
import scipy.stats

get_ipython().run_line_magic('matplotlib', 'inline')
#sets the default autosave frequency in seconds
get_ipython().run_line_magic('autosave', '60')
sns.set_style('dark')
sns.set(font_scale=1.2)

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
pd.set_option('display.width', 1000)

np.random.seed(0)
np.set_printoptions(suppress=True)


# In[2]:


df = pd.read_csv("boston_housing.csv")


# In[3]:


df


# ### Exploratory Data Analysis

# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# ### Data Visualization

# In[7]:


df.hist(bins=50, figsize=(20,10))
plt.suptitle('Feature Distribution', x=0.5, y=1.02, ha='center', fontsize='large')
plt.tight_layout()
plt.show()


# In[8]:


plt.figure(figsize=(20,20))
plt.suptitle('Pairplots of features', x=0.5, y=1.02, ha='center', fontsize='large')
sns.pairplot(df.sample(250))

plt.show()


# ### Task 4: Generate Descriptive Statistics and Visualizations

# In[9]:


#For the "Median value of owner-occupied homes" provide a boxplot
plt.figure(figsize=(10,5))
sns.boxplot(x=df.MEDV)
plt.title("Boxplot for MEDV")
plt.show()


# Note: Outliers after third quartile.

# In[10]:


#Provide a histogram for the Charles river variable
plt.figure(figsize=(10,5))
sns.distplot(a=df.CHAS,bins=10, kde=False)
plt.title("Histogram for Charles river")
plt.show()


# Note: Majority tracts don't bound Charles River

# In[11]:


#Provide a boxplot for the MEDV variable vs the AGE variable. 
#(Discretize the age variable into three groups of 35 years and younger, 
#between 35 and 70 years and 70 years and older)

df.loc[(df["AGE"] <= 35),'age_group'] = '35 years and younger'
df.loc[(df["AGE"] > 35) & (df["AGE"]<70),'age_group'] = 'between 35 and 70 years'
df.loc[(df["AGE"] >= 70),'age_group'] = '70 years and older'


# In[12]:


df


# In[13]:


plt.figure(figsize=(10,5))
sns.boxplot(x=df.MEDV, y=df.age_group, data=df)
plt.title("Boxplot for the MEDV variable vs the AGE variable")
plt.show()


# Note: 35 years or younger group pays the highest median house price while above 70s are shifting to cheaper houses

# In[14]:


#Provide a scatter plot to show the relationship between Nitric oxide concentrations and 
#the proportion of non-retail business acres per town. What can you say about the relationship?

plt.figure(figsize=(10,5))
sns.scatterplot(x=df.NOX, y=df.INDUS, data=df)
plt.title("Relationship between NOX and INDUS")
plt.show()


# Note: There seems to be a linear relationship till NOX=0.6

# In[15]:


#Create a histogram for the pupil to teacher ratio variable
plt.figure(figsize=(10,5))
sns.distplot(a=df.PTRATIO,bins=10, kde=False)
plt.title("Histogram for the pupil to teacher ratio variable")
plt.show()


# Note: Pupil to teacher ratio is highest at 20-21 range.

# ### Task 5: Use the appropriate tests to answer the questions provided

# In[16]:


df


# #### Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)

# Null Hypothesis($H_0$): Both average MEDV are the same
# 
# Alternative Hypothesis($H_1$): Both average MEDV are NOT the same

# In[17]:


df["CHAS"].value_counts()


# In[18]:


a = df[df["CHAS"] == 0]["MEDV"]
a


# In[19]:


b = df[df["CHAS"] == 1]["MEDV"]
b


# In[20]:


scipy.stats.ttest_ind(a,b,axis=0,equal_var=True)


# Since p-value more than alpha value of 0.05, we failed to reject null hypothesis since there is NO statistical significance.

# #### Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)

# In[21]:


df["AGE"].value_counts()


# In[22]:


df.loc[(df["AGE"] <= 35),'age_group'] = '35 years and younger'
df.loc[(df["AGE"] > 35) & (df["AGE"]<70),'age_group'] = 'between 35 and 70 years'
df.loc[(df["AGE"] >= 70),'age_group'] = '70 years and older'


# In[23]:


df


# State the hypothesis
# 
# -   $H_0: µ\_1 = µ\_2 = µ\_3$ (the three population means are equal)
# -   $H_1:$ At least one of the means differ
# 

# In[24]:


low = df[df["age_group"] == '35 years and younger']["MEDV"]
mid = df[df["age_group"] == 'between 35 and 70 years']["MEDV"]
high = df[df["age_group"] == '70 years and older']["MEDV"]


# In[25]:


f_stats, p_value = scipy.stats.f_oneway(low,mid,high,axis=0)


# In[26]:


print("F-Statistic={0}, P-value={1}".format(f_stats,p_value))


# Since p-value more than alpha value of 0.05, we failed to reject null hypothesis since there is NO statistical significance.

# #### Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)

# State the hypothesis
# 
# -   $H_0:$ NOX is not correlated with INDUS
# 
# -   $H_1:$ NOX is correlated with INDUS
# 

# In[27]:


pearson,p_value = scipy.stats.pearsonr(df["NOX"],df["INDUS"])


# In[28]:


print("Pearson Coefficient value={0}, P-value={1}".format(pearson,p_value))


# Since the p-value (Sig. (2-tailed) < 0.05, we reject the Null hypothesis and conclude that there exists a relationship between Nitric Oxide and non-retail business acres per town.

# #### What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)

# State Hypothesis
# 
# Null Hypothesis: weighted distances to five Boston employment centres are not related to median value
# 
# Alternative Hypothesis: weighted distances to five Boston employment centres are related to median value

# In[29]:


df.columns


# In[30]:


y = df['MEDV']
x = df['DIS']


# In[31]:


x = sm.add_constant(x)


# In[32]:


results = sm.OLS(y,x).fit()


# In[33]:


results.summary()


# In[34]:


np.sqrt(0.062)  ##Pearson Coeffiecent valuea


# The square root of R-squared is 0.25, which implies weak correlation between both features

# ### Correlation

# In[35]:


df.corr()


# In[36]:


plt.figure(figsize=(16,9))
sns.heatmap(df.corr(),cmap="coolwarm",annot=True,fmt='.2f',linewidths=2, cbar=False)
plt.show()

