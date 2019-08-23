
# coding: utf-8

# In[ ]:


# Pandas and numpy for data manipulation
import numpy as np
import pandas as pd

# Matplotlib visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for visualization
import seaborn as sns

# Splitting data into training and testing
from sklearn.linear_model import LinearRegression


# In[ ]:


# Data Cleaning and Formatting

# Load in the Data and Examine

# Read in loan data into a dataframe
try:
    dff = pd.read_csv("D:/train.csv")
except:
    print("The loan dataset could not be loaded. Is the dataset missing?")


# In[ ]:


# Make a copy of dataframe

df=dff.copy()

# Display dataframe
df


# In[ ]:


#To drop Nan values from the dataframe
df.dropna(axis=0,inplace=True)

#To remove '3+' value as it is not a proper value
df=df[df.Dependents != '3+']

#Reseting the index numerisation
df=df.reset_index()
df.head()


# In[ ]:


#To Evaluate the number of rows and columns in the DataFrame df after removing Nan and inpropiate values
df.shape


# In[ ]:


#To Print the column names
df.columns


# In[ ]:


# Statistics for each column
df.describe()


# In[ ]:


# See the column data types and non-missing values
df.info()


# In[ ]:


df.drop(labels=['index', 'Application_ID'], axis=1, inplace=True)

# These two features are only for identification, so removing them


# In[ ]:


#Convert categorical variable into dummy/indicator variables

gender=pd.get_dummies(df["Gender"],drop_first=True)
self_employed=pd.get_dummies(df['Self_Employed'],drop_first=True)
education=pd.get_dummies(df['Education'],drop_first=True)
property_area=pd.get_dummies(df['Property_Area'],drop_first=True)
loan_status=pd.get_dummies(df['Loan_Status'],drop_first=True)
df.drop(['Loan_Status','Property_Area','Education','Self_Employed',"Gender"],axis=1,inplace=True)
df=pd.concat([df,loan_status,property_area,education,self_employed,gender],axis=1)
df=df.rename(columns={'Y': 'Loan_Status','M':'Male','Yes':'Self_Employed'})

#Changing Married column seperately to avoid confusion with Self_Employed

married=pd.get_dummies(df['Married'],drop_first=True)
df.drop(['Married'],axis=1,inplace=True)
df=pd.concat([df,married],axis=1)
df=df.rename(columns={'Yes':'Married'})
df.head()


# In[ ]:


# See the column data types and non-missing values after changing data type
df.info()


# In[ ]:


#Correlation of Loan_status with other attributes
df.corr().iloc[:,5]


# In[ ]:


#Plotting a HeatMap of correlation of given attributes

matrix = df.corr()
mask = np.zeros_like(matrix)
mask[np.triu_indices_from(mask)] = True
ax = plt.subplots(figsize=(22, 11))

#Removing mirror image
with sns.axes_style("white"):
    ax=sns.heatmap(matrix, mask=mask, vmax=1, square=True, annot=True, cmap="YlGnBu",cbar_kws={"shrink":.5})
    plt.show()


# ##### It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount as compared to that of High Loan Amount which supports our hypothesis in which we considered that the chances of loan approval will be high when the loan amount is less.

# In[ ]:


# Normalize can be set to True to print proportions instead of number 

# To check the proportion of loan approval

df['Loan_Status'].value_counts(normalize=True)


# # TO CHECK STATS OF ATTRIBUTES WITH LOAN APPROVAL

# In[ ]:


# Make a copy of dataframe for graphical representation

ls=dff.copy()
ls.dropna(axis=0,inplace=True)
ls=ls.reset_index()
ls=ls.drop(columns=['index'])
ls.head()


# In[ ]:


Gender=pd.crosstab(ls.Gender,ls.Loan_Status)
Married=pd.crosstab(ls.Married,ls.Loan_Status)
Dependents=pd.crosstab(ls.Dependents,ls.Loan_Status)
Education=pd.crosstab(ls.Education,ls.Loan_Status)
Self_Employed=pd.crosstab(ls.Self_Employed,ls.Loan_Status)
CreditHistory=pd.crosstab(ls.Credit_History,ls.Loan_Status)
Property_Area=pd.crosstab(ls.Property_Area,ls.Loan_Status)


# # GRAPHS REPRESENTING COUNTS OF MEMBER OF ATTRIBUTES WITH LOAN APPROVAL

# ### Categorical features

# In[ ]:


Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.title("Married Vs Loan Status")
plt.show()


# * This graph shows Marital status is important attribute of getting loan

# In[ ]:


Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.title("Gender Vs Loan Status")
plt.show()


# * This graph shows Male applicants has more chances of getting loan

# In[ ]:


Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.title("Self Employed Vs Loan Status")
plt.show()


# * This graph shows Self_Employed is not important attribute of getting loan

# In[ ]:


CreditHistory.div(CreditHistory.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.title("Credit History Vs Loan Status")
plt.show()


# * This graph shows people with no credit history are not getting loan

# ### Ordinal features

# In[ ]:


Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Dependents Vs Loan Status")
plt.show()


# * This graph shows Number of dependents are not proportional to chances of loan approval

# In[ ]:


Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.title("Education Vs Loan Status")
plt.show()


# * This graph shows Educational status is not important attribute of getting loan

# In[ ]:


Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.title("Property Area Vs Loan Status")
plt.show()


# * This graph shows property needs to be in urban or semi-urban area to get loan approval

# ### Numerical features

# In[ ]:


plt.subplot(121)
sns.distplot(ls['ApplicantIncome']);

plt.subplot(122)
ls['ApplicantIncome'].plot.box(figsize=(16,5))

plt.show()


# ##### It can be inferred that Applicant income does not affect the chances of loan approval which contradicts our hypothesis in which we assumed that if the applicant income is high the chances of loan approval will also be high.

# In[ ]:


plt.subplot(121)
sns.distplot(ls['CoapplicantIncome']);

plt.subplot(122)
ls['CoapplicantIncome'].plot.box(figsize=(16,5))

plt.show()


# ###### It shows that if coapplicant’s income is less the chances of loan approval are high. But this does not look right. The possible reason behind this may be that most of the applicants don’t have any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.

# In[ ]:


plt.subplot(121)
sns.distplot(ls['LoanAmount']);

plt.subplot(122)
ls['LoanAmount'].plot.box(figsize=(16,5))

plt.show()


# ##### It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount as compared to that of High Loan Amount which supports our hypothesis in which we considered that the chances of loan approval will be high when the loan amount is less.

# # Model Evaluation

# In[8]:


# Logistic Regression

lm=LinearRegression()
X=df.drop("Loan_Status",axis=1)
lm.fit(X,df.Loan_Status)
print("estimated coefficient:",lm.coef_)
print("no of coefficient:",len(lm.coef_))
print("estimated intercept:",lm.intercept_)

