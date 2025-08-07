# %% [markdown]
# ## Final Project : Classification
# 
# This is the benchmark solution for the **final project : classification**. In this notebook:
# 
# 1. We will first explore the dataset provided
# 2. We will create models to predict whether a client will subscribe to a term deposit. 
# 3. We will also make predictions for the clients in the test set which you can submit in the solution_checker.xlsx file to generate accuracy score. 
# 
# Let's start by importing the libraries that we will be using.

# %%
# importing libraries
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt 
import seaborn as sn                   # For plotting graphs%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

# %%
# loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# %% [markdown]
# Let's check the features present in our data and then we will look at their data types.

# %%
train.columns

# %%
test.columns

# %% [markdown]
# It can be infered that **subscribed** is the target variable as it is not present in the test dataset. Let's look at the shape of the dataset.

# %%
train.shape, test.shape

# %% [markdown]
# We have 17 independent variables and 1 target variable, i.e. subscribed in the train dataset. We have similar features in the test dataset as the train dataset except the subscribed. We will predict the subscribed with the help of model built using the train data.
# 
# Next, let's look at how many categorical and numerical variables are there in our dataset. We will look at their data types.

# %%
# Print data types for each variable
train.dtypes

# %% [markdown]
# We can see there are two format of data types:
# 
# 1. **object**: Object format means variables are categorical. Categorical variables in our dataset are: job, marital, education, default, housing, loan, contact, month, poutcome, subscribed
# 
# 2. **int64**: It represents the integer variables. Integer variables in our dataset are: ID, age, balance, day, duration, campaign, pdays, previous

# %%
#printing first five rows of the dataset
train.head()

# %% [markdown]
# ## Univariate Analysis

# %% [markdown]
# Now Let's look at the distribution of our target variable, i.e. subscribed. As it is a categorical variable, let us look at its frequency table, percentage distribution and bar plot.

# %%
train['subscribed'].value_counts()

# %%
# Normalize can be set to True to print proportions instead of number 
train['subscribed'].value_counts(normalize=True)

# %%
# plotting the bar plot of frequencies
train['subscribed'].value_counts().plot.bar()

# %% [markdown]
# So, 3715 users out of total 31647 have subscribed which is around 12%. Let's now explore the variables to have a better understanding of the dataset. We will first explore the variables individually using univariate analysis, then we will look at the relation between various independent variables and the target variable. We will also look at the correlation plot to see which variables affects the target variable most.
# 
# Let's first look  at the distribution of age variable to see how many people belongs to a particular age group.

# %%
sn.distplot(train["age"])

# %% [markdown]
# We can infer that most of the clients fall in the age group between 20-60. Now let's look at what are the different types of jobs of the clients. As job is a categorical variable, we will look at its frequency table

# %%
train['job'].value_counts().plot.bar()

# %% [markdown]
# We see that most of the clients belongs to blue-collar job and the students are least in number as students generally do not take a term deposit. Let's also look at how many clients have default history.

# %%
train['default'].value_counts().plot.bar()

# %% [markdown]
# More than 90% of the clients have no default history. Now we will explore these variables against the target variable using bivariate analysis. We will make use of scatter plots for continuous or numeric variables and crosstabs for the categorical variables. Let's start with job and subscribed variable.

# %% [markdown]
# ## Bivariate Analysis

# %%
print(pd.crosstab(train['job'],train['subscribed']))

job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')

# %% [markdown]
# From the above graph we can infer that students and retired people have higher chances of subscribing to a term deposit, which is surprising as students generally do not subscribe to a term deposit. The possible reason is that the number of students in the dataset is less and comparatively to other job types, more students have subscribed to a term deposit.
# 
# Next, let's explore the default variable against the subscribed variable.

# %%
print(pd.crosstab(train['default'],train['subscribed']))

default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')

# %% [markdown]
# We can infer that clients having no previous default have slightly higher chances of subscribing to a term loan as compared to the clients who have previous default history.
# 
# Let's now look at how correlated our numerical variables are. We will see the correlation between each of these variables and the variable which have high negative or positive values are correlated. By this we can get an overview of the variables which might affect our target variable. We will convert our target variable into numeric values first.

# %%
train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)

# %%
corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")

# %% [markdown]
# We can infer that duration of the call is highly correlated with the target variable. This can be verified as well. As the duration of the call is more, there are higher chances that the client is showing interest in the term deposit and hence there are higher chances that the client will subscribe to term deposit.
# 
# Next we will look for any missing values in the dataset.

# %%
train.isnull().sum()

# %% [markdown]
# There are no missing values in the train dataset. 
# 
# Next, we will start to build our predictive model to predict whether a client will subscribe to a term deposit or not.
# 
# As the sklearn models takes only numerical input, we will convert the categorical variables into numerical values using dummies. We will remove the ID variables as they are unique values and then apply dummies. We will also remove the target variable and keep it in a separate variable.

# %% [markdown]
# ## Model Building

# %%
target = train['subscribed']
train = train.drop('subscribed',1)

# %%
# applying dummies on the train dataset
train = pd.get_dummies(train)

# %% [markdown]
# Now, its time to build our model. We will split the train data into training and validation set so that we will be able to validate the results of our model on the validation set. We will keep 20% data as validation set and rest as the training set. 

# %%
from sklearn.model_selection import train_test_split

# %%
# splitting into train and validation with 20% data in validation set and 80% data in train set.
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)

# %% [markdown]
# Now our data is ready. Its time to build our model and check its performance. Logistic regression is used for classification problems and as it is a classification problem let's first build a Logistic Regression model.

# %% [markdown]
# ## Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

# %%
# defining the logistic regression model
lreg = LogisticRegression()

# %%
# fitting the model on  X_train and y_train
lreg.fit(X_train,y_train)

# %%
# making prediction on the validation set
prediction = lreg.predict(X_val)

# %% [markdown]
# Now we will evaluate how accurate our predictions are. As the evaluation metric for this problem is accuracy, let's calculate the accuracy on validation set.

# %%
from sklearn.metrics import accuracy_score

# %%
# calculating the accuracy score
accuracy_score(y_val, prediction)

# %% [markdown]
# We got an accuracy score of around 90% on the validation dataset. Logistic regression has a linear decision boundary. What if our data have non linearity? We need a model that can capture this non linearity. 
# 
# Let's try decision tree algorithm now to check if we get better accuracy with that. 

# %% [markdown]
# ## Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
# defining the decision tree model with depth of 4, you can tune it further to improve the accuracy score
clf = DecisionTreeClassifier(max_depth=4, random_state=0)

# %%
# fitting the decision tree model
clf.fit(X_train,y_train)

# %%
# making prediction on the validation set
predict = clf.predict(X_val)

# %%
# calculating the accuracy score
accuracy_score(y_val, predict)

# %% [markdown]
# We got an accuracy of more than 90% on the validation set. You can try to improve the score by tuning hyperparameters of the model. Let's now make the prediction on test dataset. We will make the similar changes in the test set as we have done in the training set before making the predictions.

# %%
test = pd.get_dummies(test)

# %%
test_prediction = clf.predict(test)

# %%
submission = pd.DataFrame()

# %%
# creating a Business_Sourced column and saving the predictions in it
submission['ID'] = test['ID']
submission['subscribed'] = test_prediction

# %% [markdown]
# Since the target variable is yes or no, we will convert 1 and 0 in the predictions to yes and no respectively.

# %%
submission['subscribed'].replace(0,'no',inplace=True)
submission['subscribed'].replace(1,'yes',inplace=True)

# %%
submission.to_csv('submission.csv', header=True, index=False)


