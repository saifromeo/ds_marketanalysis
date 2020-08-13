#My first program on DS
#Understanding the Data
#author s@if

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matpotlib inline
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix
import warnings
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import subplot, subplots, figure
pd.set_option('display.max_columns',None)
import six
import sys
sys.modules['sklearn.externals.six'] = six
warnings.filterwarnings('ignore')

path='../data/new_train.csv'
dataframe = pd.read_csv(path)

#validating datatypes

print(dataframe.dtypes)

#print("Shape :", dataframe.shape)
#print (dataframe.head(10))
# IDENTIFYING NUMERICAL FEATURES

numeric_data = dataframe.select_dtypes(include=np.number) # select_dtypes selects data with numeric features
numeric_col = numeric_data.columns                                                                # we will store the numeric features in a variable

print("Numeric Features:")
print(numeric_data.head())
print("===="*20)


# IDENTIFYING CATEGORICAL FEATURES
categorical_data = dataframe.select_dtypes(exclude=np.number) # we will exclude data with numeric features
categorical_col = categorical_data.columns                                                                              # we will store the categorical features in a variable


print("Categorical Features:")
print(categorical_data.head())
print("===="*20)

# To identify the number of missing values in every feature

# Finding the total missing values and arranging them in ascending order
total = dataframe.isnull().sum()

print("Total Sum :", total)
print("----"*20)

# Converting the missing values in percentage
percent = (dataframe.isnull().sum()/dataframe.isnull().count())

# dropping features having missing values more than 60%
dataframe = dataframe.drop((percent[percent > 0.6]).index,axis= 1)

# checking null values
print(dataframe.isnull().sum())

print("----"*20)

print(dataframe.head())

# imputing missing values with mean

for column in numeric_col:
    mean = dataframe[column].mean()
    dataframe[column].fillna(mean,inplace = True)
    
#   imputing with median
# for column in numeric_col:
#     mean = dataframe[column].median()
#     dataframe[column].fillna(mean,inpalce = True)


#checking or inbalance

# we are finding the percentage of each class in the feature 'y'
ycount=dataframe['y'].value_counts()
print ("Y Count : ", ycount)

ycountsum=dataframe['y'].value_counts().sum()
print ("YSumCount : ", ycountsum)

print("Dataframe Shape :", dataframe.shape)

percentbucket = (ycount/ycountsum)*100

print("% of yes/no buckets in y column\n", percentbucket)

#Detect Outliers using percentiles

cols = list(dataframe) # A List of all features

outliers = pd.DataFrame(columns=['Feature','Number of Outliers']) # Creating a new dataframe to

for column in numeric_col: # Iterating thorough each feature            
            # first quartile (Q1)
        q1 = dataframe[column].quantile(0.25) 
            
            # third quartile (Q3)
        q3 = dataframe[column].quantile(0.75)
            
            # IQR
        iqr = q3 - q1
            # 
        fence_low = q1 - (1.5*iqr)
        
        fence_high = q3 + (1.5*iqr)
            # finding the number of outliers using 'and(|) condition. 
        total_outlier = dataframe[(dataframe[column] < fence_low) | (dataframe[column] > fence_high)].shape[0]
        
        outliers = outliers.append({'Feature':column,'Number of Outliers':total_outlier},ignore_index=True)
outliers

print("OUTLIERS\n", outliers)
print("*"*50)

#Performing EDA tasks 
#Univariate (taking a variable and analysing against others)



# Impute mising values of categorical data with mode
for column in categorical_col:
    unknowncount=len(dataframe[dataframe[column]=='unknown'])
    print("Unknown in column ", column, ": ",unknowncount)
   
    #As we had unknown values in categorical columns, imputin gthem with mode
    mode = dataframe[column].mode()[0]
    dataframe[column] = dataframe[column].replace('unknown',mode)

# Selecting the categorical columns
categorical_col = dataframe.select_dtypes(include=['object']).columns
plt.style.use('ggplot')
# Plotting a bar chart for each of the cateorical variable
for column in categorical_col:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    dataframe[column].value_counts().plot(kind='bar')
    plt.title(column)
#plt.show()

# Univariate for numerical columns (taking a variable and analysing against others)
for column in numeric_col:
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    #simply using distplot was giving error, thus handled with some custom bandwidth
    sns.distplot(dataframe[column], kde_kws={'bw':0.1})
    plt.title(column)


#Box plot to figure out the outliers
for column in numeric_col:
    plt.figure(figsize=(20,6))
    plt.subplot(122)
    sns.boxplot(dataframe[column])
    plt.title(column)

#plt.show()
#dropping pdays, previous as data seems to be single value and thier variance is quite less
dataframe.drop(['pdays','previous'],1,inplace=True)

for column in categorical_col:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.countplot(x=dataframe[column],hue=dataframe['y'],data=dataframe)
    plt.title(column)    
    plt.xticks(rotation=90)



numeric_col = dataframe.select_dtypes(include=np.number).columns

for col in numeric_col:    
    dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))
    plt.figure(figsize=(20,4))
    plt.subplot(122)
    sns.boxplot(dataframe[col])
    plt.title("After winsorization")
    
plt.show()
