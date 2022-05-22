# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def read_csv_1(data_file):
    data = pd.read_csv(data_file)
    data.drop(labels='fnlwgt', axis=1, inplace=True)
    return data


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    return len(df.index)


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    list = []
    for col in df.columns:
        list.append(col)
    return list


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    return sum(df.isnull().sum())


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
    return df.columns[df.isnull().any()].tolist()


# Return the percentage of instances corresponding to persons whose education level is
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
    return round(len(df[(df['education'] == 'Bachelors') | (df['education'] == 'Masters')]) / len(df), 3)


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    data1 = df.copy()
    return data1.dropna(axis=0, how='any', inplace=False)


# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
    data0 = df.copy()
    data0 = data_frame_without_missing_values(data0)
    data1 = data0.drop(['class'], axis=1)
    data2 = pd.get_dummies(data1, columns=data1.columns)
    return data2


# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    data1 = df.copy()
    data1 = data_frame_without_missing_values(data1)
    label_encoder = preprocessing.LabelEncoder()
    data1['class'] = label_encoder.fit_transform(data1['class'])
    class_series = data1['class']
    return class_series


# Given a training set X_train containing the input attribute values
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train, y_train):
    dt_default = DecisionTreeClassifier()
    dt_default.fit(X_train, y_train)
    y_pred = dt_default.predict(X_train)
    result = pd.Series(y_pred)
    return result


# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    er = 1 - accuracy_score(y_pred, y_true)
    return er

