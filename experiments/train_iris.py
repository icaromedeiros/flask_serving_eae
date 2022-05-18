#!/usr/bin/env python
# coding: utf-8
from calendar import c
import pandas as pd
import pickle
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
# We define a Dataframe (tabular structure) with the predictor variables
# and on the other hand a separated vector with the response variable
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target # Target variable
#df_label = iris_df.copy()
#species_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    iris_df, y, test_size=0.25, random_state=70)

# Create the model
dtree = DecisionTreeClassifier()
# Train the model
dtree.fit(X_train, y_train)

# Exporting model to file
# Saves file (serialized dtree object)
# wb stands for Write Binary
pickle.dump(dtree, open('dtree_iris.pkl', 'wb'))

# Exporting test dataset to CSV files
X_test.to_csv('X_test.csv', index=False)
# y_test is a numpy array, we need to convert to pd.DataFrame first
pd.DataFrame(y_test, columns=['label']).to_csv('y_test.csv', index=False)