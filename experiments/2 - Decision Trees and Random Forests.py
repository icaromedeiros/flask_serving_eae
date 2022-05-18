#!/usr/bin/env python
# coding: utf-8

# # Lab - Classification and Regression Trees
# 
# Pere Miquel Brull Borràs, Marta Tolós Rigueiro, Alberto Villa Manrique, Ícaro Medeiros

# # Requirements

# In[1]:


# Required
get_ipython().system('pip install scikit-learn pandas seaborn')


# ## Optional requirements for visualization (GraphViz, pydotplus, dtreeviz)

# 1. Follow instructions on https://graphviz.org/download/ and install graphviz
# > **Warning** Might include advanced Operating System configurations like changing `PATH` environment variable and possibly rebooting your computer
# 1. Install Python libraries to deal with GraphViz format (pydotplus)
# 1. Install dtreeviz library

# In[2]:


get_ipython().system('pip install pydotplus')


# ## Optional requirementes for dtreeviz

# In[3]:


get_ipython().system('pip install dtreeviz')


# ## Iris Dataset
# 
# ![image.png](attachment:image.png)
# 
# Reading the iris dataset:

# In[4]:


import sklearn.datasets as datasets

iris = datasets.load_iris()
iris


# In[5]:


print(iris.DESCR)


# ## Classifying Flowers into species by their physical characteristics

# ## Reading the iris dataset

# In[6]:


import pandas as pd

# We define a Dataframe (tabular structure) with the predictor variables
# and on the other hand a separated vector with the response variable
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target # Target variable

iris_df.head()


# Understanding flower parts [https://www.britannica.com/science/sepal]()

# In[7]:


y # Flower type 0,1,2


# In[8]:


# Perfectly balanced classes
# 50 examples for each species
pd.Series(y).hist()


# In[9]:


iris.target_names


# Let's merge df using map to create a new df_label with the name of the species to enable better visualization!

# In[10]:


df_label = iris_df.copy()
df_label.head()


# In[11]:


iris.target_names


# In[12]:


iris


# In[13]:


species_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}


# In[14]:


df_label['species'] = pd.Series(y).map(species_dict)
df_label.head()


# # Visualizing correlations between variables

# In[15]:


import seaborn as sns

sns.pairplot(df_label)


# Why there is no information about species: it's a categorical value, so `seaborn` does not know what to do with it.
# 
# We can use the species information to create a new dimension like to the visualization, though.

# In[16]:


help(sns.pairplot)


# In[17]:


sns.pairplot(df_label,
             hue='species',
             plot_kws=dict(alpha=0.3) # Transparency prevents under-representation of overlapping points
            )


# # Decision Tree Machine Learning model

# Decisions trees is a model following rule production systems, in which domain experts would create rules by hand:
# 
# ![image.png](attachment:image.png)

# In decision tree algorithms we're letting computers to discover these fluxograms automatically.

# In decision trees we're creating structures like
# 
# ![image.png](attachment:image.png)

# ## Theory introduction
# 
# **Decision Trees** or CART (Classification and Regression Trees):
# 
# This method is based on the order of the data. It is, therefore, an iterative method in which each step will try to reduce the *impurity* of the data based on the variable to be predicted. Digging a bit:
# 
# 1. We start with all our data in the same bag, therefore I have all the different categories mixed together.
# By categories we mean the class, the target variable, i.e. what we're trying to predict in a classification task.
# 
#     1. The impurity of this data bag is modeled using the **Gini impurity index** (not to be confused with the Gini coefficient).
#     1. Simplifying it, it could be seen as: If we introduce a new observation into our bag, whose response variable has been chosen based on the distribution of the different categories, what is the probability of being wrong if I try to classify it?
#     1. In other terms: if I have a sack with two pears and two oranges, the impurity is 50%, since a new variable could be with the same probability either pear or orange.
# 
# 2. Then, as we want to reduce this impurity, we will distribute the observations in different bags in each iteration based on the characteristics of our variables until we obtain groups where there is no possible error: Either everything is pears or everything is oranges.
# 
# > OBS: In this practice there are some decision tree graphs that you do not need to reproduce. To do them, you not only have to install Python libraries, but also install **GraphViz** on our system and configure the executable as an environment variable. Those of you who have time and curiosity to keep playing, go ahead. But **it is not a requirement of this practice class and used to explain the Decision Tree algorithm only**.
# 

# # Dividing the dataset into train and test datasets

# We are going to create a decision tree with a partition between train and test of the data. Look at the variable *random_state* that is applied in the `train_test_split` function. By changing this variable, the random distribution between the train and test data will change.

# In[18]:


from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier


# In[19]:


# Split data in train & test

# X_train, X_test, y_train, y_test = train_test_split(
#     df_small, y, test_size=0.25, random_state=70)

X_train, X_test, y_train, y_test = train_test_split(
    iris_df, y, test_size=0.25, random_state=70)

# Create the model
dtree = DecisionTreeClassifier()
# Train the model
dtree.fit(X_train, y_train)


# # Exporting model to file

# In[20]:


import pickle


# In[37]:


# Saves file (serialized dtree object)
# wb stands for Write Binary
pickle.dump(dtree, open('dtree_iris.pkl', 'wb'))


# In[34]:


get_ipython().system('ls | grep pkl')


# In[38]:


# You can run the rest of the notebook replacing the variable dtree with the loaded version
#  This line can be commented as well
dtree = pickle.load(open('dtree_iris.pkl', 'rb'))


# ### X and y notations

# If you notice the name of the variables above, you'll see the X's and y's. Y is that?
# 
# In math notation, we define the dataset X as the array of feature vectors, each vector representing the description of a flower in our dataset, each feature representing an aspect of a flower). Will all these values in X, we want to infer an approximate function $\hat{y}$, using the labeled dataset in variable `y` to better **fit** the data in `X`.
# 
# Or $\hat{y} = f(X)$

# ### Recommended exercise, create a model with only one feature

# In[20]:


# Replace code in cells below with df_small

df_small = iris_df[["sepal length (cm)"]]
df_small.head()


# ## Visualizing resulting decision tree

# In[39]:


import pydotplus
from io import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz


# In[40]:


dot_data = StringIO()
export_graphviz(dtree,
                out_file=dot_data,  
                feature_names=iris.feature_names,
                class_names=iris.target_names,
                filled=True,
                rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# We can see that there is a $X_2$ variable that strongly marks a first cut, so that already in the first split we obtain a partition without impurity. It is a model capable of obtaining a very accurate division of the data in very few steps.

# In[23]:


from dtreeviz.trees import dtreeviz

viz = dtreeviz(dtree, X_train, y_train,
               target_name="Species",
               feature_names=iris.feature_names,
               class_names=list(iris.target_names))

viz


# ### Let's test a different distribution of train/test data
# 
# If we now repeat this process with a different distribution between train and test, which implies that the model has been trained with other data, by changing `random_state` to other value ...

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_df, y, test_size=0.25, random_state=60
)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

viz = dtreeviz(dtree, X_train, y_train,
               target_name="Species",
               feature_names=iris.feature_names,
               class_names=list(iris.target_names))

viz


# So now the resulting tree is a bit different

# In[25]:


# Let's check the influence of the random_state variable
help(train_test_split)


# ## Some words about overfitting
# 
# Sometimes you train a ML algorithm on the labeled dataset you have but when faced with new unseen data the model starts performing bad, making wrong classifications because the model hasn't generalized the data paterns correctly, it was only replicating the data distribution in a highly specialized way.
# 
# That happens when the ML model overfits to the data it was presented and does not understand different kind of examples and patterns from unseen data.
# 
# **Overfitting in regression**
# ![image.png](attachment:image.png)
# 
# **Overfitting in classification**
# ![image-2.png](attachment:image-2.png)
# 
# **Overfitting in real life**
# ![image-4.png](attachment:image-4.png)

# # Train-validation-test split
# 
# One way to try to cope with overfitting is doing a more complex strategy for dividing the dataset into:
# 
# 1. Train datased: used to train the model
# 1. Validation dataset: used to optimize our model parameters to achieve best accuracy possible
# 1. Test dataset: unseen data separated to test our model with possibly different patterns not found in previous datasets

# In[26]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_df, y, test_size=0.25, random_state=42
)

X_t, X_val, y_t, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42
)

print(f'Training size: {len(X_t)}, Validation size: {len(X_val)}, Test size: {len(X_test)}')


# In[27]:


# Create the model
dtree = DecisionTreeClassifier()

# Train the model
dtree.fit(X_t, y_t)


# In[28]:


viz = dtreeviz(dtree, X_t, y_t,
               target_name="Species",
               feature_names=iris.feature_names,
               class_names=list(iris.target_names))

viz


# ## Checking accuracy with the validation dataset

# In[29]:


from sklearn.metrics import confusion_matrix, accuracy_score

y_hat = dtree.predict(X_val)
y_hat


# In[30]:


print(f'Accuracy with validation {accuracy_score(y_val, y_hat)}')


# At this point we can do multiple runs of train/validation checks to improve the accuracy score...

# ## Inferring over new data

# In[31]:


# The same but with X_test, y_test
y_classification = dtree.predict(X_test)
y_classification


# In[32]:


print(f'Accuracy with test dataset {accuracy_score(y_test, y_classification)}')


# ## Decision trees for regression problems

# In[33]:


# Based on https://mljar.com/blog/visualize-decision-tree/

boston = datasets.load_boston()
print(boston.DESCR)


# ![image.png](attachment:image.png)

# In the above example we can see Decision Trees create **LINEAR** decision boundaries for its variables.
# This is amazing to generate (visual) explanations for your resulting model...
# 
# ![image.png](attachment:image.png)
# 
# What would happen if the feature vector distribution is like in the image on the right?
# That's why in the next class we're studying Support Vector Machines, able to divide the feature space into curved decision boundaries.

# ## Additional reading
# 
# - [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
# - [Understanding Decision Trees with Python](https://datascience.foundation/sciencewhitepaper/understanding-decision-trees-with-python)
# - [4 Ways to Visualize Individual Decision Trees in a Random Forest](https://towardsdatascience.com/4-ways-to-visualize-individual-decision-trees-in-a-random-forest-7a9beda1d1b7)
# - [How to Visualize a Random Forest with Fitted Parameters?](https://analyticsindiamag.com/how-to-visualize-a-random-forest-with-fitted-parameters/)
# - [Understanding Random forest better through visualizations](https://garg-mohit851.medium.com/random-forest-visualization-3f76cdf6456f)

# ## Improving the Model
# 
# There are two very useful techniques that allow us to create new models:
# 
# * **Bootstrap**: It is based on choosing subsamples of our data in a uniform way and with repetition, thus creating multiple smaller samples that share the distributions of the original sample.
# 
# * **Bagging**: Generate a bootstrap of size $n$, train a model on that subsample and repeat the process $m$ times.
# ---
# 
# - **Q3**: Could bagging be useful for decision trees? Why? (Discuss in terms of robustness of the resulting model, computational capacity and parallelization)
# - **Q4**: Research about the *Out-Of-Bag* error. How can we tie it to methods like cross validation? Does it make sense to apply both?

# ## Random Forests
# 
# Now let's apply an *ensemble* method such as *Random Forest*, based on the results of multiple decision trees.
# 
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)
# 
# ![image-2.png](attachment:image-2.png)

# But first, a bit of theory in combining classifiers:
# 
# Apart from majority decision we can combine classifier
# - Mean Average
# - Product rule
# - Borda rule
# 
# ![image-2.png](attachment:image-2.png)
# 
# In probability terms, using the produt rule:
# 
# ![image.png](attachment:image.png)
# 
# 

# ### Training a Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    iris_df, y, test_size=0.33, random_state=100)

rf = RandomForestClassifier(n_jobs=-1) # parallelize the execution 
rf.fit(X_train, y_train)


# Let's analyze the code ... ([RFs in Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for literature)

# In[35]:


help(rf)


# In[36]:


y_test


# In[37]:


rf.predict(X_test)


# In[40]:


# same as help(accuracy_score)
print(accuracy_score.__doc__)


# In[41]:


rf = RandomForestClassifier(n_jobs=4)
rf


# In[42]:


rf.fit(X_train, y_train)


# In[43]:


importances = rf.feature_importances_
importances


# In[44]:


indexes = np.argsort(importances)[::-1]
indexes


# In[45]:


iris_df.columns


# In[46]:


# Map importances to species_names
names = [iris_df.columns[i] for i in indexes]
names


# In[47]:


# Prepare a barplot
plt.bar(range(X_train.shape[1]), importances[indexes])
# Add the feature names
plt.xticks(range(X_train.shape[1]), names, rotation=20, fontsize = 8)
# Add the title
plt.title("Feature Importance")
# Self explanatory...
plt.show()


# In the code of the upper cell we see not only how we can apply a model, but that it contains the importance of the different *features* or predictor variables.
# 
# - **Q6**: In the case of the Random Forest, how do you think this importance is calculated?

# ## Predict

# In[48]:


import numpy as np
from sklearn.metrics import confusion_matrix

rf_preds = rf.predict(X_test)
rf_conf_mat = confusion_matrix(y_test, rf_preds)
rf_conf_mat


# In[49]:


help(confusion_matrix)


# In[50]:


rf_conf_mat.sum()


# In[51]:


# Convert to numpy
np_mat = np.asarray(rf_conf_mat)

acc = sum(np.diagonal(np_mat)) / np_mat.sum()
print(f"My accuracy is: {acc}")

