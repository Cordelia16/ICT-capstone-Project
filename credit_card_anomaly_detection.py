# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:06:53 2023

@author: Nigel
"""
# Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn import model_selection, preprocessing

def generate_auc_roc_cruve(model, X_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.legend(loc = 4)
    plt.show()
    pass

# ----------------------------------------------------------------------------
os.getcwd()
os.chdir('C:/Uni/2023/Capstone Project/Capstone_Project_Python')

data = pd.read_csv('Data/creditcard.csv')  # Reading the file .csv
df = pd.DataFrame(data)  # Converting data to Panda DataFrame
print(df.describe())

target = 'Class' # Setting target variable

X = df.loc[:, df.columns != target]
y = df.loc[:, df.columns == target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) # Splitting whole data set

# ----------------------------------------------------------------------------
# Histogram of 'Class' variable
class_hist = sns.countplot(x = target, data = df)
print(df[target].value_counts())

# ----------------------------------------------------------------------------
# SVM for whole data set
model_SVM_all = SVC(kernel = "linear", probability = True, random_state = 42)
model_SVM_all.fit(X_train, y_train)

y_pred_SVM_all = model_SVM_all.predict(X_test)

print("\n")
print(classification_report(y_test, y_pred_SVM_all))

fpr, tpr, thresholds = roc_curve(y_test,y_pred_SVM_all)
plt.plot(fpr, tpr, 'k-', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# ----------------------------------------------------------------------------
def under_sampling():
    # Under-sampling
    num_frauds = len(df[df[target] == 1])
    print("\nNumber of  frauds: ", num_frauds)
    
    non_fraud_indices = df[df[target] == 0].index
    
    random_non_fraud_indices = np.random.choice(non_fraud_indices, num_frauds, replace = False)
    
    fraud_indices = df[df[target] == 1].index
    
    under_sample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])
    under_sample = df.loc[under_sample_indices]
    under_sample.head()
    
    num_non_frauds_under = len(under_sample[under_sample[target] == 0])
    print("Number of non_frauds (under-sampled): ", num_non_frauds_under)
    
    # Split for under-sampled data
    X_under = under_sample.loc[:, under_sample.columns != target]
    y_under = under_sample.loc[:, under_sample.columns == target]
    
    X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under, y_under, test_size = 0.3, random_state = 42)
    return

under_sampling()

# ----------------------------------------------------------------------------
# Histogram of 'Class' variable of under-sampled data
#class_under_sample_hist = sns.countplot(x = target, data = under_sample)


# ----------------------------------------------------------------------------
def SVM_under_sampled():
    # SVM for under-sampled data
    model_SVM_under = SVC(kernel = "linear", probability = True, random_state = 42)
    
    model_SVM_under.fit(X_train, y_train)
    
    y_pred_SVM_under = model_SVM_under.predict(X_test)
    
    print("\n")
    print(classification_report(y_test, y_pred_SVM_under))
    
    model = model_SVM_under
    generate_auc_roc_cruve(model, X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test,y_pred_SVM_under)
    plt.plot(fpr, tpr, 'k-', lw=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    return

SVM_under_sampled()

# ----------------------------------------------------------------------------
def SVM_under_sampled_tuning():
    # Parameter Tuning (SVM - Under-sampled)
    params_SVM = [
        {'C': [0.5, 0.1, 1, 5, 10], 'kernel': ['linear'], 'class_weight': ['balanced']},
        {'C': [0.5, 0.1, 1, 5, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05,0.5],'kernel': ['rbf'], 'class_weight': ['balanced']}
    ]
    
    grs_SVM = GridSearchCV(model_SVM_under, params_SVM)
    
    grs_SVM.fit(X_train, y_train)
    
    print("Best Hyper Parameters:",grs_SVM.best_params_)
    
    model_SVM_under_best = grs_SVM.best_estimator_
    y_pred_SVM_under_best = model_SVM_under_best.predict(X_test)
    
    print(classification_report(y_test, y_pred_SVM_under_best))
    
    fpr, tpr, thresholds = roc_curve(y_test,y_pred_SVM_under_best)
    plt.plot(fpr, tpr, 'k-', lw=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    return

SVM_under_sampled_tuning():

# ----------------------------------------------------------------------------
# Isolation Forest
def isolation_forest():
    model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42)
    
    model_IF.fit(X_train, y_train)
    
    y_pred_IF = model_IF.predict(X_test)
    y_pred_IF = [1 if i == -1 else 0 for i in y_pred_IF]
    
    print("\n")
    print(classification_report(y_test, y_pred_IF))
    return

isolation_forest()

def isolation_forest_warm_start():
    # Isolation Forest with Warm Start On New Trees
    model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42, warm_start = True).fit(X_train, y_train)
    
    model_IF.n_estimators += 20
    model_IF.fit(X_train)
    
    y_pred_IF = model_IF.predict(X_test)
    y_pred_IF = [1 if i == -1 else 0 for i in y_pred_IF]
    
    print("\n")
    print(classification_report(y_test, y_pred_IF))
    
    fpr, tpr, thresholds = roc_curve(y_test,y_pred_IF)
    plt.plot(fpr, tpr, 'k-', lw=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    return

isolation_forest_warm_start()

# ----------------------------------------------------------------------------
# Tuning (Isolation Forest)
params_IF = {'n_estimators': list(range(100, 800, 5)), 
              'max_samples': list(range(100, 500, 5)), 
              'contamination': [0.1, 0.2, 0.3, 0.4, 0.5], 
              'max_features': [5,10,15], 
              'bootstrap': [True, False], 
              'n_jobs': [5, 10, 20, 30]}

f1_sc = make_scorer(y_test, y_pred_IF)

grs_IF = model_selection.GridSearchCV(model_IF, 
                      params_IF,
                      scoring = f1_sc,
                      refit=True,
                      cv=10, 
                      return_train_score=True)

grs_IF.fit(X_train, y_train)

print("Best Hyper Parameters:", grs_IF.best_params_)

# ----------------------------------------------------------------------------
# Autoencoder
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras import Model

encoding_dim = 15
input_data = Input(shape = (30, ))

encoded = Dense(encoding_dim, activation = "relu")(input_data)
decoded = Dense(30, activation = "sigmoid")(encoded)

autoencoder = Model(input_data, decoded)

encoder = Model(input_data, encoded)
encoded_input = Input(shape = (encoding_dim, ))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer = "adam", loss = "binary_crossentropy")

autoencoder.fit(X_train, X_train, epochs = 15, batch_size = 256, validation_data = (X_test, X_test))

encoded_data = encoder.predict(X_test)
decoded_data = decoder.predict(encoded_data)

autoencoder.summary()




