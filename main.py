# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:06:53 2023

@author: Nigel
"""

# ----------------------------------------------------------------------------
# Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

# ----------------------------------------------------------------------------
# Global variables
X_train = []
y_train = []

X_test = []
y_test = []

X_train_rus = []
y_train_rus = []

X_train_ros = []
y_train_ros = []

X_train_nearmiss = []
y_train_nearmiss = []

X_train_tomek = []
y_train_tomek = []

X_train_enn = []
y_train_enn = []

# ----------------------------------------------------------------------------
def create_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    return

def create_roc_curve(y_test, y_pred, model):
    ns_probs = [0 for _ in range(len(y_test))]
    
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    
    # ns_auc = roc_auc_score(y_test, ns_probs)
    # lr_auc = roc_auc_score(y_test, lr_probs)
    
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle = '--')
    plt.plot(lr_fpr, lr_tpr, marker = '.')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    return

def create_hist(target, df):
    sns.countplot(x = target, data = df)
    return

def random_under_sampling():
    # Random Under-sampling
    global X_train_rus
    global y_train_rus
    rus = RandomUnderSampler(random_state = 42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    return X_train_rus, y_train_rus

def NearMiss_under_sampling():
    global X_train_nearmiss
    global y_train_nearmiss
    nearmiss = NearMiss(version = 3)
    X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(X_train, y_train) # Under-sample the majority class
    return  X_train_nearmiss, y_train_nearmiss

def random_over_sampling():
    global X_train_ros
    global y_train_ros
    ros = RandomOverSampler(random_state = 42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    return X_train_ros, y_train_ros

def tomek_links():
    # Reference code: https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.TomekLinks.html
    global X_train_tomek
    global y_train_tomek
    tl = TomekLinks()
    X_train_tomek, y_train_tomek = tl.fit_resample(X_train, y_train)
    return X_train_tomek, y_train_tomek

def enn_under_sampling():
    # Reference code: https://medium.com/quantyca/oversampling-and-undersampling-adasyn-vs-enn-60828a58db39
    global X_train_enn
    global y_train_enn
    enn = EditedNearestNeighbours()
    X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)
    return X_train_enn, y_train_enn
# ----------------------------------------------------------------------------
os.getcwd()
os.chdir('C:/Uni/2023/Capstone Project/Capstone_Project_Python')

# ----------------------------------------------------------------------------
# DATASET 1
data = pd.read_csv('Data/creditcard.csv')  # Reading the file .csv
df = pd.DataFrame(data)  # Converting data to Panda DataFrame
print(df.describe())

target = 'Class' # Setting target variable

X = df.drop('Class', axis = 1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------
# DATASET 2
# data = pd.read_csv('Data/PS_20174392719_1491204439457_log.csv')  # Reading the file .csv
# df = pd.DataFrame(data)  # Converting data to Panda DataFrame
# df.describe()

# target = 'isFraud'

# df = df.sample(frac = 0.05)

# df = df.drop(['type', 'nameOrig', 'nameDest'], axis = 1)

# X = df.drop('isFraud', axis = 1)
# y = df['isFraud']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------
# Histogram of 'Class' variable for whole data set
df = pd.DataFrame(data) 
create_hist(target, df)

# ----------------------------------------------------------------------------
# Random Under-sampling
random_under_sampling()

# ----------------------------------------------------------------------------
# Histogram of 'Class' variable of under-sampled data


# ---------------------------------------------------------------------------- 
# SVM (Random Under-sampling)
model_SVM = SVC(kernel = "linear", probability = True, random_state = 42) # Create and train model
model_SVM.fit(X_train_rus, y_train_rus)

y_pred = model_SVM.predict(X_test) # Testing model

support_vectors = model_SVM.support_vectors_ # Store support vectors

model = model_SVM # Change to 'model' so it can be passed to 'create_roc_curve' function
model.score(X_train_rus, y_train_rus)
model.score(X_test, y_test)

print("\nClassification Report - SVM (Random Under-sampling)\n")
create_classification_report(y_test, y_pred) # Create classification report
create_roc_curve(y_test, y_pred, model) # Create ROC curve

# ----------------------------------------------------------------------------
# One Class SVM (Under-sampled)
model_OCSVM = OneClassSVM(gamma = "scale", kernel = "linear")
model_OCSVM.fit(X_train_rus, y_train_rus)

y_pred = model_OCSVM.predict(X_test)

model = model_OCSVM # Change to 'model' so it can be passed to 'create_roc_curve' function

print("\nClassification Report - One Class SVM (Random Under-sampling)\n")
create_classification_report(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# ----------------------------------------------------------------------------
# Isolation Forest (Random Under-sampling)
# Reference code: https://medium.com/grabngoinfo/isolation-forest-for-anomaly-detection-cd7871ae99b4
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42)
model_IF.fit(X_train_rus)
y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest (Random Under-sampling)\n")
create_classification_report(y_test, y_pred) # Create classification report

model_IF.get_params()


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# Isolation Forest with Warm Start On New Trees
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42, warm_start = True)
model_IF.n_estimators += 20
model_IF.fit(X_train_rus)
y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest with Warm Start On New Trees (Random Under-sampling)\n")
create_classification_report(y_test, y_pred) # Create classification report

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# ----------------------------------------------------------------------------
# Autoencoder (Random Under-sampling)
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

# ----------------------------------------------------------------------------
# Under-sampling (NearMiss)
NearMiss_under_sampling()

# ----------------------------------------------------------------------------
# SVM (NearMiss)
model_SVM = SVC(kernel = "linear", probability = True, random_state = 42)
model_SVM.fit(X_train_nearmiss, y_train_nearmiss) # Train model

y_pred = model_SVM.predict(X_test) # Make predictions using test data

model = model_SVM # Change to 'model' so it can be passed to 'create_roc_curve' function

print("\nClassification Report - SVM (NearMiss)\n")
create_classification_report(y_test, y_pred) # Create classification report
create_roc_curve(y_test, y_pred, model) # Create ROC curve

# ----------------------------------------------------------------------------
# Isolation Forest (NearMiss)
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42)
model_IF.fit(X_train_nearmiss)

y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest (NearMiss)\n")
create_classification_report(y_test, y_pred) # Create classification report

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# Isolation Forest with Warm Start On New Trees (NearMiss)
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42, warm_start = True)
model_IF.n_estimators += 20
model_IF.fit(X_train_nearmiss)

y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest with Warm Start On New Trees (NearMiss)\n")
create_classification_report(y_test, y_pred) # Create classification report

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# ----------------------------------------------------------------------------
# Under-sampling (TomekLinks)
tomek_links()

# ---------------------------------------------------------------------------- 
# SVM (TomekLinks)
model_SVM = SVC(kernel = "linear", probability = True, random_state = 42) # Create and train model
model_SVM.fit(X_train_tomek, y_train_tomek)

y_pred = model_SVM.predict(X_test) # Testing model

support_vectors = model_SVM.support_vectors_ # Store support vectors

model = model_SVM # Change to 'model' so it can be passed to 'create_roc_curve' function
model.score(X_train_tomek, y_train_tomek)
model.score(X_test, y_test)

print("\nClassification Report - SVM (TomekLinks)\n")
create_classification_report(y_test, y_pred) # Create classification report
create_roc_curve(y_test, y_pred, model) # Create ROC curve

# ----------------------------------------------------------------------------
# Isolation Forest (TomekLinks)
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42)
model_IF.fit(X_train_tomek)

y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest (TomekLinks)\n")
create_classification_report(y_test, y_pred) # Create classification report

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# Isolation Forest with Warm Start On New Trees (TomekLinks)
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42, warm_start = True)
model_IF.n_estimators += 20
model_IF.fit(X_train_tomek)

y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest with Warm Start On New Trees (TomekLinks)\n")
create_classification_report(y_test, y_pred) # Create classification report

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# ----------------------------------------------------------------------------
# Under-sampling (ENN)
enn_under_sampling()

# ---------------------------------------------------------------------------- 
# SVM (ENN)
model_SVM = SVC(kernel = "linear", probability = True, random_state = 42) # Create and train model
model_SVM.fit(X_train_enn, y_train_enn)

y_pred = model_SVM.predict(X_test) # Testing model

support_vectors = model_SVM.support_vectors_ # Store support vectors

model = model_SVM # Change to 'model' so it can be passed to 'create_roc_curve' function
model.score(X_train_enn, y_train_enn)
model.score(X_test, y_test)

print("\nClassification Report - SVM (ENN)\n")
create_classification_report(y_test, y_pred) # Create classification report
create_roc_curve(y_test, y_pred, model) # Create ROC curve

# ----------------------------------------------------------------------------
# Isolation Forest (ENN)
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42)
model_IF.fit(X_train_enn)

y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest (ENN)\n")
create_classification_report(y_test, y_pred) # Create classification report

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# Isolation Forest with Warm Start On New Trees (ENN)
model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42, warm_start = True)
model_IF.n_estimators += 20
model_IF.fit(X_train_enn)

y_pred = model_IF.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]

print("\nClassification Report - Isolation Forest with Warm Start On New Trees (ENN)\n")
create_classification_report(y_test, y_pred) # Create classification report

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'k-', lw = 2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()