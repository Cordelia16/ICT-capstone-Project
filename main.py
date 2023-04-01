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

from sklearn.metrics import roc_curve, classification_report, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

# ----------------------------------------------------------------------------
# Global variables
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

def create_roc_curve(y_test, y_pred, model, name):
    name = name
    r_probs = [0 for _ in range(len(y_test))]
    
    model_probs = model.predict_proba(X_test)
    model_probs = model_probs[:, 1]
    
    r_auc = roc_auc_score(y_test, r_probs)
    model_auc = roc_auc_score(y_test, model_probs)   
    
    r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
    
    # Plot ROC curve
    plt.title(name)
    plt.plot(r_fpr, r_tpr, linestyle = '--')
    plt.plot(model_fpr, model_tpr, marker = '.', label= 'ROC curve (AUROC = %0.3f)' % model_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = "lower right")
    plt.show()
    return

def create_IF_roc_curve(y_test, y_pred, name):
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.title(name)
    plt.plot(fpr, tpr, color = "darkorange", lw = 2, label = "ROC curve (AUROC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = "lower right")
    plt.show()
    return

def create_autoencoder_roc_curve(y_test, y_pred, name):
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color = "darkorange", lw = 2, label = "ROC curve (AUROC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc = "lower right")
    plt.show()
    return

def create_hist(target, df, hist_title):
    sns.countplot(x = target, data = df)
    plt.title(hist_title)
    plt.show()
    return

def create_corr(corr, corr_title):
    sns.heatmap(corr, cmap='Blues')
    plt.title(corr_title)
    plt.show()
    return

def random_under_sampling():
    global X_train_rus
    global y_train_rus
    
    rus = RandomUnderSampler(random_state = 42) # Declare random under-sampler
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train) # Resample training data
    
    scaler = StandardScaler() # Declare scaler
    X_train_rus = scaler.fit_transform(X_train_rus) # Scale training data
    return X_train_rus, y_train_rus

def NearMiss_under_sampling():
    global X_train_nearmiss
    global y_train_nearmiss
    
    nearmiss = NearMiss(version = 3) # Declare near-miss under-sampler
    X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(X_train, y_train) # Resample training data
    
    scaler = StandardScaler() # Declare scaler
    X_train_nearmiss = scaler.fit_transform(X_train_nearmiss) # Scale training data
    return  X_train_nearmiss, y_train_nearmiss

def random_over_sampling():
    global X_train_ros
    global y_train_ros
    
    ros = RandomOverSampler(random_state = 42) # Declare random over-sampler
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train) # Resample training data
    
    return X_train_ros, y_train_ros

def tomek_links():
    # Reference code: https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.TomekLinks.html
    global X_train_tomek
    global y_train_tomek
    
    tl = TomekLinks() # Declare tomek links under-sampler
    X_train_tomek, y_train_tomek = tl.fit_resample(X_train, y_train) # Resample training data
    
    scaler = StandardScaler() # Declare scaler
    X_train_tomek = scaler.fit_transform(X_train_tomek) # Scale training data
    return X_train_tomek, y_train_tomek

def enn_under_sampling():
    # Reference code: https://medium.com/quantyca/oversampling-and-undersampling-adasyn-vs-enn-60828a58db39
    global X_train_enn
    global y_train_enn
    
    enn = EditedNearestNeighbours() # Declare ENN under-sampler
    X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train) # Resample training data
    
    scaler = StandardScaler() # Declare scaler
    X_train_enn = scaler.fit_transform(X_train_enn) # Scale training data
    return X_train_enn, y_train_enn

def train_SVM(X_train, y_train, X_test, y_test, name):
    model_SVM = SVC(kernel = "linear", probability = True, random_state = 42) # Create model
    model_SVM.fit(X_train, y_train) # Train model
    y_pred = model_SVM.predict(X_test) # Testing model
    
    # support_vectors = model_SVM.support_vectors_ # Store support vectors
    
    model = model_SVM # Change to 'model' so it can be passed to 'create_roc_curve' function
    
    print("\nClassification Report - SVM\n")
    create_classification_report(y_test, y_pred) # Create classification report
    create_roc_curve(y_test, y_pred, model, name) # Create ROC curve
    return

def train_IF(X_train, X_test, y_test, name):
    # Reference code: https://medium.com/grabngoinfo/isolation-forest-for-anomaly-detection-cd7871ae99b4
    model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42) # Create model
    model_IF.fit(X_train) # Train model
    y_pred = model_IF.predict(X_test) # Test model
    y_pred = [1 if i == -1 else 0 for i in y_pred] # Change values to 0 and 1 from -1 and 1
    
    print("\nClassification Report - Isolation Forest\n")
    create_classification_report(y_test, y_pred) # Create classification report
    
    create_IF_roc_curve(y_test, y_pred, name)
    return

def train_IF_warm_trees(X_train, X_test, y_test, name):
    model_IF = IsolationForest(n_estimators = 100, contamination = 0.5, random_state = 42, warm_start = True) # Create model
    model_IF.n_estimators += 20 # Add trees
    model_IF.fit(X_train) # Train model
    y_pred = model_IF.predict(X_test) # Test model
    y_pred = [1 if i == -1 else 0 for i in y_pred] # Change values to 0 and 1 from -1 and 1
    
    print("\nClassification Report - Isolation Forest with Warm Start On New Trees\n")
    create_classification_report(y_test, y_pred) # Create classification report
    
    create_IF_roc_curve(y_test, y_pred, name)
    return

def train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name):
    encoding_dim = 32

    input_data = Input(shape = (shape_size, ))
    
    encoded = Dense(encoding_dim, activation = 'relu')(input_data)
    decoded = Dense(shape_size, activation = 'sigmoid')(encoded)
    
    autoencoder = Model(input_data, decoded)
    encoder = Model(input_data, encoded)
    
    encoded_input = Input(shape = (encoding_dim, ))
    decoded_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoded_layer(encoded_input))

    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    
    name = name
    
    autoencoder.fit(X_train, X_train,
                epochs = 50,
                batch_size = 256,
                shuffle = True,
                validation_data = (X_test, X_test))

    input_data = Input(shape = (shape_size, ))
    encoded = encoder(input_data)
    output = Dense(1, activation = 'sigmoid')(encoded)
    classifier = Model(input_data, output)
    
    # Freeze the weights of the encoder
    encoder.trainable = False
    
    # Compile the binary classifier
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Train the binary classifier
    classifier.fit(X_train, y_train,
                   epochs = 50,
                   batch_size = 256,
                   shuffle = True,
                   validation_data=(X_test, y_test))
    
    y_pred = classifier.predict(X_test)
    create_autoencoder_roc_curve(y_test, y_pred, name)
    return

# ----------------------------------------------------------------------------
# Get and set working directory
os.getcwd()
os.chdir('C:/Uni/2023/Capstone Project/Capstone_Project_Python')

# ----------------------------------------------------------------------------
# DATASET 1
data = pd.read_csv('Data/creditcard.csv')  # Reading the file .csv
df = pd.DataFrame(data)  # Converting data to Panda DataFrame

target = 'Class' # Setting target variable
dataset_name = "Credit Card Dataset"
shape_size = 30 # For Credit Card Dataset

# ----------------------------------------------------------------------------
# EDA
# Correlation Heatmap
corr = df.corr()
corr_title = "Correlation Heatmap (" + dataset_name + " - Whole)"
create_corr(corr, corr_title)

# Distribution of 'Class' variable
df = pd.DataFrame(data) 
hist_title = target
create_hist(target, df, hist_title)

# ----------------------------------------------------------------------------
# Train Test Split
X = df.drop('Class', axis = 1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------------- 
# Run sampling methods
random_under_sampling()
print("Random Under-sampling Complete")
NearMiss_under_sampling()
print("Near-Miss Under-sampling Complete")
tomek_links()
print("Tomek Links Under-sampling Complete")
enn_under_sampling()
print("ENN Under-sampling Complete")

# ---------------------------------------------------------------------------- 
# SVM
name = "SVM (" + dataset_name + " - Whole)"

train_SVM(X_train, y_train, X_test, y_test, name)

# ---------------------------------------------------------------------------- 
# Isolation Forest
# Reference code: https://medium.com/grabngoinfo/isolation-forest-for-anomaly-detection-cd7871ae99b4
train_IF(X_train, X_test, y_test)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test)

# ----------------------------------------------------------------------------
# Autoencoder


# ---------------------------------------------------------------------------- 
# SVM (Random Under-sampling)
X_train = X_train_rus
y_train = y_train_rus
print("Complete")
name = "SVM (" + dataset_name + " - Random Under-sampling)"
print("Complete")
train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# One Class SVM (Under-sampled)
# model_OCSVM = OneClassSVM(gamma = "scale", kernel = "linear")
# model_OCSVM.fit(X_train_rus, y_train_rus)
# y_pred = model_OCSVM.predict(X_test)

# model = model_OCSVM # Change to 'model' so it can be passed to 'create_roc_curve' function

# print("\nClassification Report - One Class SVM (Random Under-sampling)\n")
# create_classification_report(y_test, y_pred)

# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# plt.plot(fpr, tpr, 'k-', lw = 2)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.show()

# ----------------------------------------------------------------------------
# Isolation Forest (Random Under-sampling)
X_train = X_train_rus

name = "Isolation Forest (" + dataset_name + " - Random Under-sampling)"

train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# ----------------------------------------------------------------------------
# Autoencoder (Random Under-sampling)
X_train = X_train_rus
y_train = y_train_rus

name = "Autoencoder (" + dataset_name + " - Random Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# ----------------------------------------------------------------------------
# SVM (NearMiss)
X_train = X_train_nearmiss
y_train = y_train_nearmiss

name = "SVM (" + dataset_name + " - Near-Miss Under-sampling)"

train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# Isolation Forest (NearMiss)
X_train = X_train_nearmiss
name = "Isolation Forest (" + dataset_name + " - Near-Miss Under-sampling)"
train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# ----------------------------------------------------------------------------
# Autoencoder (NearMiss)
X_train = X_train_nearmiss
y_train = y_train_nearmiss

# name = "Autoencoder (Credit Card Dataset - Near-Miss Under-sampling)"
name = "Autoencoder (PaySim Dataset - Near-Miss Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# ---------------------------------------------------------------------------- 
# SVM (TomekLinks)
X_train = X_train_tomek
y_train = y_train_tomek

name = "SVM (" + dataset_name + " - Tomek Links Under-sampling)"

train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# Isolation Forest (TomekLinks)
X_train = X_train_tomek
name = "Isolation Forest (" + dataset_name + " - Tomek Links Under-sampling)"
train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# ----------------------------------------------------------------------------
# Autoencoder (TomekLinks)
X_train = X_train_tomek
y_train = y_train_tomek

name = "Autoencoder (" + dataset_name + " - Tomek Links Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# ---------------------------------------------------------------------------- 
# SVM (ENN)
X_train = X_train_enn
y_train = y_train_enn

name = "SVM (" + dataset_name + " - ENN Under-sampling)"

train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# Isolation Forest (ENN)
X_train = X_train_enn
name = "Isolation Forest (" + dataset_name + " - ENN Under-sampling)"
train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# ----------------------------------------------------------------------------
# Autoencoder (ENN)
X_train = X_train_enn
y_train = y_train_enn

name = "Autoencoder (" + dataset_name + " - ENN Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# ---------------------------------------------------------------------------- 
# Correlation Heatmaps
# Whole dataset
# corr = df.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Whole)"
# create_corr(corr, corr_title)

# # Random Under-sampling
# df_rus = X_train_rus
# df_rus[target] = y_train_rus

# corr = df_rus.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Random Under-sampling)"
# create_corr(corr, corr_title)

# # NearMiss
# df_nearmiss = X_train_nearmiss
# df_nearmiss[target] = y_train_nearmiss

# corr = df_nearmiss.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Near-Miss Under-sampling)"
# create_corr(corr, corr_title)

# # Tomek Links
# df_tomek = X_train_tomek
# df_tomek[target] = y_train_tomek

# corr = df_tomek.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Tomek Links Under-sampling)"
# create_corr(corr, corr_title)

# # ENN
# df_enn = X_train_enn
# df_enn[target] = y_train_enn

# corr = df_enn.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - ENN Under-sampling)"
# create_corr(corr, corr_title)

# ----------------------------------------------------------------------------
# DATASET 2
data = pd.read_csv('Data/PS_20174392719_1491204439457_log.csv')  # Reading the file .csv
df = pd.DataFrame(data)  # Converting data to Panda DataFrame

target = 'isFraud'
dataset_name = "PaySim Dataset"
shape_size = 7 # For PaySim Dataset

df[target].value_counts()

df = df.sample(frac = 0.2)
df = df.drop(['type', 'nameOrig', 'nameDest'], axis = 1)
print(df.head())

# # Correlation Heatmaps - Whole dataset
# corr = df.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Whole)"
# create_corr(corr, corr_title)

X = df.drop('isFraud', axis = 1)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# ---------------------------------------------------------------------------- 
# Run sampling methods
random_under_sampling()
print("Random Under-sampling Complete")
NearMiss_under_sampling()
print("Near-Miss Under-sampling Complete")
tomek_links()
print("Tomek Links Under-sampling Complete")
enn_under_sampling()
print("ENN Under-sampling Complete")

# ---------------------------------------------------------------------------- 
# SVM (Random Under-sampling)
X_train = X_train_rus
y_train = y_train_rus
print("Complete")
name = "SVM (" + dataset_name + " - Random Under-sampling)"
print("Complete")
train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# One Class SVM (Under-sampled)
# model_OCSVM = OneClassSVM(gamma = "scale", kernel = "linear")
# model_OCSVM.fit(X_train_rus, y_train_rus)
# y_pred = model_OCSVM.predict(X_test)

# model = model_OCSVM # Change to 'model' so it can be passed to 'create_roc_curve' function

# print("\nClassification Report - One Class SVM (Random Under-sampling)\n")
# create_classification_report(y_test, y_pred)

# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# plt.plot(fpr, tpr, 'k-', lw = 2)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.show()

# ----------------------------------------------------------------------------
# Isolation Forest (Random Under-sampling)
X_train = X_train_rus

name = "Isolation Forest (" + dataset_name + " - Random Under-sampling)"

train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# ----------------------------------------------------------------------------
# Autoencoder (Random Under-sampling)
X_train = X_train_rus
y_train = y_train_rus

name = "Autoencoder (" + dataset_name + " - Random Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# ----------------------------------------------------------------------------
# SVM (NearMiss)
X_train = X_train_nearmiss
y_train = y_train_nearmiss

name = "SVM (" + dataset_name + " - Near-Miss Under-sampling)"

train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# Isolation Forest (NearMiss)
X_train = X_train_nearmiss
name = "Isolation Forest (" + dataset_name + " - Near-Miss Under-sampling)"
train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# # ----------------------------------------------------------------------------
# # Autoencoder (NearMiss)
X_train = X_train_nearmiss
y_train = y_train_nearmiss

# name = "Autoencoder (Credit Card Dataset - Near-Miss Under-sampling)"
name = "Autoencoder (PaySim Dataset - Near-Miss Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# ---------------------------------------------------------------------------- 
# SVM (TomekLinks)
X_train = X_train_tomek
y_train = y_train_tomek

name = "SVM (" + dataset_name + " - Tomek Links Under-sampling)"

train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# Isolation Forest (TomekLinks)
X_train = X_train_tomek
name = "Isolation Forest (" + dataset_name + " - Tomek Links Under-sampling)"
train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# ----------------------------------------------------------------------------
# Autoencoder (TomekLinks)
X_train = X_train_tomek
y_train = y_train_tomek

name = "Autoencoder (" + dataset_name + " - Tomek Links Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# ---------------------------------------------------------------------------- 
# SVM (ENN)
X_train = X_train_enn
y_train = y_train_enn

name = "SVM (" + dataset_name + " - ENN Under-sampling)"

train_SVM(X_train, y_train, X_test, y_test, name)
print("Complete")

# ----------------------------------------------------------------------------
# Isolation Forest (ENN)
X_train = X_train_enn
name = "Isolation Forest (" + dataset_name + " - ENN Under-sampling)"
train_IF(X_train, X_test, y_test, name)

# Isolation Forest with Warm Start On New Trees
train_IF_warm_trees(X_train, X_test, y_test, name)

# ----------------------------------------------------------------------------
# Autoencoder (ENN)
X_train = X_train_enn
y_train = y_train_enn

name = "Autoencoder (" + dataset_name + " - ENN Under-sampling)"

train_autoencoder(X_train, y_train, X_test, y_test, shape_size, name)

# # ---------------------------------------------------------------------------- 
# # Correlation Heatmaps
# # Random Under-sampling
# df_rus = X_train_rus
# df_rus[target] = y_train_rus

# corr = df_rus.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Random Under-sampling)"
# create_corr(corr, corr_title)

# # NearMiss
# df_nearmiss = X_train_nearmiss
# df_nearmiss[target] = y_train_nearmiss

# corr = df_nearmiss.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Near-Miss Under-sampling)"
# create_corr(corr, corr_title)

# # Tomek Links
# df_tomek = X_train_tomek
# df_tomek[target] = y_train_tomek

# corr = df_tomek.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - Tomek Links Under-sampling)"
# create_corr(corr, corr_title)

# # ENN
# df_enn = X_train_enn
# df_enn[target] = y_train_enn

# corr = df_enn.corr()
# corr_title = "Correlation Heatmap (" + dataset_name + " - ENN Under-sampling)"
# create_corr(corr, corr_title)
