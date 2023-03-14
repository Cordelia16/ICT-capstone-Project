# Importing libraries
import pandas as pd
import numpy as np

# Scikit-learn library: For SVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

# Matplotlib library to plot the charts
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Library for the statistic data visualisation
import seaborn

# %matplotlib inline

data = pd.read_csv('Data/creditcard.csv')  # Reading the file .csv
df = pd.DataFrame(data)  # Converting data to Panda DataFrame
df.describe()

df_fraud = df[df['Class'] == 1]  # Recovery of fraud data
plt.figure(figsize=(15, 10))
plt.scatter(df_fraud['Time'], df_fraud['Amount'])  # Display fraud amounts according to their time
plt.title('Scratter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0, 175000])
plt.ylim([0, 2500])
plt.show()

df_corr = df.corr()
plt.figure(figsize=(15, 10))
seaborn.heatmap(df_corr, cmap="YlGnBu")  # Displaying the Heatmap
seaborn.set(font_scale=2, style='white')

plt.title('Heatmap correlation')
plt.show()

rank = df_corr['Class']  # Retrieving the correlation coefficients per feature in relation to the feature class
df_rank = pd.DataFrame(rank)
df_rank = np.abs(df_rank).sort_values(by='Class', ascending=False)
df_rank.dropna(inplace=True)  # Removing Missing Data (not a number)

# We separate ours data in two groups : a train dataset and a test dataset

# First we build our train dataset
df_train_all = df[0:150000]  # We cut in two the original dataset
df_train_1 = df_train_all[df_train_all['Class'] == 1]  # We separate the data which are the frauds and the no frauds
df_train_0 = df_train_all[df_train_all['Class'] == 0]

df_sample = df_train_0.sample(300)
df_train = pd.concat([df_train_1, df_sample])  # We gather the frauds with the no frauds.
df_train = df_train.sample(frac=1)  # Then we mix our dataset

X_train = df_train.drop(['Time', 'Class'], axis=1)  # We drop the features Time (useless), and the Class (label)
y_train = df_train['Class']  # We create our label
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

df_test_all = df[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'], axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)

X_train_rank = df_train[df_rank.index[1:11]]  # We take the first ten ranked features
X_train_rank = np.asarray(X_train_rank)

X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)

# Confusion Matrix
class_names = np.array(['0', '1'])


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Model Selection
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Testing the Model
prediction_SVM_all = classifier.predict(X_test_all)

cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm, class_names)

print('\n---------------------- TRAINING A ----------------------\n')
print('Our criterion give a result of '
      + str(((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))

print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1] / (cm[1][1] + cm[1][0])))
print("the accuracy is : " + str((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))

# Model Rank
classifier.fit(X_train_rank, y_train)
prediction_SVM = classifier.predict(X_test_all_rank)

cm = confusion_matrix(y_test_all, prediction_SVM)
plot_confusion_matrix(cm, class_names)

print('\n--------------------- TESTING A ---------------------\n')
print('Our criterion give a result of '
      + str(((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))

print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1] / (cm[1][1] + cm[1][0])))
print("the accuracy is : " + str((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))

# Re-balanced
classifier_b = svm.SVC(kernel='linear', class_weight={0: 0.60, 1: 0.40})
classifier_b.fit(X_train, y_train)

# Testing Model B
prediction_SVM_b_all = classifier_b.predict(X_test_all)

cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
plot_confusion_matrix(cm, class_names)

print('\n---------------------- TRAINING B ----------------------\n')
print('Our criterion give a result of '
      + str(((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))

print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1] / (cm[1][1] + cm[1][0])))
print("the accuracy is : " + str((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))

# Models Rank
classifier_b.fit(X_train_rank, y_train)
prediction_SVM = classifier_b.predict(X_test_all_rank)

cm = confusion_matrix(y_test_all, prediction_SVM)
plot_confusion_matrix(cm, class_names)

print('\n--------------------- TESTING B ---------------------\n')
print('Our criterion give a result of '
      + str(((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))

print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1] / (cm[1][1] + cm[1][0])))
print("the accuracy is : " + str((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))
