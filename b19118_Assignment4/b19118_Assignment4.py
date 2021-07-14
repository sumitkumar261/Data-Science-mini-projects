#Sumit Kumar
#Roll no - B19118
#Mobile no - 7549233722

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math


def KNN_classifier(X_train, X_test, X_label_train, X_label_test):
    accuracy_list = []
    for i in range(3):
        neigh = KNeighborsClassifier(n_neighbors=2*i + 1)#KNN classifier
        neigh.fit(X_train,X_label_train)#fitting the training dataset into KNN
        y_pred = neigh.predict(X_test)#prediction for test dataset
        print("\nThe confusion matrix when the value of K is ",2*i+1," , is :- ")
        print(confusion_matrix(y_pred,X_label_test))#calculating confusion matrix
        print("\nThe accuracy score when the value of K is ",2*i+1," , is :- ")
        accuracy = metrics.accuracy_score(test['class'],y_pred)#calculating the accuracy
        print('%.3f'%(accuracy*100),"%")
        accuracy_list.append(accuracy*100)
    max_accuracy = accuracy_list[0]
    max_accuracy_K = 1
    for i in range(1,3):
        if accuracy_list[i]>max_accuracy:
            max_accuracy = accuracy_list[i]
            max_accuracy_K = 2*i +1
    print("The value of K for which the accuracy is maximum is:- ",max_accuracy_K)
    return max_accuracy
    
    
#Question 1
print("---------------------------Question 1-------------------------------")
df = pd.read_csv('seismic_bumps1.csv')#importing the dataframe
X = df.drop(['nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89'],axis=1)#all input features


X0 = X[X['class']==0]#input features for class = 0
X_label0 = X0['class']
X0 = X0.drop(['class'],axis = 1)


X1 = X[X['class']==1]#input features for class = 0
X_label1 = X1['class']
X1 = X1.drop(['class'],axis = 1)

#separating 70% for train and 30% for test from the dataset whose class is 0
[X_train0, X_test0, X_label_train0, X_label_test0] = train_test_split(X0,X_label0,test_size=0.3,random_state=42,shuffle=True)
#separating 70% for train and 30% for test from the dataset whose class is 1
[X_train1, X_test1, X_label_train1, X_label_test1] = train_test_split(X1,X_label1,test_size=0.3,random_state=42,shuffle=True)
X_train = pd.concat([X_train0,X_train1],axis=0)#joining the 70% of class 0 and 70% of class 1 (input features)
X_test = pd.concat([X_test0,X_test1],axis=0)#joining the 70% of class 0 and 70% of class 1 (output features)
X_label_train = pd.concat([X_label_train0,X_label_train1],axis=0)#joining the 30% of class 0 and 30% of class 1 (input features)
X_label_test = pd.concat([X_label_test0,X_label_test1],axis=0)#joining the 30% of class 0 and 30% of class 1 (output features)



train = pd.concat([X_train,X_label_train],axis=1)#combining train datsets
test = pd.concat([X_test,X_label_test],axis=1)#combining test datasets
train.to_csv('seismic-bumps-train.csv')#creating the csv file of training dataset
test.to_csv('seismic-bumps-test.csv')#creating the csv file of test dataset
max_accuracy1 = KNN_classifier(X_train, X_test, X_label_train, X_label_test)


#Question 2
print("---------------------------Question 2-------------------------------")
print("After normalising the train and the test set :--  ")   
X_train_normalised = (X_train-X_train.min())/(X_train.max()-X_train.min())#normalising the train set
X_test_normalised = (X_test- X_train.min())/(X_train.max()-X_train.min())#normalising the test set
train_normalised = pd.concat([X_train_normalised,X_label_train],axis = 1)#combining the normalised train dataset
test_normalised = pd.concat([X_train_normalised,X_label_test],axis = 1)#combining the normalised test dataset
train_normalised.to_csv('seismic-bumps-train-normalised.csv')#creating the csv file of normalised training dataset
test_normalised.to_csv('seismic-bumps-test_normalised.csv')#creating the csv file of normalised test dataset  
max_accuracy2 = KNN_classifier(X_train_normalised, X_test_normalised, X_label_train, X_label_test)

#Question 3
print("---------------------------Question 3-------------------------------")
columns_list = list(X.columns)

mean_list0 = train[train['class']==0]
mean_list0 = mean_list0.drop(['class'],axis=1)
mean_list0 = list(mean_list0.mean())#calculating the mean of attributes for class = 0


mean_list1 = train[train['class']==1]
mean_list1 = mean_list1.drop(['class'],axis=1)
mean_list1 = list(mean_list1.mean())#calculating the mean of attributes for class = 1

cov_class0 = train[train['class']==0]
cov_class0 = cov_class0.drop(['class'],axis = 1)
cov_class0 = cov_class0.cov()#calculating the covariance for class = 0
pd.set_option("display.max_rows", None, "display.max_columns", None)

mod_cov0 = np.linalg.det(cov_class0)#calculating the determinant for class = 0

cov_class1 = train[train['class']==1]
cov_class1 = cov_class1.drop(['class'],axis = 1)
cov_class1 = cov_class1.cov()#calculating the covariance for class = 1

mod_cov1 = np.linalg.det(cov_class1)#calculating the determinant for class = 1

Prior_0 = (train[train['class']==0]).shape[0]/X_train.shape[0]#calculating the value of prior for class 0
Prior_1 = (train[train['class']==1]).shape[0]/X_train.shape[0]#calculating the value of prior for class 0

no_row_test = X_test.shape[0]#no of examples in test dataset
no_col_test = X_test.shape[1]#dimensions in test dataset
y_test_pred = []#empty list for prediction of bayes classifier

for i in range(no_row_test):
    #calculation of Mahalanobis distance for class 0
    Mahalanobis_distance0 = np.dot(np.dot(X_test.iloc[[i]]-mean_list0,np.linalg.inv(cov_class0)),np.transpose(X_test.iloc[[i]]-mean_list0))
    #calculation of Mahalanobis distance for class 1
    Mahalanobis_distance1 = np.dot(np.dot(X_test.iloc[[i]]-mean_list1,np.linalg.inv(cov_class1)),np.transpose(X_test.iloc[[i]]-mean_list1))
    #calculation of likelihood for class 0
    P_like0 = ((1/(((2*math.pi**(int(no_col_test/2)))*(mod_cov0**0.5))))*(math.exp(-Mahalanobis_distance0[0][0]/2)))*Prior_0
    #calculation of likelihood for class 1
    P_like1 = ((1/(((2*math.pi**(int(no_col_test/2)))*(mod_cov1**0.5))))*(math.exp(-Mahalanobis_distance1[0][0]/2)))*Prior_1
    if(P_like0>P_like1):#checking for which class likelihood is higher
        y_test_pred.append(0)
    else:
        y_test_pred.append(1)
    
print("\nThe accuracy of the bayes classifier is :-")
max_accuracy3 = metrics.accuracy_score(X_label_test,y_test_pred)
print(max_accuracy3*100," %")#accuracy of bayes classifier
print("\nThe confusion matrix of the bayes classifier is :-")
print(confusion_matrix(y_test_pred,X_label_test))#confusion matrix of bayes classifier

#Question 4
print("---------------------------Question 4-------------------------------")
print("\nAccuracy of different methods :-")
print("\nKNN classifier   |   KNN classifier after normalisation   |   Bayes classifier   ")
print('%.3f'%max_accuracy1,"%         |            ",'%.3f'%max_accuracy2,"%                   |   ",'%.3f'%(max_accuracy3*100),"%")
print("\nFrom the above observation, it appears that KNN classifier is the best method.")






