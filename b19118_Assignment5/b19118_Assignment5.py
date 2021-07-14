#Sumit Kumar
#Roll no - B19118
#Mobile no - 7549233722

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error    
import sklearn

#PART A
#Question 1
print("-----PART A-------")
print("---------------------------Question 1-------------------------------")

train = pd.read_csv('seismic-bumps-train.csv')#importing the train data
test = pd.read_csv('seismic-bumps-test.csv')#importing the test data

X_train0 = train[train['class']==0] 
X_train1 = train[train['class']==1]
X_train0 = X_train0.drop(['class'],axis = 1)#training input set for class 0
X_train1 = X_train1.drop(['class'],axis = 1)#training input set for class 1

X_test = test.drop(['class'],axis = 1)#test input set
X_label_test = test['class']#actual test output set

Bayes_acc = 0
for i in range(1,5):
    gauss_pred = []
    GMM0 = GaussianMixture(n_components=2**i, covariance_type='full')#gaussian mixture
    GMM0.fit(X_train0)#fitting train data on GMM
    score_samples_0 = GMM0.score_samples(X_test)#weighted log probabilities for class 0

    GMM1 = GaussianMixture(n_components=2**i, covariance_type='full')#gaussian mixture
    GMM1.fit(X_train1)#fitting test data on GMM
    score_samples_1 = GMM1.score_samples(X_test)#weighted log probabilities for class 1
    
    for j in range(len(score_samples_1)):
        if score_samples_0[j]>score_samples_1[j]:
            gauss_pred.append(0)
        else:
            gauss_pred.append(1)
    print("\nFor Q = ",2**i)
    print("\nConfusion matrix is : ")
    print(confusion_matrix(gauss_pred,X_label_test))#calculating confusion matrix 
    accuracy = metrics.accuracy_score(X_label_test,gauss_pred)#calculating the accuracy
    print("Accuracy is : ",'%.3f'%(accuracy*100),"%")
    Bayes_acc = '%.3f'%(accuracy*100)

#Question 2
print("\nQuestion 2")
print("\nAccuracy of different methods :-")
print("\nKNN classifier   |   KNN classifier after normalisation   |   Bayes classifier   | ")
print(93.170,"%          |            ",92.912,"%                   |   ",87.500,"%","           |  ")
print("\n\nBayes classifier using GMM    |")
print(Bayes_acc,"%                      |")
print("\nFrom the above observation, it appears that KNN classifier is the best method.")
    
    
    
#PART B 
#Question 1(a)
print("-----PART B-------")
print("---------------------------Question 1-------------------------------")
print("\nQuestion 1(a)")
df1 = pd.read_csv('atmosphere_data.csv')#importing the dataframe
[train_atm, test_atm] = train_test_split(df1,test_size=0.3,random_state=42,shuffle=True)

xtrain = np.asarray(train_atm['pressure'])
xtrain = xtrain[:,np.newaxis]#training input data or 'pressure' attribute

ytrain = np.asarray(train_atm['temperature'])
ytrain = ytrain[:,np.newaxis]#training output data or 'temperature' attribute

train_atm.to_csv('atmosphere-train.csv')#creating the csv file of training dataset
test_atm.to_csv('atmosphere-test.csv')#creating the csv file of test dataset
regr = LinearRegression()#linear regression
regr.fit(xtrain,ytrain) #fitting the training data into linear regression model

y_pred_train = regr.predict(xtrain)#predicting the training data
plt.figure(figsize = (12,8))
plt.scatter(train_atm['pressure'], train_atm['temperature'])#plotting scatter plot of training data
plt.plot(xtrain, y_pred_train, color ='k')#plotting the best fit line
plt.ylabel("Temperature")
plt.xlabel("pressure")
plt.title("Linear regression fit line on training set")
plt.show()

#Question 1(b)

pred_accuracy_train = np.sqrt(mean_squared_error(ytrain, y_pred_train))#calculating the RMSE error on train set
print("\nQuestion 1(b)\n")
print("\nThe prediction accuracy on the training data is :-  ",pred_accuracy_train)

#Question 1(c)
print("\nQuestion 1(c)\n")
xtest = np.asarray(test_atm['pressure'])
xtest = xtest[:,np.newaxis]#test set input data

ytest = np.asarray(test_atm['temperature'])
ytest = ytest[:,np.newaxis]#test set actual output data


y_pred_test = regr.predict(xtest)#predicting the output for test set

pred_accuracy_test = np.sqrt(mean_squared_error(ytest, y_pred_test))#predicting the RMSE for test set
print("\nThe prediction accuracy on the test data is :-  ",pred_accuracy_test)

#Question 1(d)
print("\nQuestion 1(d)\n")
plt.figure(figsize = (12,8))
plt.scatter(ytest, y_pred_test,color = 'r')#scatter plot of actual temperature vs predicted temperature
plt.ylabel("Predicted Temperature")
plt.xlabel("Actual temperature")
plt.title("scatter plot of actual temperature vs predicted temperature")
plt.show()


#Question 2
print("---------------------------Question 2-------------------------------")
RMSE_train = []#list for RMSE value of train set
RMSE_test = []#list for RMSE value of test set
train_best_pred = []#best prediction on training set
test_best_pred = []#best prediction on test set

#polynomial regression

for i in range(2,6):
    poly = PolynomialFeatures(degree=i)#polynomial feature of degree = i
    X_ = poly.fit_transform(xtrain)#transforming the train data into polynomial
    regr.fit(X_,ytrain)#fitting the transformed data into linear regression model
    y_pred_train_poly = regr.predict(X_)#prediction for training set
    pred_accuracy_train_poly = np.sqrt(mean_squared_error(ytrain, y_pred_train_poly))#predicting the RMSE value for train set
    print("For p = ",i)
    print("The prediction accuracy on the train data is :-  ",pred_accuracy_train_poly)
    
    
    train_best_pred.append(y_pred_train_poly)

    x_test = poly.fit_transform(xtest)#transforming the test set into polynomial
    y_pred_test_poly = regr.predict(x_test)#prediction for test set
    pred_accuracy_test_poly = np.sqrt(mean_squared_error(ytest, y_pred_test_poly))#predicting the RMSE value for test  set
    print("The prediction accuracy on the test data is :-  ",pred_accuracy_test_poly)
    
    test_best_pred.append(y_pred_test_poly)

    RMSE_train.append(pred_accuracy_train_poly)
    RMSE_test.append(pred_accuracy_test_poly)
p_train = 2
p_test = 2
for i in [2,3,4,5]:
    x1 = min(RMSE_train)
    x2 = min(RMSE_test)
    if(RMSE_train[i-2]==x1):
        p_train = i
    if(RMSE_test[i-2]==x2):
        p_test = i
    
#Question 2(a)
p_val = [2,3,4,5]
print("Question 2(a)\n")
plt.figure(figsize = (12,8))
plt.plot(p_val,RMSE_train)
plt.scatter(p_val,RMSE_train,color = 'r')
plt.bar(p_val,RMSE_train,color = 'k')#bar graph of RMSE for training set
plt.ylabel("RMSE")
plt.xlabel("value of p")
plt.title("bar graph of RMSE for training set")
plt.show()
 
#Question 2(b)
print("Question 2(b)\n")  
plt.figure(figsize = (12,8)) 
plt.plot(p_val,RMSE_test)
plt.scatter(p_val,RMSE_test,color = 'r')
plt.bar(p_val,RMSE_test,color = 'k')#bar graph of RMSE for test set
plt.ylabel("RMSE")
plt.xlabel("value of p")
plt.title("bar graph of RMSE for test set")
plt.show()

#Question 2(c)
print("Question 2(c)\n")
print("For p = ",p_test," , the RMSE value is lesser for the training dataset")
print("scatter plot and polynomial fit for p = ",p_test," for training set : ")
plt.figure(figsize = (12,8))
plt.scatter(xtrain,ytrain)#scatter plot of train data
plt.scatter(xtrain,train_best_pred[p_test-2],color = 'k')#scatter plot of prediction data in training set for best degree of polynomial
plt.ylabel("Temperature")
plt.xlabel("pressure")
plt.title("polynomial regression fit curve on training set p = 5")
plt.show()


#Question 2(d)
print("Question 2(d)\n")
print("For p = ",p_test," , the RMSE value is lesser for the test dataset")
print("scatter plot and polynomial fit for p = ",p_test," for test set : ")
plt.figure(figsize = (12,8))
plt.scatter(ytest,test_best_pred[p_test-2],color = 'r')#scatter plot of test data
plt.ylabel("Actual Temperature")
plt.xlabel("Predicted temperature")
plt.title("Scatter plot of Actual temperature and predicted temperature")
plt.show()

print(sklearn.__version__)



