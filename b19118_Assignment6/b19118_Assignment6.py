
#Sumit Kumar
#Roll no - B19118
#Mobile no - 7549233722

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

print("---------------------------Question 1-------------------------------\n")
print('Question(a)\n')
df = pd.read_csv('datasetA6_HP.csv')#importing the csv file
n_rows = df.shape[0]#number of rows


plt.figure(figsize = (12,8))
plt.plot(range(1,n_rows+1),df['HP'])#line plot between index of day and power consumed
plt.scatter(range(1,n_rows+1),df['HP'],color = 'k')
plt.xlabel('index of day')#xlabel
plt.ylabel('Power consumed in MU')#ylabel
plt.title('Line plot')#title of the plot
plt.show()


print('Question1(b)\n')
new_time1 = df['HP'][1:500,]#x(t-1) lag time data
new_time2 = df['HP'][0:499,]#x(t) given data
pr,_ = pearsonr(new_time1,new_time2)#finding the correlation between x(t-1) and x(t)
print("The pearson correlation coefficient between one day lag time sequence and given time sequence is : ",pr)


print('Question1(c)\n')
plt.figure(figsize = (12,8))
plt.scatter(new_time1,new_time2)#scatter plot between x(t-1) and x(t)
plt.xlabel("lag time")#xlabel
plt.ylabel("given time")#ylabel
plt.title("scatter plot between given time and one day lag time")#title of the plot
plt.show()



print("Question1(d)\n")
correlation_list = []#list of correlation between given time series and lag value time series
for i in range(1,8):
    given_time = df['HP'][0:500-i,]#give time series
    lag_time = df['HP'][i:500,]#lag value time series
    corr_coefficient,_ = pearsonr(lag_time,given_time)#finding correlation between them
    print('\nThe correlation coefficient if the lag value is ',i,' days is: ',corr_coefficient)
    correlation_list.append(corr_coefficient)

plt.figure(figsize = (12,8))
plt.stem(range(1,8),correlation_list,use_line_collection = True)#stem plot of correlation and lag value
plt.plot(range(1,8),correlation_list)
plt.xlabel("lag value in days")#xlabel
plt.ylabel("correlation coefficient")#ylabel
plt.title("line plot between lag value and correlation coefficient")#title of the plot
plt.show()    
   



print("Question1(e)\n")
plot_acf(df['HP'], lags=7)#inbuilt function to plot the correlation vs lag value 
plt.xlabel("lag value in days")#xlabel
plt.ylabel("correlation coefficient")#ylabel
plt.title("line plot between lag value and correlation coefficient")#title of the plot
plt.show()    
   

print("---------------------------Question 2-------------------------------\n")
X = df['HP']
train, test = X[0:len(X)-250], X[len(X)-250:]#splitting into train and test set

test_x = test[1:250,]#test set with lag time sequence x(t-1)
test_y = test[0:249,]#test set with given time sequence x(t)

#In persistence algorithm the predicted value is equal to the lag time sequence x(t-1)
#So predicted value is test_y(x(t-1))
#original value is x(t)
RMSE_error = mean_squared_error(test_y, test_x)#root mean square error between x(t-1) and x(t)
print("The RMSE between predicted power consumed for test data and original values for test data is: ",round(RMSE_error**0.5,3))  

print("---------------------------Question 3-------------------------------\n")
print("Question3(a)\n")
def Autoregression(lag):
    model = AutoReg(train, lags=lag,old_names=False)#Autoregression model with lag value 5
    model_fit = model.fit()#fitting into the AutoRegression model
    
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)#predicting the test set
    rmse = (mean_squared_error(test, predictions))**0.5#calculating the root mean square error
    print("\nRMSE for the test data for lag value ",lag,"is: ",round(rmse,4))
    return predictions

plt.figure(figsize = (12,8))
plt.scatter(test,Autoregression(5))#scatter plot of original test data vs predicted test data
plt.xlabel("Original test data")#xlabel
plt.ylabel("Predicted test data")#ylabel
plt.title("Scatter plot between original test data and predicted test data")#title of the plot
plt.show()

print("Question3(b)\n")

lag_val = [1,5,10,15,25]#given lag value


for i in lag_val:#Applying the Autoregreesion for lag values = [1,5,10,15,25]
    Autoregression(i)
    
    
print("\nQuestion3(c)\n")
lag = 1#initialize lag = 1
auto_corr,_ = pearsonr(train[0:248],train[1:249])#initialize auto_corr
T = 249#initializing T
while(abs(auto_corr)>2/(T)**0.5):#checking the condition for autocorrelation
        lag+=1#calculating the value of optimal lag
        auto_corr,_ = pearsonr(train[0:249-lag,],train[lag:249,])
        T = 250-lag

print("The optimum of heuristics value of lag is  : ",lag-1)
Autoregression(lag-1)



