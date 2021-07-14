#Sumit Kumar
#B19118
#Mobile no - 7549233722

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('pima_indians_diabetes_miss.csv')#loading csv dataframe containing missing values
df_orig = pd.read_csv('pima_indians_diabetes_original.csv')#loading csv dataframe containing original values

def meanmedmode(x,y,i):#defining finction for calculating mean median mode and standard deviation
    print("For attribute "+i+" (after replacing the missing values) :- ")
    print("Mean is :- ",x.mean())
    print("Mode is :- ",x.mode()[0])
    print("Median is :- ",x.median())
    print("Standard deviation is :- ",x.std(),"\n")
    print("For attribute "+i+" (from original data) :- ")
    print("Mean is :- ",y.mean())
    print("Mode is :- ",y.mode()[0])
    print("Median is :- ",y.median())
    print("Standard deviation is :- ",y.std(),"\n")
    
def Iquartile(x,i):#defining function for calculating first and third quartile and interquartile range
    print("The quartile Q1 of the attribute "+i+" is :- ",np.quantile(x,0.25),"\n")
    print("The quartile Q3 of the attribute "+i+" is :- ",np.quantile(x,0.75),"\n")
    print("The interquartile range Q3-Q1 of the attribute "+i+" is :- ",np.quantile(x,0.75)-np.quantile(x,0.25),"\n")
    Q1 = np.quantile(x,0.25)#First quartile
    Q3 = np.quantile(x,0.75)#third quartile
    IQR = Q3 - Q1#interquartile range
    print("The Outliers of "+i+" are :- ")
    out = []
    for j in x:
        if j<Q1-1.5*IQR or j>Q3+1.5*IQR:#calculating outliers
            print(j,end=", ")
            out.append(j)
    return out#returning the list containing outliers
    

#Question 1:-
print("-------------------------Question 1:----------------------------------------")
#loading csv dataframe
l1 = list(df.columns)#extracting the names of the column
l2 = []
for i in range(len(l1)):
    l2.append(df[l1[i]].isnull().sum())#appending the number of missing values in a list

plt.bar(l1,l2)#plotting attributes vs missing values
plt.title('attribute vs number of misising values')#setting title of the plot
plt.ylabel('number of missing values')#setting ylabel
plt.xlabel('attribute names')#setting xlabel
plt.show()

#Question 2(a):- 
print("\n-----------------------Question 2(a):--------------------------------------")
df1 = df[df.isnull().sum(axis=1)<3]#dataframe having less than 3 missing values in a tuple
df2 = df[df.isnull().sum(axis=1)>=3]#dataframe having more than or equal to 3 missing values in a tuple
missingtuples =len(df2.axes[0])
print("Total number of tuples deleted :- ", missingtuples)
print("\nThe row numbers of the deleted tuples are :- ")
l = []
for row in df2.index:
    print(row+1, end=", ")#printing the index of the missing tuples
    l.append(row)
    
#Question 2(b):- 
print("\n-------------------------Question 2(b):-------------------------------------")
df3 = df1[df1["class"].isnull()==True]#dataframe with missing class
df_final = df1[df1["class"].isnull()==False]#dataframe after deletion of the missing class values
missingtuples =len(df3.axes[0])
print("Total number of tuples deleted :- ", missingtuples)
print("\nThe row numbers of the deleted tuples are :- ")
for row in df3.index:
    print(row+1, end=", ")#printing the index of the missing class tuples
    l.append(row)

#Question 3:- 
print("\n------------------------------Question 3 :-----------------------------")
for i in range(len(l1)):
    print("The number of missing values in attribute ",l1[i]," is :- ",df_final[l1[i]].isnull().sum()) 
    #printing total no of missing vallues in a column 
print("\nTotal number of missing values in the file is :- ",df_final.isnull().sum().sum())
#printing total no of missing vallues in the dataframe
##############################

def RMSE(x,z):#defining a function for calculation of RMSE values
    df_origx = df_orig.drop(l)
    df_new = (x-df_origx)**2
    columns = []
    RMSE_value = []
    for i in range(len(l1)):
        if df_final[l1[i]].isnull().sum()!=0:
            columns.append(l1[i])
            rmse = df_new[l1[i]].sum()/df_final[l1[i]].isnull().sum()
            RMSE_value.append(rmse**0.5)
            print("The RMSE value of the attribute "+l1[i]+" is :- ",rmse**0.5,"\n")
        else:
            RMSE_value.append(0)
            columns.append(l1[i])
            print("The RMSE value of the attribute "+l1[i]+" is :- ",0,"\n")
            
    plt.plot(columns,RMSE_value,label=z)#plotting RMSE value vs attribute name
    plt.xlabel("attributes")
    plt.ylabel("RMSE value")
    
    
    
###############################
#Question 4(a).(i)
df_final1 = df_final.fillna(df_final.mean())#filling the missing values with median of the attribute
print("\n-----------------------------------Question 4(a).(i) :------------------------------")
print("\nafter filling the missing values with the mean of the respective attributes---\n")
for i in l1:
    meanmedmode(df_final[i],df_orig[i],i)
       
#Question 4(a).(ii)
print("\n-------------------------Question 4(a).(ii) :------------------------------------")
RMSE(df_final1,'mean')

#Question 4(b).(i)
print("\n-----------------------------Question 4(b).(i) :---------------------------") 
print("\nafter filling the missing values using the interpolation method---\n")
df_final2  = df_final.fillna(df_final.interpolate())#filling the missing values using the prediction by interpolation
for i in l1:
    meanmedmode(df_final2[i],df_orig[i],i)
#Question 4(b).(ii)
print("\n----------------------------Question 4(b).(ii) :----------------------------") 
RMSE(df_final2,'interpolation')
plt.legend()
plt.show()

#Question 5(i)
print("\n------------------------Question 5(i) :---------------------") 
age = Iquartile(df_final2['Age'],'Age')
plt.boxplot(df_final2['Age'])#plotting boxplot of Age attribute
plt.xlabel("boxplot")
plt.ylabel("Age")
plt.title("boxplot of attribute Age before removing outliers")
plt.show()

BMI = Iquartile(df_final2['BMI'],'BMI')
plt.boxplot(df_final2['BMI'])#plotting boxplot of BMI attribute  
plt.xlabel("boxplot")
plt.ylabel("BMI")
plt.title("boxplot of attribute BMI before removing outliers") 
plt.show()

#Question 5(ii)
print("\n-----------------------------Question 5(ii) :-----------------------------") 

newdfage = df_final2['Age'].replace(to_replace = age, value = df_final2['Age'].median())
#replacing the outliers with the median of the attribute
plt.boxplot(newdfage)
plt.xlabel("boxplot")#boxplot after replacing
plt.ylabel("Age")
plt.title("boxplot of attribute Age after replacing outliers")
plt.show()


newdfBMI = df_final2['BMI'].replace(to_replace = BMI, value = df_final2['BMI'].median())
#replacing the outliers with the median of the attribute
plt.boxplot(newdfBMI)
plt.xlabel("boxplot")#boxplot after replacing
plt.ylabel("BMI")
plt.title("boxplot of attribute BMI after replacing outliers")
plt.show()


