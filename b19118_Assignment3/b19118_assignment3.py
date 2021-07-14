#Sumit Kumar
#B19118
#Mobile no - 7549233722

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Question1 : 

df = pd.read_csv('landslide_data3.csv')
column_name = list(df.columns)#extracting the names of the column
dataframes_list = []#list of dataframes
for i in range(len(column_name)):
    dataframes_list.append(df[column_name[i]])
df_last7 = df.drop(columns=['dates','stationid'])

def Iquartile(x):#defining function for calculating first and third quartile and interquartile range and outliers
    Q1 = np.quantile(x,0.25)#First quartile
    Q3 = np.quantile(x,0.75)#third quartile
    IQR = Q3 - Q1#interquartile range
    out = []
    for j in x:
        if j<Q1-1.5*IQR or j>Q3+1.5*IQR:#calculating outliers
            out.append(j)
    return out

def maxminormalize(x):
    print("For "+x+" before normalization :-")
    print("Minimum is :- ",'%.3f'%df_last7[x].min())
    print("Maximum is :- ",'%.3f'%df_last7[x].max(),"\n")
    df_last7[x] = (df_last7[x]-df_last7[x].min())/(df_last7[x].max()-df_last7[x].min())*(9-3) + 3#maximum - minimum normalization
    print("For "+x+" after normalization :-")
    print("Minimum is :- ",'%.3f'%df_last7[x].min())
    print("Maximum is :- ",'%.3f'%df_last7[x].max(),"\n")

def standardization(x):
    print("For "+x+" before standardization :-")
    print("Mean is :- ",'%.3f'%df_last7[x].mean())
    print("standard deviation is :- ",'%.3f'%df_last7[x].std(),"\n")
    df_last7[x] = (df_last7[x] - df_last7[x].mean())/df_last7[x].std()#standardization
    print("For "+x+" after standardization :-")
    print("Mean is :- ",'%.3f'%df_last7[x].mean())
    print("standard deviation is :- ",'%.3f'%df_last7[x].std(),"\n")
      
print("-----------------------Question 1(a)--------------------------")    
for i in column_name:
    if i!='dates' and i!='stationid':
        z = Iquartile(df[i])#calculating outliers for each attribute 
        df_last7[i] = df_last7[i].replace(to_replace = z, value = df_last7[i].median())#replacing the attributes with median
        maxminormalize(i)

list_col = []#list of column names except dates and stationid
print("-----------------------Question 1(b)--------------------------")  
for i in column_name:
    if i!='dates' and i!='stationid':
        list_col.append(i)
        standardization(i)#standardizing the data
        
#Question 2:-
print("-----------------------Question 2(a)--------------------------")  
E_mean = [0, 0]#given mean
cov = [[6.84806467, 7.63444163], [7.63444163, 13.02074623]] #given covariance
rand_data = np.random.multivariate_normal(E_mean,cov, 1000).T#generating the random data

plt.scatter(rand_data[0],rand_data[1],marker = 'x')
plt.axis('equal')#plotting the random data
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("scatter plot and eigen vector")
plt.show()

#cov_matrix = np.cov(rand_data)#finding the covariance
w, eigvec = np.linalg.eig(cov)#finding the eigen value and the eigen vector
print("-----------------------Question 2(b)--------------------------")  
print("\nEigen values are :- ")
print(w)
print("\nEigen vectors are :-")
print(eigvec)
print("The scatter plot of the data points and the eigen vectors :-")
plt.scatter(rand_data[0],rand_data[1],marker = 'x')#plotting the random data
plt.quiver(0,0,eigvec[0][0],eigvec[1][0],angles="xy",scale=7,color='r')#plotting the vector
plt.quiver(0,0,eigvec[0][1],eigvec[1][1],angles="xy",scale=3)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("scatter plot")
plt.axis('equal')
plt.show()

print("-----------------------Question 2(c)--------------------------")  
plt.scatter(rand_data[0],rand_data[1],marker = 'x')#plotting the scatter
plt.quiver(0,0,eigvec[0][0],eigvec[1][0],angles="xy",scale=7,color='r')#plotting the vector
plt.quiver(0,0,eigvec[0][1],eigvec[1][1],angles="xy",scale=3)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("The plot after projection on the 1st eigen vector")
projection = np.dot(rand_data.T,eigvec)#finding the projection
plt.scatter(projection[:,0]*eigvec[0][0],projection[:,0]*eigvec[1][0],color='magenta',marker='x')#plotting the projection on one eigen vector
plt.axis('equal')
plt.show()


plt.scatter(rand_data[0],rand_data[1],marker = 'x')#plotting the scatter
plt.quiver(0,0,eigvec[0][0],eigvec[1][0],angles="xy",scale=7,color='r')#plotting the vector
plt.quiver(0,0,eigvec[0][1],eigvec[1][1],angles="xy",scale=3)

plt.scatter(projection[:,1]*eigvec[0][1],projection[:,1]*eigvec[1][1],color='magenta',marker='x')#plotting the projection on second eigen vector
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("The plot after projection on the 2nd eigen vector")
plt.axis('equal')
plt.show()
print("-----------------------Question 2(d)--------------------------")  
recovered_data = np.dot(projection,eigvec.T).T#recovering the data from the projection and the eigen vector
plt.scatter(recovered_data[0],recovered_data[1],marker='x')#plotting the recovered data
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("plot of the recovered data")
plt.show()
RMSE = ((recovered_data-rand_data)**2).sum()/len(recovered_data)#calculating the root mean square error
print("Root mean square error is : - ",RMSE**0.5)

#Question 3:-
print("-----------------------Question 3(a)--------------------------")  
eigval1,eigvec1 = np.linalg.eig(np.cov(df_last7.T))#finding the eigen vectors after standardizartion
eigval1 = sorted(eigval1, reverse=True)#sorting the eigen values in decreasing order

pca1 = PCA(n_components=2)#principal component analysis
principalComponents = pca1.fit_transform(df_last7)#reducing the dimension
print("The variance of the 1st data is :-",np.var(principalComponents.T[0]))
print("The eigen value corresponding to the 1st eigen vector :-",eigval1[0])

print("\nThe variance of the 2nd data is :-",np.var(principalComponents.T[1]))
print("The eigen value corresponding to the 2nd eigen vector :-",eigval1[1])
print("-----------------------Question 3(b)--------------------------")  
plt.scatter(principalComponents.T[0],principalComponents.T[1])#plotting the data after reduction of dimension
plt.ylabel("Principal component 2")
plt.xlabel("Principal component 1")
plt.title("scatter plot after reducing the dimension")
plt.show()

new_Rmse = []#list of RMSE value
plt.bar(range(1,8),eigval1,color = 'r')
plt.plot(range(1,8),eigval1)
plt.scatter(range(1,8),eigval1)
plt.ylabel("eigen values")
plt.xlabel("index")
plt.title("plot of eigen values")
plt.show()
for i in range(1,8):
    pca = PCA(n_components=i)#principal component analysis
    principalComponents1 = pca.fit_transform(df_last7)#reducing the dimension
    recons_data=pca.inverse_transform(principalComponents1)#reconstructing the data
    rmse = ((df_last7-recons_data)**2).sum().sum()/len(recons_data)
    new_Rmse.append(rmse**0.5)
print("-----------------------Question 3(c)--------------------------")  

plt.ylabel("RMSE")
plt.xlabel("value of L")
plt.title("plot of the RMSE value")
plt.plot(range(1,8),new_Rmse)#plotting the line plot of RMSE value
plt.scatter(range(1,8),new_Rmse)
plt.show()



  
    