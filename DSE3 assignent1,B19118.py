#Sumit Kumar
#B19118
#Mobile no - 7549233722

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('landslide_data3.csv')#loading csv dataframe
l1 = list(df.columns)#extracting the names of the column

l2 = []
for i in l1:#creating list of particular columns of dataframe
    l2.append(df[str(i)])
#each element of the list is a particular column of the dataframe

def mean(X):
    sum1 = np.sum(X)
    mean = sum1/len(df) #len(df) gives number of rows excluding the first row
    return mean

def median(X):
    arranged = np.sort(X)
    if len(df)%2!=0:#if N is odd then median is (N+1)/2 th term
        mid = int((len(df)+1)/2)
        return arranged[mid-1]
    else: 
        mid = int(len(df)/2)#if n is even median is average of N/2 and N/2+1 th term
        s = (arranged[mid-1] + arranged[mid])/2
        return s
        
        
def mode(X):#finding the value whose frequency is high
    s = list(set(X))
    s.sort()
    l = (np.sort(X)).tolist()
    l1=[]
    for i in s:
        p=l.count(i)
        l1.append(p)
    maxi = max(l1)
    s1 = l1.index(maxi)
    return s[s1]
                
def maximum(X):
    s = list(set(X))
    s.sort()
    return s[-1]#last element after sorting gives maximum value

def minimum(X):
    s = list(set(X))
    s.sort()
    return s[0]#first element after sorting gives minimum value

def std(X):
    E = mean(X)
    l = sum((E-X)**2)
    return (l/(len(df)-1))**0.5#finding standard deviation 

def scatter(X,Y,xlabel,ylabel):#scatter plot
    plt.scatter(X,Y)
    plt.xlabel(xlabel+' --->')
    plt.ylabel(ylabel+' --->')
    plt.title('scatter plot between '+xlabel+' and '+ylabel)
    plt.show()


def coeffcorr(X,Y):
    Ex = mean(X)
    Ey = mean(Y)
    cov = sum((X-Ex)*(Y-Ey))/len(df)#finding correlation coefficient using formula of covariance and standard deviation
    return cov/(std(X)*std(Y))
    
def hist(X,title,xlabel):#histogram plot
    fig,ax = plt.subplots(1,1)
    plt.rcParams["figure.figsize"] = (10,7)#for increasing the size of the graph
    ax.hist(X,bins=25,color='#000000', edgecolor='#FFFFFF')#settig bins and color
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of stations")
    ax.set_title(title)
    plt.show()

#Question 1
print("Question 1 :--")
for j in range(len(l1)):#iterating through the list excluding dates and stationid
    if l1[j]!='dates' and l1[j]!='stationid':
        print('For '+str(l1[j])+' :')
        print("Mean is ",mean(l2[j]))
        print("Median is ",median(l2[j]))
        print("Mode is ",mode(l2[j]))
        print("Maximum is ",maximum(l2[j]))
        print("Minimum is ",minimum(l2[j]))
        print("Standard Deviation is ",std(l2[j]))
        print('\n')
        

#Question 2
print("Question 2(a) :--\n")
for j in range(len(l1)):#iterating through the list excluding dates, stationid and rain
    if l1[j]!='dates' and l1[j]!='stationid' and l1[j]!='rain':
        scatter(df['rain'],l2[j],'rain',l1[j])
    
print("Question 2(b) :--\n")
for j in range(len(l1)):#iterating through the list excluding dates, stationid and temperature
    if l1[j]!='dates' and l1[j]!='stationid' and l1[j]!='temperature':
        scatter(df['temperature'],l2[j],'temperature',l1[j])
        
#Question 3       
print("Question 3(a) :--\n")
for j in range(len(l1)):#iterating through the list excluding dates, stationid and rain
    if l1[j]!='dates' and l1[j]!='stationid' and l1[j]!='rain':
        print("correlation coefficient between rain and "+l1[j]+" is ",coeffcorr(df['rain'],l2[j]))
        print('\n')
        
print("Question 3(b) :--\n")
for j in range(len(l1)):#iterating through the list excluding dates, stationid and temperature
    if l1[j]!='dates' and l1[j]!='stationid' and l1[j]!='temperature':
        print("correlation coefficient between temperature and "+l1[j]+" is ",coeffcorr(df['temperature'],l2[j]))
        print('\n')

#Question 4
print("Question 4 :--\n")
hist(df['rain'],"histogram of rain",'rain')
hist(df['moisture'],"histogram of moisture",'moisture')

#Question 5
print("Question 5 :--\n")
l = df.groupby('stationid')#grouping all unique stationid
unique = df['stationid'].unique()
for i in unique:    
    df1 = l.get_group(i)#histogram plot of rain for each unique stationid
    hist(df1['rain'],'Histogram for stationid '+i,'rain')


#Question 6
print("Question 6 :--\n")
plt.title("Boxplot of rain")
plt.yscale('log')#improving aesthetics of the boxplot
plt.boxplot(df['rain'])
plt.show()

plt.title("Boxplot of moisture")
plt.boxplot(df['moisture'])
plt.show()
