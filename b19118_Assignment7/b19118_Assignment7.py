
#Sumit Kumar
#Roll no - B19118
#Mobile no - 7549233722

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy as sp
from scipy import spatial as spatial


df = pd.read_csv('mnist-tsne-train.csv')#importing the csv file of training data
df1 = pd.read_csv('mnist-tsne-test.csv')#importing the csv file of test data

train = df[df.columns[0:2]]#train data (first two columns)
train_class = df[df.columns[2:3]]#train class (last column)

test = df1[df1.columns[0:2]]#test data (first two columns)
test_class = df1[df1.columns[2:3]]#test class (last column)


print("---------------------------Question 1-------------------------------\n")

#function for finding the purity score
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true,y_pred)  
    # confusion matrix C  such that C(i,j) is the number of samples in true class i and in predicted class j.
    
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # Return cluster accuracy
    return np.sum(contingency_matrix[row_ind, col_ind]) / np.sum(contingency_matrix)

def kmeans_method(num_clusters):
     
    print("                 K = ",num_clusters,"\n")
    kmeans = KMeans(num_clusters)#Kmeans
    kmeans.fit(train)#fitting the training data in the kmeans
    centres_kmeans = kmeans.cluster_centers_#centres of the clusters
    kmeans_prediction_train = kmeans.predict(train)#predicting the class for training data
    
    plt.figure(figsize=(13,10))
    #scatter plot of the train data of different clusters after kmeans
    plt.scatter(train['dimention 1'],train['dimension 2'], c=kmeans_prediction_train, cmap='rainbow')
    plt.colorbar()
    #scatter plot of the centres of the clusters
    plt.scatter(centres_kmeans[:,:1],centres_kmeans[:,1:2],marker = '*',s = 250,color = 'k')
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title("scatter plot of the clusters for training data ")
    plt.show()

    #finding the purity score of the training data
    print("Purity Score for KNN Training Data :", purity_score(train_class, kmeans_prediction_train))
  
   
    kmeans_prediction_test = kmeans.predict(test)
    
    plt.figure(figsize=(13,10))
    #scatter plot of the test data of different clusters after kmeans
    plt.scatter(test['dimention 1'],test['dimention 2'], c=kmeans_prediction_test, cmap='rainbow')
    #scatter plot of the centres of the clusters
    plt.colorbar()
    plt.scatter(centres_kmeans[:,:1],centres_kmeans[:,1:2],marker = '*',s = 250,color = 'k')
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title("scatter plot of the clusters for test data")
    plt.show()
    
  
    #finding the purity score of the test data
    print("Purity Score for KNN Test Data :", purity_score(test_class, kmeans_prediction_test))
    print("\n----------------------------------------------------------------------------\n")
    return sum(np.min(cdist(df[df.columns[:-1]].values, centres_kmeans, 'euclidean'), axis=1)) /train_class.shape[0]

kmeans_method(10)


print("---------------------------Question 2-------------------------------\n")


def GMM_method(num_clusters):
   
    print("                 K = ",num_clusters,"\n")
    gmm = GaussianMixture(num_clusters)#GMM with n components
    gmm.fit(train)#fitting the train data into GMM
    GMM_prediction_train = gmm.predict(train)#predictin of the training data
    centres_gmm = gmm.means_#centres of the clusters
    
    plt.figure(figsize=(13,10))
    #scatter plot of the train data of different clusters after GMM
    plt.scatter(train['dimention 1'],train['dimension 2'], c=GMM_prediction_train, cmap='rainbow')
    #scatter plot of the centres of the clusters
    plt.colorbar()
    plt.scatter(centres_gmm[:,:1],centres_gmm[:,1:2],marker = '*',s = 250,color = 'k')
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title("( GMM )scatter plot of the clusters for train data ")
    plt.show()
    
  
    #finding the purity score of the train data
    print("Purity Score for GMM Training Data :", purity_score(train_class, GMM_prediction_train))
    
 
    #predicting the test data
    GMM_prediction_test = gmm.predict(test)
    
    plt.figure(figsize=(13,10))
    #scatter plot of the test data of different clusters after GMM
    plt.scatter(test['dimention 1'],test['dimention 2'], c=GMM_prediction_test, cmap='rainbow')
    #scatter plot of the centres of the clusters
    plt.colorbar()
    plt.scatter(centres_gmm[:,:1],centres_gmm[:,1:2],marker = '*',s = 250,color = 'k')
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title("( GMM )scatter plot of the clusters for test data ")
    plt.show()
    
  
    #finding the purity score of the test data
    print("Purity Score for GMM Test Data :", purity_score(test_class, GMM_prediction_test))
    print("\n----------------------------------------------------------------------------\n")
    return gmm.lower_bound_
    
GMM_method(10)
print("---------------------------Question 3-------------------------------\n")

#function for predicting the data
def dbscan_predict(dbscan_mod, X_new, metric=spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
    # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_mod.components_):
            if metric(x_new, x_core) < dbscan_mod.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_mod.labels_[dbscan_mod.core_sample_indices_[i]]
                break
    return y_new


def DBSCAN_method(eps,min_samples):
    print("         For epsilon = ",eps,"  and     min_samples = ",min_samples," :- \n")

    #DBSCAN model with epsilon value = eps and minimum samples = min_samples
    dbscan_model=DBSCAN(eps, min_samples).fit(train)
    DBSCAN_prediction_train = dbscan_model.labels_#prediction of the training data
        
    plt.figure(figsize=(13,10))
    #scatter plot of the train data of different clusters after DBSCAN
    
    plt.scatter(train['dimention 1'],train['dimension 2'], c=DBSCAN_prediction_train, cmap='rainbow')
    plt.colorbar()
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title("( DBSCAN )scatter plot of the clusters for train data ")
    plt.show()
    

    #finding the purity score of the training data
    print("Purity Score for DBSCAN Training Data :-", purity_score(train_class, DBSCAN_prediction_train))

    
    
    #prediction of class for the test data
    DBSCAN_prediction_test = dbscan_predict(dbscan_model, np.array(test), metric=spatial.distance.euclidean)
    
    plt.figure(figsize=(13,10))
    #scatter plot of the test data of different clusters after DBSCAN
    
    plt.scatter(test['dimention 1'],test['dimention 2'], c=DBSCAN_prediction_test, cmap='rainbow')
    plt.colorbar()
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.title("( DBSCAN )scatter plot of the clusters for test data ")
    plt.show()

    #finding the purity score for test data
    print("Purity Score for DBSCAN Test Data :-", purity_score(test_class, DBSCAN_prediction_test))
    print("\n----------------------------------------------------------------------------\n")
DBSCAN_method(5,10)

#BONUS QUESTION
print("----------------------------BONUS QUESTION-----------------------------------")
print("For K means : \n")
k = [2,5,8,12,18,20]#given values of K
Distortion_measure = []#list of distortion for different values of K
for i in k:
    
    kmeans = KMeans(i)#Kmeans
    kmeans.fit(train)#fitting the training data in the kmeans
    centres_kmeans = kmeans.cluster_centers_#centres of the clusters
    kmeans_prediction_train = kmeans.predict(train)#predicting the class for training data
    kmeans_prediction_test = kmeans.predict(test)
    dist = sum(np.min(cdist(df[df.columns[:-1]].values, centres_kmeans, 'euclidean'), axis=1)) /train_class.shape[0]
    Distortion_measure.append(dist)#Kmeans for different values of K
    print("\n          K = ",i)
    print("\nPurity Score on training data  = ",purity_score(train_class, kmeans_prediction_train))
    print("Purity Score on test data  = ",purity_score(test_class, kmeans_prediction_test))



plt.figure(figsize=(10,7))
plt.plot(k,Distortion_measure)#plot of k and distortion measure
plt.scatter(k,Distortion_measure,color = 'k')
plt.xlabel("value of K")
plt.ylabel("Distortion measure(J)")
plt.title("Elbow method for convergence")
plt.show()
print("optimal value of K for kmeans is 8")
kmeans_method(8)

print("For GMM : \n")
log_likelihood = []#list of log likelihood for different values of K
for i in k:
    
    gmm = GaussianMixture(i)#GMM with n components
    gmm.fit(train)#fitting the train data into GMM
    GMM_prediction_train = gmm.predict(train)#predictin of the training data
    centres_gmm = gmm.means_#centres of the clusters
    GMM_prediction_test = gmm.predict(test)
    print("\n          K = ",i)
    print("Purity Score on training data  = ",purity_score(train_class, GMM_prediction_train))
    print("Purity Score on test data  = ",purity_score(test_class, GMM_prediction_test))
    log_likelihood.append(gmm.lower_bound_)#GMM methods for different values of K
    
plt.figure(figsize=(10,7))
plt.plot(k,log_likelihood)#plot of k and log likelihood
plt.scatter(k,log_likelihood,color = 'k')
plt.xlabel("value of K")
plt.ylabel("log_likelihood")
plt.title("Elbow method for convergence")
plt.show()   
print("optimal value of K for GMM is 8")    
GMM_method(8)

print("\nDBSCAN method: ")
epsilon = [1,5,10]#given values of epsilon
minimum_samples = [1,10,30,50]#given values of minimum samples
for eps in epsilon:#for each value of epsilon
    for min_sample in minimum_samples:#for each value of minimum samples
        #DBSCAN model with epsilon value = eps and minimum samples = min_samples
        dbscan_model=DBSCAN(eps, min_sample).fit(train)
        DBSCAN_prediction_train = dbscan_model.labels_#prediction of the training data
        DBSCAN_prediction_test = dbscan_predict(dbscan_model, np.array(test), metric=spatial.distance.euclidean)
        
        print("\n          epsilon = ",eps,"  minimum samples = ",min_sample)
        print("Purity Score on training data  = ",purity_score(train_class, DBSCAN_prediction_train))
        print("Purity Score on test data  = ",purity_score(test_class, DBSCAN_prediction_test))









