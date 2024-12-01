import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering,DBSCAN

#--- clustering algorithms ---

#-- Kmeans --

def genBaseKmeans(data,minIter,maxIter,maxClustSize):
    '''Generate the base partitions and clusters using the Kmeans algorithm.

    PARAMETERS:
    --------
    data: dataset.
    minIter: minimal K given as parameter to Kmeans.
    maxIter: maximal K given as parameter to Kmeans.
    maxClustSize (NOT USED): maximal cluster size. If a cluster is larger, then it will be decomposed.'''
    print("Generate base partitions with Kmeans.")
    candClustIds=[]
    BasePartitions=[]
    for k in range(minIter,maxIter):
        #print(k)
        kmeans_model = KMeans(n_clusters=k,n_init=10).fit(data) #Added n_init=10 because it is the default value, that need to be explicitlfy specified due to update 1.4 of sklearn
        labels = kmeans_model.labels_
        clustIds=getClusterElementIds(labels,k)
        BasePartitions.append(clustIds)
        for i in range(len(clustIds)):
            candClustIds.append(clustIds[i])
    return candClustIds,BasePartitions

#-- Agglomerative Clustering --

def genBaseAgglo(matrix,minIter,maxIter):
    '''Generate base partitions with Agglomerative clustering (on a distance matrix)'''
    print("Generate base partition with agglomerative Clustering: ")
    candClustIds=[]
    BasePartitions=[]
    for k in range(minIter,maxIter):
        #print(k)
        labels = launchAggloPrecomputed(k,matrix)
        clustIds=getClusterElementIds(labels,k)
        BasePartitions.append(clustIds)
        for i in range(len(clustIds)):
            candClustIds.append(clustIds[i])
    return candClustIds,BasePartitions

#Generate base partitions with Agglomerative clustering (precomputed=distance matrix)
def launchAggloPrecomputed(k,matrix):
	#Consensus function using Single Link
	clustering = AgglomerativeClustering(metric='precomputed',linkage='complete',n_clusters=k).fit(matrix)
	#clustering = AgglomerativeClustering(matrix='precomputed',linkage='complete',n_clusters=k).fit(matrix)
	clustering
	finalLabels=clustering.labels_
	return finalLabels

#-- Spectral Clustering --

def genSpectralPrecomp(matrix,minIter,maxIter):
    print("Generate base partition with Spectral Clustering: ")
    candClustIds=[]
    basePartitions=[]
    for k in range(minIter,maxIter):
        #print(k)
        labels = launchSpectralPrecomp(k,matrix)
        clustIds=getClusterElementIds(labels,k)
        basePartitions.append(clustIds)
        for i in range(len(clustIds)):
            candClustIds.append(clustIds[i])
    return candClustIds,basePartitions

def launchSpectralPrecomp(k,matrix):
	'''Generate base partitions with Agglomerative clustering (on a similarity/affinity matrix)'''
	clustering = SpectralClustering(affinity='precomputed',n_clusters=k).fit(matrix) #,assign_labels='discretize'
	clustering
	finalLabels=clustering.labels_
	return finalLabels

def genSpectralNN(X,minIter,maxIter):
    print("Generate base partition with Spectral Clustering: ")
    candClustIds=[]
    basePartitions=[]
    for k in range(minIter,maxIter):
        labels = launchSpectralNN(k,X)
        clustIds=getClusterElementIds(labels,k)
        basePartitions.append(clustIds)
        for i in range(len(clustIds)):
            candClustIds.append(clustIds[i])
    return candClustIds,basePartitions

def launchSpectralNN(k,X):
	'''Generate base partitions with Agglomerative clustering on raw data with nearest neighbors.'''
	clustering = SpectralClustering(affinity="nearest_neighbors",n_clusters=k).fit(X) #,assign_labels='discretize'
	clustering
	finalLabels=clustering.labels_
	return finalLabels

#-- DBSCAN --

def genDBSCAN(matrix):
    print("Generate base partition with DBSCAN: ")
    candClustIds=[]
    basePartitions=[]
    labels = launchDBSCAN(matrix)
    clustIds=getClusterElementIds(labels,max(labels))
    basePartitions.append(clustIds)
    for i in range(len(clustIds)):
        candClustIds.append(clustIds[i])
    print(candClustIds)
    print(z)
    return candClustIds,basePartitions

#Generate base partitions with DBSCAN (on a similarity/affinity matrix)
def launchDBSCAN(matrix):
    '''DBSCAN'''
    #print('matrix: ',matrix)
    clustering = DBSCAN(eps=0.08, min_samples=7).fit(matrix) #PB: all at -1
    clustering
    finalLabels=clustering.labels_
    print(finalLabels)
    return finalLabels

#--Other--

def getClusterElementIds(Z,K):
    '''Given a partition Z where Zik=1 signify that instance i is in cluster k,
        Returns a list of K lists of ints corresponding to the ids of the instances in corresponding clusters'''
    ids=[[] for i in range(K)]
    for i in range(len(Z)):
        ids[Z[i]].append(i)
    return ids
