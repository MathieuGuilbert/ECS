import csv

from scipy.spatial import distance #for tanimoto and cosine distance

import pandas as pd
import numpy as np
import math
import ast

def readDataFile(dataFile:str,delimiter:str=';', quotechar:str='µ'):
    '''Read all lines of a csv datafile.'''
    res=[]
    with open(dataFile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        for row in datareader:
            res.append(row)
    return res

def writeMat(fileName:str,mat:list):
    '''Writes a matrix  at the given path.'''
    with open(fileName, mode='w',newline="") as employee_file:
        dataWriter = csv.writer(employee_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(mat)):
            dataWriter.writerow(mat[i])
    print("Matrix written succefully")
    return

def readBP(file,K):
    '''Read base partitions from a file. If specified, keep only tohse of size K.'''
    #TODO: convert str into list of int
    allBP=readDataFile(file)
    finalBPs=[]
    if(K!=None):
        for BP in allBP:
            if(len(BP)==K):
                strBP=[clust.strip('][').split(', ') for clust in BP]
                intBP=[[eval(i) for i in clust] for clust in strBP]
                finalBPs.append(intBP)
        return finalBPs
    else:
        return allBP

#----------

def getTagsIdsForClusterInstances(descrSpace:list,Cid:list):
    '''convert a list of list where L[i][j] indicates that instance i is covered by tag j 
    into a list of list where res[i] contains the ids of all the tags covering instance i'''
    res=[]
    for i in Cid:
        LI=[]
        for j in range(len(descrSpace[0])):
            #print(i,j,descrSpace[i][j],type(descrSpace[i][j]))
            if descrSpace[i][j]==1:
                LI.append(j)
        res.append(LI)
    return res

#---------- Data format ----------

def createDF(mat:list):
    '''Return the data with the descriptors in the form of a dataframe.'''
    dic={}
    for i in range(len(mat)):
        dic[i]=mat[i]
    df=pd.DataFrame(dic)
    res=df.transpose() #tranpose to have instances as lines and tags as columns
    return res

def createNegDF(mat:list):
    '''Return the data with the descriptors (and the absence of descriptors) in the form of a dataframe.'''
    dic={}
    M=len(mat)
    for i in range(M):
        dic[i]=mat[i]
        dic[i]+=[abs(1-dic[i][j]) for j in range(len(dic[i]))]
        #dic[i+M]=[abs(1-mat[i][j]) for j in range(len(mat[i]))]
    df=pd.DataFrame(dic)
    res=df.transpose() #tranpose to have instances as lines and tags as columns
    return res

#---------- Distances

def defineMatrixTanimoto(data):
    '''Compute a distance matrix using the tanimoto distance.'''
    print("Define Tanimoto distance matrix")
    resDist=[]
    for i in range(len(data)):
        dist=[]
        for j in range(len(data)):
            distanc=distance.rogerstanimoto(data[i],data[j])
            dist.append(distanc)
        resDist.append(dist)
    return np.array(resDist)

def defineMatrixCosine(data):
    '''Compute a distance matrix using the cosine distance.'''
    print("Define Cosine distance matrix")
    resDist=[]
    for i in range(len(data)):
        dist=[]
        for j in range(len(data)):
            distanc=distance.cosine(data[i],data[j])
            dist.append(distanc)
        resDist.append(dist)
    return np.array(resDist)

def defineMatrixEuclidean(data):
    '''Compute a distance matrix using the euclidean distance.'''
    print("Define Euclidean distance matrix")
    resDist=[]
    for i in range(len(data)):
        dist=[]
        for j in range(len(data)):
            distanc=distance.euclidean(data[i],data[j])
            dist.append(distanc)
        resDist.append(dist)
    return np.array(resDist)



def computeFasterDistMat(l, data):
    '''return an euclidean distance matrix between the points cited in list l, using base python functions.'''
    res = []
    maxtreated=0
    for i in range(len(l)):
        lineI = []
        for j in range(len(l)):
            if(j < maxtreated):
                lineI.append(res[j][i])
            else:
                dist = [(a - b)**2 for a, b in zip(data[i],data[j])]
                lineI.append(math.sqrt(sum(dist)))
        res.append(lineI)
        maxtreated+=1
    return res

def computeFastestEuclideanDistance(x,y):
    '''Compute the euclidean distance between two points in the fastest way.'''
    dist = [(a - b)**2 for a, b in zip(x,y)]
    return math.sqrt(sum(dist))



def convertDistInSim(distMat):
    '''Convert a given distance matrix into a similarity matrix.'''
    maxDist=max([max(subList) for subList in distMat]) #find maximum distance
    dist_norm=[[dist/maxDist for dist in subList] for subList in distMat] #normalize the distance matrix
    simMat=[[round(1-dist,3) for dist in subList] for subList in dist_norm]
    return simMat

#---------- Other

def createNegTagSpace(tagNames:list,tagSpace:list):
    '''Return the data with the descriptors (and the absence of descriptors) in the form of a matrix.'''
    resNames=tagNames+['not_'+t for t in tagNames]
    M=len(tagSpace)
    T=len(tagSpace[0])
    resSpace=[]
    for i in range(M):
        line=tagSpace[i].copy()
        for t in range(T):
            if(tagSpace[i][t]==0):
                line.append(1)
            else:
                line.append(0)
        resSpace.append(line)
    return resNames,resSpace