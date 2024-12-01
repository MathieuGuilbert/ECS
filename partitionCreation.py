import numpy as np
from cpmpy import *
from cpmpy.tools.tune_solver import ParameterTuner

import pandas as pd
import time
from sklearn import metrics
from sklearn.cluster import KMeans
from basePartitions import genBaseKmeans

from clusterSelection import main
from postProcess import *
from vizual import
from particularDataTreatment import *
from clusterQuality import computeWCSSs,accurate_jaccard,jaccard_sample,overlapping_ari_with_alignment,jaccard_micro
from modelZeta import launchCPmodelZeta

#--- CP MODEL ---

def launchCPmodel(idObj:int,N:int,V:int,instanceClusterMatrix:list,Kmin:int,Kmax:int,
                  apparMin:int,apparMax:int,D:list,clustPatDF:pd.DataFrame,clustSizes:list,
                  maxCovOutPer:int,nbMaxMoreThan1:int,nbMaxLessThan1:int,nbMaxDiffThan1:int,
                  nbMinMoreThan1:int,nbMinLessThan1:int,
                  MSs:list=[],CSs:list=[],customAppar:dict={}):
    """
    CP model aiming at finding a partition or clustering of a dataset.

    Parameters
    --------
    idObj: int
        Objective criterion to apply.
        Objective criterions of the CP models are as follows:
            0 : minimize number of instances attributed to 0 cluster
            1 : minimize number of clusters selected
            5 : minimize overall number of patterns selected
            2 : maximize number of instances attributed to one and only one cluster
            3 : maximize overall number of descriptors selected in cluster descriptions
            4 : maximize number of clusters selected
    N: int
        number of instances in the dataset.
    V: int
        number of candidate clusters.
    instanceClusterMatrix : int matrix
        a N*V matrix where 1 for instance i and cluster c indicate that i is in c.
    Kmin : int
        minimum number of clusters that have to be selected.
    Kmax : int
        maximum number of clusters that can be selected.
    apparMin : int
        minimal number of apparition of each instances in the partition (i.e. minimal number of selected cluster each instance has to be attributed to).
    apparMax :int
    maximal number of apparition of each instances in the partition.
    D :
        list of the descriptions of each of the candidate V clusters.
    clustPatDF :
            dataframe displayining for each candidate pattern the number of instances it covers in each candidate cluster.
    clustSizes : List of int
            a list containing the size of all candidate clusters.
    maxCovOutPer : int
            a percentage of maximum outside coverage (dsicriminativeness parameter).
    nbMaxMoreThan1 : int
            maximum number of instances attributed to more than 1 cluster.
    nbMaxLessThan1 : int
            maximum number of instances attributed to less than 1 cluster.
    nbMaxDiffThan1 : int
            maximum number of instances attributed to anything different than 1 cluster.
    MSs : list of int
            Must-Select constraints: list of candidate cluster ids that must be selected in the final clustering.
    CSs : list of tuple of int
            Cannot-Select constraints: list of couples of cluster ids that are not allowed to be simultaniously selected.
    customAppar :
        specific boundaries for the number of apparition of particular instances.

    Returns
    --------
    F: list of int
        Final Clustering: Fi=1 if candidate cluster i is selected, 0 otherwise.
    nbOfClusterForEachInstance: list of int
        for each instance, number of selected cluster to which it belongs.
    candidateDescr: list of lists of int
        1 or 0 depending if corresponding candidate descriptor is selected.
    """

    #Model
    model = Model()

    #Variables
    F = intvar(0,1, shape=V, name="Clustering")
    nbOfClusterForEachInstance= intvar(0,V, shape=N, name="clustersPerInst")
    candidateDescr=[]
    for c in range(V):
        candidateDescr.append(intvar(0,1, shape=len(D[c]), name="descr "+str(c)))

    #Objectives
    nbrInstNoClust=(nbOfClusterForEachInstance==0).sum() #number of instances attributed to 0 cluster
    nbrInstOneClust=(nbOfClusterForEachInstance==1).sum() #number of instances attributed to one and only one cluster
    nbrInstMultipleClust=N-(nbrInstNoClust+nbrInstOneClust) #number of instances attributed to 0 cluster
    nbrClustSelected=sum(F) #number of clusters selected
    nbrOverallPat=sum( sum([candidateDescr[c]]) for c in range(V) ) #overall number of descriptors selected in cluster descriptions

    if(idObj==0): #minimize number of instances attributed to 0 cluster
        obj=nbrInstNoClust
        model.minimize(obj)
    elif(idObj==1): #minimize number of clusters selected
        obj=nbrClustSelected
        model.minimize(obj)
    elif(idObj==5): #minimize overall number of patterns selected
        obj=nbrOverallPat
        model.minimize(obj)
    else:
        if(idObj==2): #maximize number of instances attributed to one and only one cluster
            obj=nbrInstOneClust
        elif(idObj==3): #maximize overall number of descriptors selected in cluster descriptions
            obj=nbrOverallPat
        elif(idObj==4): #maximize number of clusters selected
            obj=nbrClustSelected
        model.maximize(obj)

    # - Constraints -

    #number of clusters
    model += sum(F)>=Kmin
    model += sum(F)<=Kmax

    #number of apparition of the instances in selected clusters
    for i in range(N):
        model += (nbOfClusterForEachInstance[i]== sum([F[c]*instanceClusterMatrix[i][c] for c in range(V)]) )
        if(i in customAppar): #User constraint: specific boundaries for the number of apparition of a particular instance
            model += (nbOfClusterForEachInstance[i]>= customAppar[i][0])
            model += (nbOfClusterForEachInstance[i]<= customAppar[i][1])
        else:
            model += (nbOfClusterForEachInstance[i]>= apparMin)
            model += (nbOfClusterForEachInstance[i]<= apparMax)

    #number of instances attributed to more or less than 1 cluster:
    if(nbMaxLessThan1!=None):
        model+= ( nbrInstNoClust<=nbMaxLessThan1 )
    if(nbMaxMoreThan1!=None):
        model+= ( nbrInstMultipleClust<=nbMaxMoreThan1 )
    if(nbMinLessThan1!=None):
        model+= ( nbrInstNoClust>=nbMinLessThan1 )
    if(nbMinMoreThan1!=None):
        model+= ( nbrInstMultipleClust>=nbMinMoreThan1 )
    if(nbMaxDiffThan1!=None):
        model+= ( nbrInstNoClust+nbrInstMultipleClust<=nbMaxDiffThan1 )

    #Cannot select patterns as cluster descriptors if they cover too much of the instances in other selected clusters
    #not explicitly in article
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d]
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100 #max number of instances in c1 covered by outside patterns
                    if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                        #if c and c1 are selected, then the presence of each pattern of c is possible (not mandatory, ence the <= instead of <)
                        # iff the coverage of p on c1 is inferior to patThreshold
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c] <= int(clustPatDF[c1][p]<=patThreshold) )
                    else:
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c][d] <= int(clustPatDF[c1][p]<=patThreshold) )

    #if a pattern j is selected in the final description of a selected cluster,
    # then all other cluster for which j covers more then patThreshold elements cannot be selected.
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d] #get the corresponding pattern id
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100
                    if(clustPatDF[c1][p]>patThreshold):
                        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                            model += (candidateDescr[c]==1).implies(F[c1]==0)
                        else:
                            model += (candidateDescr[c][d]==1).implies(F[c1]==0)

    #Non-selected clusters must have all their candidate descriptors set to 0
    for c in range(V):
        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
            model += (candidateDescr[c]==F[c]) #the selection of the cluster must be complementary with the selection of its unique pattern
        else: #Note: we could formulate following constr as implications
            model += (F[c]<=sum(candidateDescr[c])) #cluster cannot be selected if at least one of its descr isn't
            for p in range(len(D[c])):
                model += (candidateDescr[c][p]<=F[c]) #descriptors cannot be selected if their cluster is not

    #If a cluster is selected then all of its patterns that do not cover any other cluster is selected in its final description.
    for c in range(V):
        for d in range(len(D[c])):
            p=D[c][d]
            patThreshold=(maxCovOutPer*clustSizes[c])/100 #max number of instances in c1 covered by outside patterns
            covC=int(clustPatDF[c][p] > patThreshold)
            if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c]==F[c]) #if c is selected and the sum of clusters selected and covered by p (minus c) is equal to 0 than p must describe c
            else:
                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d]==F[c])

    #User Constraints
    for c in MSs: #Must Select
        model += F[c]==1
    for (c1,c2) in CSs: #Cannot Select c1 and c2 at the same time
        model += F[c1]+F[c2]<2

    #Solver
    #if SolverLookup.get("ortools", model).solve(**tuner.best_params):
    if SolverLookup.get("ortools", model).solve():
        print("Number of instances not in any cluster : ",nbrInstNoClust.value())
        print("Number of instances in exactly one cluster : ",nbrInstOneClust.value())
        print("Number of cluster selected : ",nbrClustSelected.value())
        print("Overall number of selected patterns : ", nbrOverallPat.value())
        print("objective value ("+str(idObj)+") : ",obj.value())
        print()
        return F,nbOfClusterForEachInstance,candidateDescr,obj
    else:
        print("No solution found")

    return

#TODO added obj on number of covered object  (using instPatDF) and limit on number of selected patterns per cluster
#TODO add obj on discrimination
def launchCPmodelNewObj(idObj:int,N:int,V:int,instanceClusterMatrix:list,Kmin:int,Kmax:int,
                  apparMin:int,apparMax:int,D:list,clustPatDF:pd.DataFrame,instPatDF:pd.DataFrame,listPat:list,clustSizes:list,
                  maxCovOutPer:int,nbMaxMoreThan1:int,nbMaxLessThan1:int,nbMaxDiffThan1:int,
                  nbMinMoreThan1:int,nbMinLessThan1:int,patPreference:int=0,
                  MSs:list=[],CSs:list=[],customAppar:dict={},maxNbPatPerClust:int=5):#None):
    """
    CP model aiming at finding a partition or clustering of a dataset.

    Parameters
    --------
    idObj: int
        Objective criterion to apply.
        Objective criterions of the CP models are as follows:
            0 : minimize number of instances attributed to 0 cluster
            1 : minimize number of clusters selected
            5 : minimize overall number of patterns selected
            2 : maximize number of instances attributed to one and only one cluster
            3 : maximize overall number of descriptors selected in cluster descriptions
            4 : maximize number of clusters selected
    N: int
        number of instances in the dataset.
    V: int
        number of candidate clusters.
    instanceClusterMatrix : int matrix
        a N*V matrix where 1 for instance i and cluster c indicate that i is in c.
    Kmin : int
        minimum number of clusters that have to be selected.
    Kmax : int
        maximum number of clusters that can be selected.
    apparMin : int
        minimal number of apparition of each instances in the partition (i.e. minimal number of selected cluster each instance has to be attributed to).
    apparMax :int
    maximal number of apparition of each instances in the partition.
    D :
        list of the descriptions of each of the candidate V clusters.
    clustPatDF :
            dataframe displayining for each candidate pattern the number of instances it covers in each candidate cluster.
    clustSizes : List of int
            a list containing the size of all candidate clusters.
    maxCovOutPer : int
            a percentage of maximum outside coverage (dsicriminativeness parameter).
    nbMaxMoreThan1 : int
            maximum number of instances attributed to more than 1 cluster.
    nbMaxLessThan1 : int
            maximum number of instances attributed to less than 1 cluster.
    nbMaxDiffThan1 : int
            maximum number of instances attributed to anything different than 1 cluster.
    patPreference : int
            Information about what type of pattern is favorised.
                0 : no preference.
                1 : favorize more general patterns.
                2 : favorize more specific patterns.
    MSs : list of int
            Must-Select constraints: list of candidate cluster ids that must be selected in the final clustering.
    CSs : list of tuple of int
            Cannot-Select constraints: list of couples of cluster ids that are not allowed to be simultaniously selected.
    customAppar :
        specific boundaries for the number of apparition of particular instances.

    Returns
    --------
    F: list of int
        Final Clustering: Fi=1 if candidate cluster i is selected, 0 otherwise.
    nbOfClusterForEachInstance: list of int
        for each instance, number of selected cluster to which it belongs.
    candidateDescr: list of lists of int
        1 or 0 depending if corresponding candidate descriptor is selected.
    """

    #Model
    model = Model()

    #Variables
    F = intvar(0,1, shape=V, name="Clustering")
    nbOfClusterForEachInstance= intvar(0,V, shape=N, name="clustersPerInst")
    #isInstanceCovered= intvar(0,1, shape=N, name="covInst")
    isInstanceCovered= boolvar(shape=N, name="covInst")
    isInstanceCovAnd1Clust= boolvar(shape=N, name="covAnd1Clust")
    candidateDescr=[]
    for c in range(V):
        candidateDescr.append(intvar(0,1, shape=len(D[c]), name="descr "+str(c)))

    #Objectives
    nbrInstNoClust=(nbOfClusterForEachInstance==0).sum() #number of instances attributed to 0 cluster
    nbrInstOneClust=(nbOfClusterForEachInstance==1).sum() #number of instances attributed to one and only one cluster
    nbrInstMultipleClust=N -(nbrInstNoClust+nbrInstOneClust) #number of instances attributed to 0 cluster
    nbrClustSelected=sum(F) #number of clusters selected
    nbrCoveredInstances=sum(isInstanceCovered) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    #nbrCoveredAndOneClust=sum((isInstanceCovered[o]==True and (nbOfClusterForEachInstance[o]==1)) for o in range(N)) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    nbrCoveredAndOneClust=sum(isInstanceCovAnd1Clust) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    nbrOverallPat=sum( sum([candidateDescr[c]]) for c in range(V) ) #overall number of descriptors selected in cluster descriptions
    #TODO min or mean discr ?

    if(idObj==0): #minimize number of instances attributed to 0 cluster
        obj=nbrInstNoClust
        model.minimize(obj)
    elif(idObj==1): #minimize number of clusters selected
        obj=nbrClustSelected
        model.minimize(obj)
    elif(idObj==5): #minimize overall number of patterns selected
        obj=nbrOverallPat
        model.minimize(obj)
    elif(idObj==8): #minimize number of instances covered by at least one pattern.
        obj=nbrCoveredInstances
        model.minimize(obj)
    else:
        if(idObj==2): #maximize number of instances attributed to one and only one cluster
            obj=nbrInstOneClust
        elif(idObj==3): #maximize overall number of descriptors selected in cluster descriptions
            obj=nbrOverallPat
        elif(idObj==4): #maximize number of clusters selected
            obj=nbrClustSelected
        elif(idObj==7): #maximize number of instances covered by at least one pattern.
            obj=nbrCoveredInstances
        elif(idObj==9): #maximize number of instances covered by at least one pattern AND belonging to precisly one cluster.
            obj=nbrCoveredAndOneClust
        model.maximize(obj)

    # - Constraints -

    #number of clusters
    model += sum(F)>=Kmin
    model += sum(F)<=Kmax

    #number of apparition of the instances in selected clusters
    for i in range(N):
        model += (nbOfClusterForEachInstance[i]== sum([F[c]*instanceClusterMatrix[i][c] for c in range(V)]) )
        if(i in customAppar): #User constraint: specific boundaries for the number of apparition of a particular instance
            model += (nbOfClusterForEachInstance[i]>= customAppar[i][0])
            model += (nbOfClusterForEachInstance[i]<= customAppar[i][1])
        else:
            model += (nbOfClusterForEachInstance[i]>= apparMin)
            model += (nbOfClusterForEachInstance[i]<= apparMax)

    #number of instances attributed to more or less than 1 cluster:
    if(nbMaxLessThan1!=None):
        model+= ( nbrInstNoClust<=nbMaxLessThan1 )
    if(nbMaxMoreThan1!=None):
        model+= ( nbrInstMultipleClust<=nbMaxMoreThan1 )
    if(nbMinLessThan1!=None):
        model+= ( nbrInstNoClust>=nbMinLessThan1 )
    if(nbMinMoreThan1!=None):
        model+= ( nbrInstMultipleClust>=nbMinMoreThan1 )
    if(nbMaxDiffThan1!=None):
        model+= ( nbrInstNoClust+nbrInstMultipleClust<=nbMaxDiffThan1 )

    #Cannot select patterns as cluster descriptors if they cover too much of the instances in other selected clusters
    #not explicitly in article
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d]
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100 #max number of instances in c1 covered by outside patterns
                    if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                        #if c and c1 are selected, then the presence of each pattern of c is possible (not mandatory, ence the <= instead of <)
                        # iff the coverage of p on c1 is inferior to patThreshold
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c] <= int(clustPatDF[c1][p]<=patThreshold) )
                    else:
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c][d] <= int(clustPatDF[c1][p]<=patThreshold) )

    #if a pattern j is selected in the final description of a selected cluster,
    # then all other cluster for which j covers more then patThreshold elements cannot be selected.
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d] #get the corresponding pattern id
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100
                    if(clustPatDF[c1][p]>patThreshold):
                        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                            model += (candidateDescr[c]==1).implies(F[c1]==0)
                        else:
                            model += (candidateDescr[c][d]==1).implies(F[c1]==0)

    #Non-selected clusters must have all their candidate descriptors set to 0
    for c in range(V):
        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
            model += (candidateDescr[c]==F[c]) #the selection of the cluster must be complementary with the selection of its unique pattern
        else: #Note: we could formulate following constr as implications
            model += (F[c]<=sum(candidateDescr[c])) #cluster cannot be selected if at least one of its descr isn't
            for p in range(len(D[c])):
                model += (candidateDescr[c][p]<=F[c]) #descriptors cannot be selected if their cluster is not

    #Limit the number of patterns that can be selected for each cluster.
    #Note: if obj is not cov or discr, there is no constraint on the selection of patterns.
    if(maxNbPatPerClust!=None):
        for c in range(V):
            if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>" and len((candidateDescr[c])>maxNbPatPerClust)): #If more than one original descriptor
                model += (sum(candidateDescr[c])<=maxNbPatPerClust)

    else:   #does not need to work; unable to find results OR simple param pb ? #TODO this constraint needs to be modified if obj is to max the discrimination (via the else ?)
        #If a cluster is selected then all of its patterns that do not cover any other cluster are selected in its final description.
        for c in range(V):
            for d in range(len(D[c])):
                p=D[c][d]
                patThreshold=(maxCovOutPer*clustSizes[c])/100 #max number of instances in c1 covered by outside patterns
                covC=int(clustPatDF[c][p] > patThreshold)
                if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                    model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c]==F[c]) #if c is selected and the sum of clusters selected and covered by p (minus c) is equal to 0 than p must describe c
                else:
                    model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d]==F[c])


    #NEW: isInstanceCovered and isInstanceCovAnd1Clust constraints
    for i in range(N):
        #If point is not in any selected cluster it is not covered
        model += (nbOfClusterForEachInstance[i]==0).implies(isInstanceCovered[i]==False)

        #isInstanceCovAnd1Clust
        model += isInstanceCovAnd1Clust[i]==(nbOfClusterForEachInstance[i]==1 and isInstanceCovered[i]==True)
        model += (isInstanceCovered[i]==False).implies(isInstanceCovAnd1Clust[i]==False)
        model += (nbOfClusterForEachInstance[i]!=1).implies(isInstanceCovAnd1Clust[i]==False)

        #If want covered by any clust point, remove instClusterMatrix from constraint
        model += isInstanceCovered[i]==(
            sum([ F[c]*(sum(F[c]*instanceClusterMatrix[i][c]*candidateDescr[c]*instPatDF[listPat[D[c][d]]][i] for d in range(len(D[c])) if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"))
                       +sum(F[c]*instanceClusterMatrix[i][c]*candidateDescr[c][d]*instPatDF[listPat[D[c][d]]][i] for d in range(len(D[c])) if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>" ))
            )for c in range(V) ])>=1)

    #Preference and redundancy constraints
    if(patPreference!=0):
        for c in range(V):
            if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>"):
                #for each pairs of patterns
                for d1 in range(len(D[c])):
                    p1=D[c][d1]
                    for d2 in range(len(D[c])):
                        p2=D[c][d2]
                        #Check if p1 is a subset of p2.
                        if(d1!=d2 and p1!=p2 and ast.literal_eval(listPat[p1])!=ast.literal_eval(listPat[p2]) and set(ast.literal_eval(listPat[p1])).issubset(ast.literal_eval(listPat[p2]))):
                            #print('DEBUG',listPat[p1],'is a subset of',listPat[p2],'; ds:',d1,d2,'; ps:',p1,p2)
                            #Redondancy constraint : Both cannot be selected at the same time.
                            model+= (candidateDescr[c][d1]+candidateDescr[c][d2]<2)

                            #Preference constraints
                            if(patPreference==1): #fav more general patterns (p2).
                                patThreshold=(maxCovOutPer*clustSizes[c])/100
                                covC=int(clustPatDF[c][p2] > patThreshold)
                                #If p2 can be selected (i.e. is discriminative), then p1 is forbidden.
                                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p2] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d1]==0)

                            elif(patPreference==2): #fav more specific patterns (p1).
                                patThreshold=(maxCovOutPer*clustSizes[c])/100
                                covC=int(clustPatDF[c][p1] > patThreshold)
                                #If p1 can be selected (i.e. is discriminative), then p2 is forbidden.
                                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p1] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d2]==0)

                           # elif(patPreference>2):
                                #print('No preferences')

    #User Constraints
    for c in MSs: #Must Select
        model += F[c]==1
    for (c1,c2) in CSs: #Cannot Select c1 and c2 at the same time
        model += F[c1]+F[c2]<2

    #Solver
    #if SolverLookup.get("ortools", model).solve(**tuner.best_params):
    if SolverLookup.get("ortools", model).solve():
        print("Number of instances not in any cluster : ",nbrInstNoClust.value())
        print("Number of instances in exactly one cluster : ",nbrInstOneClust.value())
        print("Number of instances in more than one cluster : ",nbrInstMultipleClust.value())
        print("Number of cluster selected : ",nbrClustSelected.value())
        print("Overall number of selected patterns : ", nbrOverallPat.value())
        print("Overall number of covered objects : ", nbrCoveredInstances.value()," out of",N)
        print("Overall number of covered and 1-clustered objects : ", nbrCoveredAndOneClust.value()," out of",N)
        print("objective value ("+str(idObj)+") : ",obj.value())
        print()
        return F,nbOfClusterForEachInstance,candidateDescr,obj
    else:
        print("No solution found")

    return

#TODO added obj on number of covered object  (using instPatDF) and limit on number of selected patterns per cluster
#TODO add obj on discrimination
def launchCPmodelNewObjRephrased(idObj:int,N:int,V:int,instanceClusterMatrix:list,Kmin:int,Kmax:int,
                  apparMin:int,apparMax:int,D:list,clustPatDF:pd.DataFrame,instPatDF:pd.DataFrame,listPat:list,clustSizes:list,
                  maxCovOutPer:int,nbMaxMoreThan1:int,nbMaxLessThan1:int,nbMaxDiffThan1:int,
                  nbMinMoreThan1:int,nbMinLessThan1:int,patPreference:int=0,
                  MSs:list=[],CSs:list=[],customAppar:dict={},maxNbPatPerClust:int=5):#None):
    """
    CP model aiming at finding a partition or clustering of a dataset.

    Parameters
    --------
    idObj: int
        Objective criterion to apply.
        Objective criterions of the CP models are as follows:
            0 : minimize number of instances attributed to 0 cluster
            1 : minimize number of clusters selected
            5 : minimize overall number of patterns selected
            2 : maximize number of instances attributed to one and only one cluster
            3 : maximize overall number of descriptors selected in cluster descriptions
            4 : maximize number of clusters selected
    N: int
        number of instances in the dataset.
    V: int
        number of candidate clusters.
    instanceClusterMatrix : int matrix
        a N*V matrix where 1 for instance i and cluster c indicate that i is in c.
    Kmin : int
        minimum number of clusters that have to be selected.
    Kmax : int
        maximum number of clusters that can be selected.
    apparMin : int
        minimal number of apparition of each instances in the partition (i.e. minimal number of selected cluster each instance has to be attributed to).
    apparMax :int
    maximal number of apparition of each instances in the partition.
    D :
        list of the descriptions of each of the candidate V clusters.
    clustPatDF :
            dataframe displayining for each candidate pattern the number of instances it covers in each candidate cluster.
    clustSizes : List of int
            a list containing the size of all candidate clusters.
    maxCovOutPer : int
            a percentage of maximum outside coverage (dsicriminativeness parameter).
    nbMaxMoreThan1 : int
            maximum number of instances attributed to more than 1 cluster.
    nbMaxLessThan1 : int
            maximum number of instances attributed to less than 1 cluster.
    nbMaxDiffThan1 : int
            maximum number of instances attributed to anything different than 1 cluster.
    patPreference : int
            Information about what type of pattern is favorised.
                0 : no preference.
                1 : favorize more general patterns.
                2 : favorize more specific patterns.
    MSs : list of int
            Must-Select constraints: list of candidate cluster ids that must be selected in the final clustering.
    CSs : list of tuple of int
            Cannot-Select constraints: list of couples of cluster ids that are not allowed to be simultaniously selected.
    customAppar :
        specific boundaries for the number of apparition of particular instances.

    Returns
    --------
    F: list of int
        Final Clustering: Fi=1 if candidate cluster i is selected, 0 otherwise.
    nbOfClusterForEachInstance: list of int
        for each instance, number of selected cluster to which it belongs.
    candidateDescr: list of lists of int
        1 or 0 depending if corresponding candidate descriptor is selected.
    """

    #Model
    model = Model()

    #Variables
    F = intvar(0,1, shape=V, name="Clustering")
    nbOfClusterForEachInstance= intvar(0,V, shape=N, name="clustersPerInst")
    isInstanceCovered= boolvar(shape=N, name="covInst")
    isInstanceCovAnd1Clust= boolvar(shape=N, name="covAnd1Clust")
    candidateDescr=[]
    for c in range(V):
        candidateDescr.append(intvar(0,1, shape=len(D[c]), name="descr "+str(c)))

    #Objectives
    nbrInstNoClust=(nbOfClusterForEachInstance==0).sum() #number of instances attributed to 0 cluster
    nbrInstOneClust=(nbOfClusterForEachInstance==1).sum() #number of instances attributed to one and only one cluster
    nbrInstMultipleClust=N -(nbrInstNoClust+nbrInstOneClust) #number of instances attributed to 0 cluster
    nbrClustSelected=sum(F) #number of clusters selected
    nbrCoveredInstances=sum(isInstanceCovered) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    #nbrCoveredAndOneClust=sum((isInstanceCovered[o]==True and (nbOfClusterForEachInstance[o]==1)) for o in range(N)) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    nbrCoveredAndOneClust=sum(isInstanceCovAnd1Clust) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    nbrOverallPat=sum( sum([candidateDescr[c]]) for c in range(V) ) #overall number of descriptors selected in cluster descriptions
    #TODO min or mean discr ?

    if(idObj==0): #minimize number of instances attributed to 0 cluster
        obj=nbrInstNoClust
        model.minimize(obj)
    elif(idObj==1): #minimize number of clusters selected
        obj=nbrClustSelected
        model.minimize(obj)
    elif(idObj==5): #minimize overall number of patterns selected
        obj=nbrOverallPat
        model.minimize(obj)
    elif(idObj==8): #minimize number of instances covered by at least one pattern.
        obj=nbrCoveredInstances
        model.minimize(obj)
    else:
        if(idObj==2): #maximize number of instances attributed to one and only one cluster
            obj=nbrInstOneClust
        elif(idObj==3): #maximize overall number of descriptors selected in cluster descriptions
            obj=nbrOverallPat
        elif(idObj==4): #maximize number of clusters selected
            obj=nbrClustSelected
        elif(idObj==7): #maximize number of instances covered by at least one pattern.
            obj=nbrCoveredInstances
        elif(idObj==9): #maximize number of instances covered by at least one pattern AND belonging to precisly one cluster.
            obj=nbrCoveredAndOneClust
        model.maximize(obj)

    # - Constraints -

    #number of clusters
    model += sum(F)>=Kmin
    model += sum(F)<=Kmax

    #number of instances attributed to more or less than 1 cluster:
    if(nbMaxLessThan1!=None):
        model+= ( nbrInstNoClust<=nbMaxLessThan1 )
    if(nbMaxMoreThan1!=None):
        model+= ( nbrInstMultipleClust<=nbMaxMoreThan1 )
    if(nbMinLessThan1!=None):
        model+= ( nbrInstNoClust>=nbMinLessThan1 )
    if(nbMinMoreThan1!=None):
        model+= ( nbrInstMultipleClust>=nbMinMoreThan1 )
    if(nbMaxDiffThan1!=None):
        model+= ( nbrInstNoClust+nbrInstMultipleClust<=nbMaxDiffThan1 )

    for i in range(N):
        #number of apparition of the instances in selected clusters
        model += (nbOfClusterForEachInstance[i]== sum([F[c]*instanceClusterMatrix[i][c] for c in range(V)]) )
        if(i in customAppar): #User constraint: specific boundaries for the number of apparition of a particular instance
            model += (nbOfClusterForEachInstance[i]>= customAppar[i][0])
            model += (nbOfClusterForEachInstance[i]<= customAppar[i][1])
        else:
            model += (nbOfClusterForEachInstance[i]>= apparMin)
            model += (nbOfClusterForEachInstance[i]<= apparMax)

        #NEW: isInstanceCovered and isInstanceCovAnd1Clust constraints
        #If point is not in any selected cluster it is not covered
        model += (nbOfClusterForEachInstance[i]==0).implies(isInstanceCovered[i]==False)

        #isInstanceCovAnd1Clust
        model += isInstanceCovAnd1Clust[i]==(nbOfClusterForEachInstance[i]==1 and isInstanceCovered[i]==True)
        model += (isInstanceCovered[i]==False).implies(isInstanceCovAnd1Clust[i]==False)
        model += (nbOfClusterForEachInstance[i]!=1).implies(isInstanceCovAnd1Clust[i]==False)

        #If want covered by any clust point, remove instClusterMatrix from constraint
        model += isInstanceCovered[i]==(
            sum([ F[c]*(sum(F[c]*instanceClusterMatrix[i][c]*candidateDescr[c]*instPatDF[listPat[D[c][d]]][i] for d in range(len(D[c])) if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"))
                       +sum(F[c]*instanceClusterMatrix[i][c]*candidateDescr[c][d]*instPatDF[listPat[D[c][d]]][i] for d in range(len(D[c])) if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>" ))
            )for c in range(V) ])>=1)

    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    #Cannot select patterns as cluster descriptors if they cover too much of the instances in other selected clusters
                    #not explicitly in article
                    p=D[c][d]
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100 #max number of instances in c1 covered by outside patterns
                    if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                        #if c and c1 are selected, then the presence of each pattern of c is possible (not mandatory, ence the <= instead of <)
                        # iff the coverage of p on c1 is inferior to patThreshold
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c] <= int(clustPatDF[c1][p]<=patThreshold) )
                    else:
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c][d] <= int(clustPatDF[c1][p]<=patThreshold) )

                    #if a pattern j is selected in the final description of a selected cluster,
                    # then all other cluster for which j covers more then patThreshold elements cannot be selected.
                    p=D[c][d] #get the corresponding pattern id
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100
                    if(clustPatDF[c1][p]>patThreshold):
                        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                            model += (candidateDescr[c]==1).implies(F[c1]==0)
                        else:
                            model += (candidateDescr[c][d]==1).implies(F[c1]==0)

        #Non-selected clusters must have all their candidate descriptors set to 0
        #for c in range(V):
        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
            model += (candidateDescr[c]==F[c]) #the selection of the cluster must be complementary with the selection of its unique pattern
        else: #Note: we could formulate following constr as implications
            model += (F[c]<=sum(candidateDescr[c])) #cluster cannot be selected if at least one of its descr isn't
            for p in range(len(D[c])):
                model += (candidateDescr[c][p]<=F[c]) #descriptors cannot be selected if their cluster is not

        #NEW: Preference and redundancy constraints
        if(patPreference!=0):
        #for c in range(V):
            if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>"):
                #for each pairs of patterns
                for d1 in range(len(D[c])):
                    p1=D[c][d1]
                    for d2 in range(len(D[c])):
                        p2=D[c][d2]
                        #Check if p1 is a subset of p2.
                        if(d1!=d2 and p1!=p2 and ast.literal_eval(listPat[p1])!=ast.literal_eval(listPat[p2]) and set(ast.literal_eval(listPat[p1])).issubset(ast.literal_eval(listPat[p2]))):
                            #print('DEBUG',listPat[p1],'is a subset of',listPat[p2],'; ds:',d1,d2,'; ps:',p1,p2)
                            #Redondancy constraint : Both cannot be selected at the same time.
                            model+= (candidateDescr[c][d1]+candidateDescr[c][d2]<2)

                            #Preference constraints
                            if(patPreference==1): #fav more general patterns (p2).
                                patThreshold=(maxCovOutPer*clustSizes[c])/100
                                covC=int(clustPatDF[c][p2] > patThreshold)
                                #If p2 can be selected (i.e. is discriminative), then p1 is forbidden.
                                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p2] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d1]==0)

                            elif(patPreference==2): #fav more specific patterns (p1).
                                patThreshold=(maxCovOutPer*clustSizes[c])/100
                                covC=int(clustPatDF[c][p1] > patThreshold)
                                #If p1 can be selected (i.e. is discriminative), then p2 is forbidden.
                                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p1] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d2]==0)

                           # elif(patPreference>2):
                                #print('No preferences')

        #Limit the number of patterns that can be selected for each cluster.
        #Note: if obj is not cov or discr, there is no constraint on the selection of patterns.
        if(maxNbPatPerClust!=None):
        #for c in range(V):
            if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>" and len((candidateDescr[c])>maxNbPatPerClust)): #If more than one original descriptor
                model += (sum(candidateDescr[c])<=maxNbPatPerClust)

        else:   #does not need to work; unable to find results OR simple param pb ? #TODO this constraint needs to be modified if obj is to max the discrimination (via the else ?)
        #If a cluster is selected then all of its patterns that do not cover any other cluster are selected in its final description.
        #for c in range(V):
            for d in range(len(D[c])):
                p=D[c][d]
                patThreshold=(maxCovOutPer*clustSizes[c])/100 #max number of instances in c1 covered by outside patterns
                covC=int(clustPatDF[c][p] > patThreshold)
                if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                    model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c]==F[c]) #if c is selected and the sum of clusters selected and covered by p (minus c) is equal to 0 than p must describe c
                else:
                    model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d]==F[c])


    #User Constraints
    for c in MSs: #Must Select
        model += F[c]==1
    for (c1,c2) in CSs: #Cannot Select c1 and c2 at the same time
        model += F[c1]+F[c2]<2

    #Solver
    #if SolverLookup.get("ortools", model).solve(**tuner.best_params):
    if SolverLookup.get("ortools", model).solve():
        print("Number of instances not in any cluster : ",nbrInstNoClust.value())
        print("Number of instances in exactly one cluster : ",nbrInstOneClust.value())
        print("Number of instances in more than one cluster : ",nbrInstMultipleClust.value())
        print("Number of cluster selected : ",nbrClustSelected.value())
        print("Overall number of selected patterns : ", nbrOverallPat.value())
        print("Overall number of covered objects : ", nbrCoveredInstances.value()," out of",N)
        print("Overall number of covered and 1-clustered objects : ", nbrCoveredAndOneClust.value()," out of",N)
        print("objective value ("+str(idObj)+") : ",obj.value())
        print()
        return F,nbOfClusterForEachInstance,candidateDescr,obj
    else:
        print("No solution found")

    return

#TODO added obj on number of covered object  (using instPatDF) and limit on number of selected patterns per cluster
#TODO add obj on discrimination
def launchCPmodelNewObj(idObj:int,N:int,V:int,instanceClusterMatrix:list,Kmin:int,Kmax:int,
                  apparMin:int,apparMax:int,D:list,clustPatDF:pd.DataFrame,instPatDF:pd.DataFrame,listPat:list,clustSizes:list,
                  maxCovOutPer:int,nbMaxMoreThan1:int,nbMaxLessThan1:int,nbMaxDiffThan1:int,
                  nbMinMoreThan1:int,nbMinLessThan1:int,patPreference:int=0,
                  MSs:list=[],CSs:list=[],customAppar:dict={},maxNbPatPerClust:int=5):#None):
    """
    CP model aiming at finding a partition or clustering of a dataset.

    Parameters
    --------
    idObj: int
        Objective criterion to apply.
        Objective criterions of the CP models are as follows:
            0 : minimize number of instances attributed to 0 cluster
            1 : minimize number of clusters selected
            5 : minimize overall number of patterns selected
            2 : maximize number of instances attributed to one and only one cluster
            3 : maximize overall number of descriptors selected in cluster descriptions
            4 : maximize number of clusters selected
    N: int
        number of instances in the dataset.
    V: int
        number of candidate clusters.
    instanceClusterMatrix : int matrix
        a N*V matrix where 1 for instance i and cluster c indicate that i is in c.
    Kmin : int
        minimum number of clusters that have to be selected.
    Kmax : int
        maximum number of clusters that can be selected.
    apparMin : int
        minimal number of apparition of each instances in the partition (i.e. minimal number of selected cluster each instance has to be attributed to).
    apparMax :int
    maximal number of apparition of each instances in the partition.
    D :
        list of the descriptions of each of the candidate V clusters.
    clustPatDF :
            dataframe displayining for each candidate pattern the number of instances it covers in each candidate cluster.
    clustSizes : List of int
            a list containing the size of all candidate clusters.
    maxCovOutPer : int
            a percentage of maximum outside coverage (dsicriminativeness parameter).
    nbMaxMoreThan1 : int
            maximum number of instances attributed to more than 1 cluster.
    nbMaxLessThan1 : int
            maximum number of instances attributed to less than 1 cluster.
    nbMaxDiffThan1 : int
            maximum number of instances attributed to anything different than 1 cluster.
    patPreference : int
            Information about what type of pattern is favorised.
                0 : no preference.
                1 : favorize more general patterns.
                2 : favorize more specific patterns.
    MSs : list of int
            Must-Select constraints: list of candidate cluster ids that must be selected in the final clustering.
    CSs : list of tuple of int
            Cannot-Select constraints: list of couples of cluster ids that are not allowed to be simultaniously selected.
    customAppar :
        specific boundaries for the number of apparition of particular instances.

    Returns
    --------
    F: list of int
        Final Clustering: Fi=1 if candidate cluster i is selected, 0 otherwise.
    nbOfClusterForEachInstance: list of int
        for each instance, number of selected cluster to which it belongs.
    candidateDescr: list of lists of int
        1 or 0 depending if corresponding candidate descriptor is selected.
    """

    #Model
    model = Model()

    #Variables
    F = intvar(0,1, shape=V, name="Clustering")
    nbOfClusterForEachInstance= intvar(0,V, shape=N, name="clustersPerInst")
    #isInstanceCovered= intvar(0,1, shape=N, name="covInst")
    isInstanceCovered= boolvar(shape=N, name="covInst")
    isInstanceCovAnd1Clust= boolvar(shape=N, name="covAnd1Clust")
    candidateDescr=[]
    for c in range(V):
        candidateDescr.append(intvar(0,1, shape=len(D[c]), name="descr "+str(c)))

    #Objectives
    nbrInstNoClust=(nbOfClusterForEachInstance==0).sum() #number of instances attributed to 0 cluster
    nbrInstOneClust=(nbOfClusterForEachInstance==1).sum() #number of instances attributed to one and only one cluster
    nbrInstMultipleClust=N -(nbrInstNoClust+nbrInstOneClust) #number of instances attributed to 0 cluster
    nbrClustSelected=sum(F) #number of clusters selected
    nbrCoveredInstances=sum(isInstanceCovered) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    #nbrCoveredAndOneClust=sum((isInstanceCovered[o]==True and (nbOfClusterForEachInstance[o]==1)) for o in range(N)) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    nbrCoveredAndOneClust=sum(isInstanceCovAnd1Clust) #number of covered instances, i.e. instances represented by at least 1 selected pattern of a cluster they belong to.
    nbrOverallPat=sum( sum([candidateDescr[c]]) for c in range(V) ) #overall number of descriptors selected in cluster descriptions
    #TODO min or mean discr ?

    if(idObj==0): #minimize number of instances attributed to 0 cluster
        obj=nbrInstNoClust
        model.minimize(obj)
    elif(idObj==1): #minimize number of clusters selected
        obj=nbrClustSelected
        model.minimize(obj)
    elif(idObj==5): #minimize overall number of patterns selected
        obj=nbrOverallPat
        model.minimize(obj)
    elif(idObj==8): #minimize number of instances covered by at least one pattern.
        obj=nbrCoveredInstances
        model.minimize(obj)
    else:
        if(idObj==2): #maximize number of instances attributed to one and only one cluster
            obj=nbrInstOneClust
        elif(idObj==3): #maximize overall number of descriptors selected in cluster descriptions
            obj=nbrOverallPat
        elif(idObj==4): #maximize number of clusters selected
            obj=nbrClustSelected
        elif(idObj==7): #maximize number of instances covered by at least one pattern.
            obj=nbrCoveredInstances
        elif(idObj==9): #maximize number of instances covered by at least one pattern AND belonging to precisly one cluster.
            obj=nbrCoveredAndOneClust
        model.maximize(obj)

    # - Constraints -

    #number of clusters
    model += sum(F)>=Kmin
    model += sum(F)<=Kmax

    #number of apparition of the instances in selected clusters
    for i in range(N):
        model += (nbOfClusterForEachInstance[i]== sum([F[c]*instanceClusterMatrix[i][c] for c in range(V)]) )
        if(i in customAppar): #User constraint: specific boundaries for the number of apparition of a particular instance
            model += (nbOfClusterForEachInstance[i]>= customAppar[i][0])
            model += (nbOfClusterForEachInstance[i]<= customAppar[i][1])
        else:
            model += (nbOfClusterForEachInstance[i]>= apparMin)
            model += (nbOfClusterForEachInstance[i]<= apparMax)

    #number of instances attributed to more or less than 1 cluster:
    if(nbMaxLessThan1!=None):
        model+= ( nbrInstNoClust<=nbMaxLessThan1 )
    if(nbMaxMoreThan1!=None):
        model+= ( nbrInstMultipleClust<=nbMaxMoreThan1 )
    if(nbMinLessThan1!=None):
        model+= ( nbrInstNoClust>=nbMinLessThan1 )
    if(nbMinMoreThan1!=None):
        model+= ( nbrInstMultipleClust>=nbMinMoreThan1 )
    if(nbMaxDiffThan1!=None):
        model+= ( nbrInstNoClust+nbrInstMultipleClust<=nbMaxDiffThan1 )

    #Cannot select patterns as cluster descriptors if they cover too much of the instances in other selected clusters
    #not explicitly in article
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d]
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100 #max number of instances in c1 covered by outside patterns
                    if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                        #if c and c1 are selected, then the presence of each pattern of c is possible (not mandatory, ence the <= instead of <)
                        # iff the coverage of p on c1 is inferior to patThreshold
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c] <= int(clustPatDF[c1][p]<=patThreshold) )
                    else:
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c][d] <= int(clustPatDF[c1][p]<=patThreshold) )

    #if a pattern j is selected in the final description of a selected cluster,
    # then all other cluster for which j covers more then patThreshold elements cannot be selected.
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d] #get the corresponding pattern id
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100
                    if(clustPatDF[c1][p]>patThreshold):
                        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                            model += (candidateDescr[c]==1).implies(F[c1]==0)
                        else:
                            model += (candidateDescr[c][d]==1).implies(F[c1]==0)

    #Non-selected clusters must have all their candidate descriptors set to 0
    for c in range(V):
        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
            model += (candidateDescr[c]==F[c]) #the selection of the cluster must be complementary with the selection of its unique pattern
        else: #Note: we could formulate following constr as implications
            model += (F[c]<=sum(candidateDescr[c])) #cluster cannot be selected if at least one of its descr isn't
            for p in range(len(D[c])):
                model += (candidateDescr[c][p]<=F[c]) #descriptors cannot be selected if their cluster is not

    #Limit the number of patterns that can be selected for each cluster.
    #Note: if obj is not cov or discr, there is no constraint on the selection of patterns.
    if(maxNbPatPerClust!=None):
        for c in range(V):
            if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>" and len((candidateDescr[c])>maxNbPatPerClust)): #If more than one original descriptor
                model += (sum(candidateDescr[c])<=maxNbPatPerClust)

    else:   #does not need to work; unable to find results OR simple param pb ? #TODO this constraint needs to be modified if obj is to max the discrimination (via the else ?)
        #If a cluster is selected then all of its patterns that do not cover any other cluster are selected in its final description.
        for c in range(V):
            for d in range(len(D[c])):
                p=D[c][d]
                patThreshold=(maxCovOutPer*clustSizes[c])/100 #max number of instances in c1 covered by outside patterns
                covC=int(clustPatDF[c][p] > patThreshold)
                if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                    model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c]==F[c]) #if c is selected and the sum of clusters selected and covered by p (minus c) is equal to 0 than p must describe c
                else:
                    model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d]==F[c])


    #NEW: isInstanceCovered and isInstanceCovAnd1Clust constraints
    for i in range(N):
        #If point is not in any selected cluster it is not covered
        model += (nbOfClusterForEachInstance[i]==0).implies(isInstanceCovered[i]==False)

        #isInstanceCovAnd1Clust
        model += isInstanceCovAnd1Clust[i]==(nbOfClusterForEachInstance[i]==1 and isInstanceCovered[i]==True)
        model += (isInstanceCovered[i]==False).implies(isInstanceCovAnd1Clust[i]==False)
        model += (nbOfClusterForEachInstance[i]!=1).implies(isInstanceCovAnd1Clust[i]==False)

        #If want covered by any clust point, remove instClusterMatrix from constraint
        model += isInstanceCovered[i]==(
            sum([ F[c]*(sum(F[c]*instanceClusterMatrix[i][c]*candidateDescr[c]*instPatDF[listPat[D[c][d]]][i] for d in range(len(D[c])) if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"))
                       +sum(F[c]*instanceClusterMatrix[i][c]*candidateDescr[c][d]*instPatDF[listPat[D[c][d]]][i] for d in range(len(D[c])) if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>" ))
            )for c in range(V) ])>=1)

    #Preference and redundancy constraints
    if(patPreference!=0):
        for c in range(V):
            if(str(type(candidateDescr[c]))!="<class 'cpmpy.expressions.variables._IntVarImpl'>"):
                #for each pairs of patterns
                for d1 in range(len(D[c])):
                    p1=D[c][d1]
                    for d2 in range(len(D[c])):
                        p2=D[c][d2]
                        #Check if p1 is a subset of p2.
                        if(d1!=d2 and p1!=p2 and ast.literal_eval(listPat[p1])!=ast.literal_eval(listPat[p2]) and set(ast.literal_eval(listPat[p1])).issubset(ast.literal_eval(listPat[p2]))):
                            #print('DEBUG',listPat[p1],'is a subset of',listPat[p2],'; ds:',d1,d2,'; ps:',p1,p2)
                            #Redondancy constraint : Both cannot be selected at the same time.
                            model+= (candidateDescr[c][d1]+candidateDescr[c][d2]<2)

                            #Preference constraints
                            if(patPreference==1): #fav more general patterns (p2).
                                patThreshold=(maxCovOutPer*clustSizes[c])/100
                                covC=int(clustPatDF[c][p2] > patThreshold)
                                #If p2 can be selected (i.e. is discriminative), then p1 is forbidden.
                                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p2] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d1]==0)

                            elif(patPreference==2): #fav more specific patterns (p1).
                                patThreshold=(maxCovOutPer*clustSizes[c])/100
                                covC=int(clustPatDF[c][p1] > patThreshold)
                                #If p1 can be selected (i.e. is discriminative), then p2 is forbidden.
                                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p1] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d2]==0)

                           # elif(patPreference>2):
                                #print('No preferences')

    #User Constraints
    for c in MSs: #Must Select
        model += F[c]==1
    for (c1,c2) in CSs: #Cannot Select c1 and c2 at the same time
        model += F[c1]+F[c2]<2

    #Solver
    #if SolverLookup.get("ortools", model).solve(**tuner.best_params):
    if SolverLookup.get("ortools", model).solve():
        print("Number of instances not in any cluster : ",nbrInstNoClust.value())
        print("Number of instances in exactly one cluster : ",nbrInstOneClust.value())
        print("Number of instances in more than one cluster : ",nbrInstMultipleClust.value())
        print("Number of cluster selected : ",nbrClustSelected.value())
        print("Overall number of selected patterns : ", nbrOverallPat.value())
        print("Overall number of covered objects : ", nbrCoveredInstances.value()," out of",N)
        print("Overall number of covered and 1-clustered objects : ", nbrCoveredAndOneClust.value()," out of",N)
        print("objective value ("+str(idObj)+") : ",obj.value())
        print()
        return F,nbOfClusterForEachInstance,candidateDescr,obj
    else:
        print("No solution found")

    return

def launchCPmodelNewObjRephrasedPT(idObj: int, N: int, V: int, instanceClusterMatrix: list, Kmin: int, Kmax: int,
                        apparMin: int, apparMax: int, D: list, clustPatDF: pd.DataFrame, instPatDF: pd.DataFrame, listPat: list, clustSizes: list,
                        maxCovOutPer: int, nbMaxMoreThan1: int, nbMaxLessThan1: int, nbMaxDiffThan1: int,
                        nbMinMoreThan1: int, nbMinLessThan1: int, patPreference: int = 0,
                        MSs: list = [], CSs: list = [], customAppar: dict = {}, maxNbPatPerClust: int = 5):
    """
    CP model aiming at finding a partition or clustering of a dataset.
    """
    # Model
    model = Model()

    # Variables
    F = boolvar(shape=V, name="Clustering")
    nbOfClusterForEachInstance = intvar(0, V, shape=N, name="clustersPerInst")
    isInstanceCovered = boolvar(shape=N, name="covInst")
    candidateDescr = [boolvar(shape=len(D[c]), name=f"descr_{c}") if len(D[c]) > 1 else boolvar(name=f"descr_{c}") for c in range(V)]

    # Objectives
    nbrInstNoClust = (nbOfClusterForEachInstance == 0).sum()
    nbrInstOneClust = (nbOfClusterForEachInstance == 1).sum()
    nbrClustSelected = F.sum()
    nbrCoveredInstances = isInstanceCovered.sum()
    nbrOverallPat = sum(candidateDescr[c].sum() if isinstance(candidateDescr[c], np.ndarray) else candidateDescr[c] for c in range(V))

    obj_dict = {
        0: nbrInstNoClust,
        1: nbrClustSelected,
        5: nbrOverallPat,
        8: nbrCoveredInstances,
        2: nbrInstOneClust,
        3: nbrOverallPat,
        4: nbrClustSelected,
        7: nbrCoveredInstances,
        9: nbrCoveredInstances
    }

    if idObj in obj_dict:
        if idObj in [0, 1, 5, 8]:
            model.minimize(obj_dict[idObj])
        else:
            model.maximize(obj_dict[idObj])

    # Constraints
    model += [Kmin <= F.sum(), F.sum() <= Kmax]

    for i in range(N):
        model += (nbOfClusterForEachInstance[i] == sum(F[c] * instanceClusterMatrix[i][c] for c in range(V)))
        if i in customAppar:
            model += [nbOfClusterForEachInstance[i] >= customAppar[i][0],
                      nbOfClusterForEachInstance[i] <= customAppar[i][1]]
        else:
            model += [nbOfClusterForEachInstance[i] >= apparMin,
                      nbOfClusterForEachInstance[i] <= apparMax]

    if nbMaxLessThan1 is not None:
        model += (nbrInstNoClust <= nbMaxLessThan1)
    if nbMaxMoreThan1 is not None:
        model += ((N - (nbrInstNoClust + nbrInstOneClust)) <= nbMaxMoreThan1)
    if nbMinLessThan1 is not None:
        model += (nbrInstNoClust >= nbMinLessThan1)
    if nbMinMoreThan1 is not None:
        model += ((N - (nbrInstNoClust + nbrInstOneClust)) >= nbMinMoreThan1)
    if nbMaxDiffThan1 is not None:
        model += ((nbrInstNoClust + (N - (nbrInstNoClust + nbrInstOneClust))) <= nbMaxDiffThan1)

    for c in range(V):
        for c1 in range(V):
            if c1 != c:
                for d in range(len(D[c])):
                    p = D[c][d]
                    patThreshold = (maxCovOutPer * clustSizes[c1]) / 100
                    if isinstance(candidateDescr[c], np.ndarray):
                        model += (F[c] & F[c1]).implies(candidateDescr[c][d].implies(clustPatDF[c1][p] <= patThreshold))
                    else:
                        model += (F[c] & F[c1]).implies(candidateDescr[c].implies(clustPatDF[c1][p] <= patThreshold))

    for c in range(V):
        for c1 in range(V):
            if c1 != c:
                for d in range(len(D[c])):
                    p = D[c][d]
                    patThreshold = (maxCovOutPer * clustSizes[c1]) / 100
                    if clustPatDF[c1][p] > patThreshold:
                        if isinstance(candidateDescr[c], np.ndarray):
                            model += candidateDescr[c][d].implies(F[c1] == 0)
                        else:
                            model += candidateDescr[c].implies(F[c1] == 0)

    for c in range(V):
        if isinstance(candidateDescr[c], np.ndarray):
            model += (F[c] <= candidateDescr[c].sum())
        else:
            model += (F[c] == candidateDescr[c])

    if maxNbPatPerClust is not None:
        for c in range(V):
            if isinstance(candidateDescr[c], np.ndarray) and len(candidateDescr[c]) > maxNbPatPerClust:
                model += candidateDescr[c].sum() <= maxNbPatPerClust

    for i in range(N):
        model += (nbOfClusterForEachInstance[i] == 0).implies(isInstanceCovered[i] == False)
        model += isInstanceCovered[i] == (
            sum(F[c] * instanceClusterMatrix[i][c] * candidateDescr[c][d] * instPatDF[listPat[D[c][d]]][i] for c in range(V) for d in range(len(D[c])) if isinstance(candidateDescr[c], np.ndarray)) +
            sum(F[c] * instanceClusterMatrix[i][c] * candidateDescr[c] * instPatDF[listPat[D[c][0]]][i] for c in range(V) if not isinstance(candidateDescr[c], np.ndarray)) >= 1)

    if patPreference != 0:
        for c in range(V):
            if isinstance(candidateDescr[c], np.ndarray):
                for d1 in range(len(D[c])):
                    p1 = D[c][d1]
                    for d2 in range(len(D[c])):
                        p2 = D[c][d2]
                        if d1 != d2 and p1 != p2 and set(ast.literal_eval(listPat[p1])).issubset(ast.literal_eval(listPat[p2])):
                            model += (candidateDescr[c][d1] + candidateDescr[c][d2] < 2)
                            if patPreference == 1:
                                patThreshold = (maxCovOutPer * clustSizes[c]) / 100
                                model += (candidateDescr[c][d1] == 0).implies(
                                    sum(F[c1] * (clustPatDF[c1][p2] > patThreshold) for c1 in range(V)) == 0)
                            elif patPreference == 2:
                                patThreshold = (maxCovOutPer * clustSizes[c]) / 100
                                model += (candidateDescr[c][d2] == 0).implies(
                                    sum(F[c1] * (clustPatDF[c1][p1] > patThreshold) for c1 in range(V)) == 0)

    for c in MSs:
        model += F[c] == 1
    for c1, c2 in CSs:
        model += F[c1] + F[c2] < 2

    if SolverLookup.get("ortools", model).solve():
        print("Number of instances not in any cluster:", nbrInstNoClust.value())
        print("Number of instances in exactly one cluster:", nbrInstOneClust.value())
        print("Number of instances in more than one cluster:", (N - (nbrInstNoClust.value() + nbrInstOneClust.value())))
        print("Number of clusters selected:", nbrClustSelected.value())
        print("Overall number of selected patterns:", nbrOverallPat.value())
        print("Overall number of covered objects:", nbrCoveredInstances.value(), "out of", N)
        print("objective value (", idObj, "):", obj_dict[idObj].value())
        return F, nbOfClusterForEachInstance, candidateDescr, obj_dict[idObj]
    else:
        print("No solution found")


    return

def launchCPmodelMultiAnswer(idObj:int,N:int,V:int,instanceClusterMatrix:list,Kmin:int,Kmax:int,
                  apparMin:int,apparMax:int,D:list,clustPatDF:pd.DataFrame,clustSizes:list,
                  maxCovOutPer:int,nbMaxMoreThan1:int,nbMaxLessThan1:int,nbMaxDiffThan1:int,
                  nbMinMoreThan1:int,nbMinLessThan1:int,
                  objectiveValue:int,MSs:list=[],CSs:list=[],customAppar:dict={}):
    """
    CP model aiming at finding a partition or clustering of a dataset.

    Parameters
    --------
    idObj: int
        Objective criterion to apply.
        Objective criterions of the CP models are as follows:
            0 : minimize number of instances attributed to 0 cluster
            1 : minimize number of clusters selected
            5 : minimize overall number of patterns selected
            2 : maximize number of instances attributed to one and only one cluster
            3 : maximize overall number of descriptors selected in cluster descriptions
            4 : maximize number of clusters selected
    N: int
        number of instances in the dataset.
    V: int
        number of candidate clusters.
    instanceClusterMatrix : int matrix
        a N*V matrix where 1 for instance i and cluster c indicate that i is in c.
    Kmin : int
        minimum number of clusters that have to be selected.
    Kmax : int
        maximum number of clusters that can be selected.
    apparMin : int
        minimal number of apparition of each instances in the partition (i.e. minimal number of selected cluster each instance has to be attributed to).
    apparMax :int
    maximal number of apparition of each instances in the partition.
    D :
        list of the descriptions of each of the candidate V clusters.
    clustPatDF :
            dataframe displayining for each candidate pattern the number of instances it covers in each candidate cluster.
    clustSizes : List of int
            a list containing the size of all candidate clusters.
    maxCovOutPer : int
            a percentage of maximum outside coverage (dsicriminativeness parameter).
    nbMaxMoreThan1 : int
            maximum number of instances attributed to more than 1 cluster.
    nbMaxLessThan1 : int
            maximum number of instances attributed to less than 1 cluster.
    nbMaxDiffThan1 : int
            maximum number of instances attributed to anything different than 1 cluster.
    MSs : list of int
            Must-Select constraints: list of candidate cluster ids that must be selected in the final clustering.
    CSs : list of tuple of int
            Cannot-Select constraints: list of couples of cluster ids that are not allowed to be simultaniously selected.
    customAppar :
        specific boundaries for the number of apparition of particular instances.

    Returns
    --------
    F: list of int
        Final Clustering: Fi=1 if candidate cluster i is selected, 0 otherwise.
    nbOfClusterForEachInstance: list of int
        for each instance, number of selected cluster to which it belongs.
    candidateDescr: list of lists of int
        1 or 0 depending if corresponding candidate descriptor is selected.
    """

    #Model
    model = Model()

    #Variables
    F = intvar(0,1, shape=V, name="Clustering")
    nbOfClusterForEachInstance= intvar(0,V, shape=N, name="clustersPerInst")
    candidateDescr=[]
    for c in range(V):
        candidateDescr.append(intvar(0,1, shape=len(D[c]), name="descr "+str(c)))

    #Objectives
    nbrInstNoClust=(nbOfClusterForEachInstance==0).sum() #number of instances attributed to 0 cluster
    nbrInstOneClust=(nbOfClusterForEachInstance==1).sum() #number of instances attributed to one and only one cluster
    nbrInstMultipleClust=N-(nbrInstNoClust+nbrInstOneClust) #number of instances attributed to 0 cluster
    nbrClustSelected=sum(F) #number of clusters selected
    nbrOverallPat=sum( sum([candidateDescr[c]]) for c in range(V) ) #overall number of descriptors selected in cluster descriptions

    if(idObj==0): #minimize number of instances attributed to 0 cluster
        obj=nbrInstNoClust
        #model.minimize(obj)
    elif(idObj==1): #minimize number of clusters selected
        obj=nbrClustSelected
        #model.minimize(obj)
    elif(idObj==5): #minimize overall number of patterns selected
        obj=nbrOverallPat
        #model.minimize(obj)
    else:
        if(idObj==2): #maximize number of instances attributed to one and only one cluster
            obj=nbrInstOneClust
        elif(idObj==3): #maximize overall number of descriptors selected in cluster descriptions
            obj=nbrOverallPat
        elif(idObj==4): #maximize number of clusters selected
            obj=nbrClustSelected
        #model.maximize(obj)

    # - Objective value fixing -

    model += (obj==objectiveValue)

    # - Constraints -

    #number of clusters
    model += sum(F)>=Kmin
    model += sum(F)<=Kmax

    #number of apparition of the instances in selected clusters
    for i in range(N):
        model += (nbOfClusterForEachInstance[i]== sum([F[c]*instanceClusterMatrix[i][c] for c in range(V)]) )
        if(i in customAppar): #User constraint: specific boundaries for the number of apparition of a particular instance
            model += (nbOfClusterForEachInstance[i]>= customAppar[i][0])
            model += (nbOfClusterForEachInstance[i]<= customAppar[i][1])
        else:
            model += (nbOfClusterForEachInstance[i]>= apparMin)
            model += (nbOfClusterForEachInstance[i]<= apparMax)

    #number of instances attributed to more or less than 1 cluster:
    model+= ( nbrInstNoClust<=nbMaxLessThan1 )
    model+= ( nbrInstMultipleClust<=nbMaxMoreThan1 )
    model+= ( nbrInstNoClust>=nbMinLessThan1 )
    model+= ( nbrInstMultipleClust>=nbMinMoreThan1 )
    model+= ( nbrInstNoClust+nbrInstMultipleClust<=nbMaxDiffThan1 )

    #Cannot select patterns as cluster descriptors if they cover too much of the instances in other selected clusters
    #not explicitly in article
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d]
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100 #max number of instances in c1 covered by outside patterns
                    if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                        #if c and c1 are selected, then the presence of each pattern of c is possible (not mandatory, ence the <= instead of <)
                        # iff the coverage of p on c1 is inferior to patThreshold
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c] <= int(clustPatDF[c1][p]<=patThreshold) )
                    else:
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c][d] <= int(clustPatDF[c1][p]<=patThreshold) )

    #if a pattern j is selected in the final description of a selected cluster,
    # then all other cluster for which j covers more then patThreshold elements cannot be selected.
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d] #get the corresponding pattern id
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100
                    if(clustPatDF[c1][p]>patThreshold):
                        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                            model += (candidateDescr[c]==1).implies(F[c1]==0)
                        else:
                            model += (candidateDescr[c][d]==1).implies(F[c1]==0)

    #Non-selected clusters must have all their candidate descriptors set to 0
    for c in range(V):
        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
            model += (candidateDescr[c]==F[c]) #the selection of the cluster must be complementary with the selection of its unique pattern
        else: #Note: we could formulate following constr as implications
            model += (F[c]<=sum(candidateDescr[c])) #cluster cannot be selected if at least one of its descr isn't
            for p in range(len(D[c])):
                model += (candidateDescr[c][p]<=F[c]) #descriptors cannot be selected if their cluster is not

    #If a cluster is selected then all of its patterns that do not cover any other cluster is selected in its final description.
    for c in range(V):
        for d in range(len(D[c])):
            p=D[c][d]
            patThreshold=(maxCovOutPer*clustSizes[c])/100 #max number of instances in c1 covered by outside patterns
            covC=int(clustPatDF[c][p] > patThreshold)
            if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c]==F[c]) #if c is selected and the sum of clusters selected and covered by p (minus c) is equal to 0 than p must describe c
            else:
                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d]==F[c])

    #User Constraints
    for c in MSs: #Must Select
        model += F[c]==1
    for (c1,c2) in CSs: #Cannot Select c1 and c2 at the same time
        model += F[c1]+F[c2]<2

    #Solver
    nbrInstNoClustList=[]
    nbrInstOneClustList=[]
    nbrClustSelectedList=[]
    nbrOverallPatList=[]
    F_List=[]
    nbOfClusterForEachInstance_List=[]
    candidateDescr_List=[]
    #TODO using multiple solutions
    def collect():
        print('Find a new optimal solution with obj'+str(idObj)+" =",obj.value())
        nbrInstNoClustList.append(nbrInstNoClust.value())
        nbrInstOneClustList.append(nbrInstOneClust.value())
        nbrClustSelectedList.append(nbrClustSelected.value())
        nbrOverallPatList.append(nbrOverallPat.value())
        F_List.append(F.value())
        nbOfClusterForEachInstance_List.append(nbOfClusterForEachInstance.value())
        #candidateDescr_List.append(candidateDescr.value())
        candidateDescr_List.append([candidateDescr[e].value() for e in range(len(candidateDescr))])

    #if SolverLookup.get("ortools", model).solve(**tuner.best_params):
    nSol=SolverLookup.get("ortools", model).solveAll(display=collect, solution_limit=200) #cpmpy.exceptions.NotSupportedError: OR-tools does not support finding all optimal solutions.
    #nSol=model.solveAll(display=collect, solution_limit=100)
    if nSol>0:
        print(nSol,'solutions found !')
    else:
        print("No solution found")

    return nbrInstNoClustList,nbrInstOneClustList,nbrClustSelectedList,nbrOverallPatList,F_List,nbOfClusterForEachInstance_List,candidateDescr_List

# -- END launchCPmodelMultiAnswer --


#adds WCSS criteria
def launchCPmodelWCSS(idObj:int,N:int,V:int,instanceClusterMatrix:list,Kmin:int,Kmax:int,
                  apparMin:int,apparMax:int,D:list,clustPatDF:pd.DataFrame,clustSizes:list,
                  maxCovOutPer:int,nbMaxMoreThan1:int,nbMaxLessThan1:int,nbMaxDiffThan1:int,
                  clustWCSSs:list,MSs:list,CSs:list,customAppar:dict={}):
    #Model
    model = Model()

    #Variables
    F = intvar(0,1, shape=V, name="Clustering")
    nbOfClusterForEachInstance= intvar(0,V, shape=N, name="clustersPerInst")
    candidateDescr=[]
    for c in range(V):
        candidateDescr.append(intvar(0,1, shape=len(D[c]), name="descr "+str(c)))

    #Objectives
    nbrInstNoClust=(nbOfClusterForEachInstance==0).sum() #number of instances attributed to 0 cluster
    nbrInstOneClust=(nbOfClusterForEachInstance==1).sum() #number of instances attributed to one and only one cluster
    nbrInstMultipleClust=N-(nbrInstNoClust+nbrInstOneClust) #number of instances attributed to 0 cluster
    nbrClustSelected=sum(F) #number of clusters selected
    nbrOverallPat=sum( sum([candidateDescr[c]]) for c in range(V) ) #overall number of descriptors selected in cluster descriptions

    wcss=intvar(0,sum(clustWCSSs))
    div=intvar(1,V) #Only needed to compute wcss

    if(idObj==0): #minimize number of instances attributed to 0 cluster
        obj=nbrInstNoClust
        model.minimize(obj)
    elif(idObj==1): #minimize number of clusters selected
        obj=nbrClustSelected
        model.minimize(obj)
    elif(idObj==5): #minimize overall number of patterns selected
        obj=nbrOverallPat
        model.minimize(obj)
    elif(idObj==6):
        obj=wcss
        model.minimize(obj)
    else:
        if(idObj==2): #maximize number of instances attributed to one and only one cluster
            obj=nbrInstOneClust
        elif(idObj==3): #maximize overall number of descriptors selected in cluster descriptions
            obj=nbrOverallPat
        elif(idObj==4): #maximize number of clusters selected
            obj=nbrClustSelected
        model.maximize(obj)

    # - Constraints -

    #WCSS
    model+= div==max([sum(F),1]) #Cannot do a division with a variable whose domain starts at 0, so need to do this
    model += wcss==sum([F[c]*clustWCSSs[c] for c in range(len(F))])//div

    #number of clusters
    model += sum(F)>=Kmin
    model += sum(F)<=Kmax

    #number of apparition of the instances in selected clusters
    for i in range(N):
        model += (nbOfClusterForEachInstance[i]== sum([F[c]*instanceClusterMatrix[i][c] for c in range(V)]) )
        if(i in customAppar): #User constraint: specific boundaries for the number of apparition of a particular instance
            model += (nbOfClusterForEachInstance[i]>= customAppar[i][0])
            model += (nbOfClusterForEachInstance[i]<= customAppar[i][1])
        else:
            model += (nbOfClusterForEachInstance[i]>= apparMin)
            model += (nbOfClusterForEachInstance[i]<= apparMax)

    #number of instances attributed to more or less than 1 cluster:
    model+= ( nbrInstNoClust<=nbMaxLessThan1 )
    model+= ( nbrInstMultipleClust<=nbMaxMoreThan1 )
    model+= ( nbrInstNoClust+nbrInstMultipleClust<=nbMaxDiffThan1 )

    #Cannot select patterns as cluster descriptors if they cover too much of the instances in other selected clusters
    #not explicitly in article
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d]
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100 #max number of instances in c1 covered by outside patterns
                    if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                        #if c and c1 are selected, then the presence of each pattern of c is possible (not mandatory, ence the <= instead of <)
                        # iff the coverage of p on c1 is inferior to patThreshold
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c] <= int(clustPatDF[c1][p]<=patThreshold) )
                    else:
                        model += (F[c]==1 and F[c1]==1).implies(candidateDescr[c][d] <= int(clustPatDF[c1][p]<=patThreshold) )

    #if a pattern j is selected in the final description of a selected cluster,
    # then all other cluster for which j covers more then patThreshold elements cannot be selected.
    for c in range(V):
        for c1 in range(V):
            if(c1!=c):
                for d in range(len(D[c])):
                    p=D[c][d] #get the corresponding pattern id
                    patThreshold=(maxCovOutPer*clustSizes[c1])/100
                    if(clustPatDF[c1][p]>patThreshold):
                        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                            model += (candidateDescr[c]==1).implies(F[c1]==0)
                        else:
                            model += (candidateDescr[c][d]==1).implies(F[c1]==0)

    #Non-selected clusters must have all their candidate descriptors set to 0
    for c in range(V):
        if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
            model += (candidateDescr[c]==F[c]) #the selection of the cluster must be complementary with the selection of its unique pattern
        else: #Note: we could formulate following constr as implications
            model += (F[c]<=sum(candidateDescr[c])) #cluster cannot be selected if at least one of its descr isn't
            for p in range(len(D[c])):
                model += (candidateDescr[c][p]<=F[c]) #descriptors cannot be selected if their cluster is not

    #If a cluster is selected then all of its patterns that do not cover any other cluster is selected in its final description.
    for c in range(V):
        for d in range(len(D[c])):
            p=D[c][d]
            patThreshold=(maxCovOutPer*clustSizes[c])/100 #max number of instances in c1 covered by outside patterns
            covC=int(clustPatDF[c][p] > patThreshold)
            if(str(type(candidateDescr[c]))=="<class 'cpmpy.expressions.variables._IntVarImpl'>"): #If only one original descriptor
                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c]==F[c]) #if c is selected and the sum of clusters selected and covered by p (minus c) is equal to 0 than p must describe c
            else:
                model+= ((F[c]==1)and( sum([ F[c1]*int(clustPatDF[c1][p] >(maxCovOutPer*clustSizes[c1])/100 ) for c1 in range(V) ])-covC==0)).implies(candidateDescr[c][d]==F[c])

    #User Constraints
    for c in MSs: #Must Select
        model += F[c]==1
    for (c1,c2) in CSs: #Cannot Select c1 and c2 at the same time
        model += F[c1]+F[c2]<2

    #Solver
    #if SolverLookup.get("ortools", model).solve(**tuner.best_params):
    if SolverLookup.get("ortools", model).solve():
        print("Number of instances not in any cluster : ",nbrInstNoClust.value())
        print("Number of instances in exactly one cluster : ",nbrInstOneClust.value())
        print("Number of cluster selected : ",nbrClustSelected.value())
        print("Overall number of selected patterns : ", nbrOverallPat.value())
        print("Mean WCSS: ",wcss.value())
        print("objective value ("+str(idObj)+") : ",obj.value())
        print()
        return F,nbOfClusterForEachInstance,candidateDescr
    else:
        print("No solution found")

    return


#-----------------------------

def launchICES(dataName,dataPath,resultFolderPath,resultFileName,patternCoveragePercentage,maxCovOutPer,patType,
               bpKmin,bpKmax,finalKmin,finalKmax,baseAlgorithms,idObj,
               minAppar=0,maxAppar=2,nbMaxMoreThan1=400,nbMaxLessThan1=400,nbMinLessThan1=0,nbMinMoreThan1=0,
               precompBC=[],precompFiltered=[],tsne=False,repeatBPgen=1,maxNbPatPerClust=5,
               actKinPer=50,maxPharmaSize=7,ncEx=None,
               modelExec:bool=True,treeCompare:bool=False,verbose:bool=False,enforcedClusters=[],precomputedObjValue=None,
               allowInclusion:bool=True,patPreference:int=0,gaussianNoise=(0,0)):
    """ Launches our full approach.

    Parameters
    --------
    idObj: int
        Objective criterion to apply in the CP model. Options are as follows:
            0 : minimize number of instances attributed to 0 cluster
            1 : minimize number of clusters selected
            5 : minimize overall number of patterns selected
            2 : maximize number of instances attributed to one and only one cluster
            3 : maximize overall number of descriptors selected in cluster descriptions
            4 : maximize number of clusters selected.
            6 : maximize number of covered instances."""

    precomputedDist=[]
    precomputedSim=[]

    #Choose and read the dataset
    ifif(dataName in ["Automobile","automobile","auto","Auto","AUTOMOBILE","car","cars"]):
        #featureSpace,tagSpace,tagNames=prepareAutomobileData(readAutomobileData(dataPath,False)) #AUTOMOBILE data, deprecated
        numVals,binVals,binValsNames,prices,convertedVals,convertedValsNames,priceDistances,wholeDescriptorSpace,wholeDescriptorSpaceNames=prepareAutomobileData(readAutomobileData(dataPath,False)) #AUTOMOBILE data
        featureSpace=prices
        featureSpace=[[f,1] for f in featureSpace]
        tagSpace=wholeDescriptorSpace
        tagNames=wholeDescriptorSpaceNames
        precomputedSim=convertDistInSim(priceDistances)

        #allClust=[[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 84, 85, 86, 87, 88, 89, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 132, 133, 134, 140, 141, 142, 143, 144, 145, 146, 147], [0, 18, 30, 42, 71, 72, 73, 77, 78, 91, 92, 104, 108, 129, 130, 136, 148, 149], [1, 2, 3, 4, 5, 6, 7, 32, 43, 44, 45, 46, 47, 48, 74, 75, 76, 79, 80, 81, 82, 83, 90, 93, 94, 95, 96, 131, 137, 138, 139, 150, 151, 152, 153, 154, 155, 156, 157, 158]]
        #allClust=[[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 84, 85, 86, 87, 88, 89, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 132, 133, 134, 140, 141, 142, 143, 144, 145, 146, 147], [0, 18, 30, 42, 71, 72, 73, 77, 78, 91, 92, 104, 108, 129, 130, 135, 136, 148, 149], [1, 2, 3, 4, 5, 6, 7, 32, 43, 44, 45, 46, 47, 48, 74, 75, 76, 79, 80, 81, 82, 83, 90, 93, 94, 95, 96, 131, 137, 138, 139, 150, 151, 152, 153, 154, 155, 156, 157, 158]]
        #nbOfClusterForEachInstance=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        #nbOfClusterForEachInstance=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        #allClustLen=[100, 18, 40]
        #allClustLen=[100, 19, 40]-
        #labels=genLabels(allClust,len(featureSpace),nbOfClusterForEachInstance)
        #print(len(labels),len(nbOfClusterForEachInstance),max(labels))
        #tagSpace bc featurespace only contains prices
        #vizualizePartition1D(len(allClust),featureSpace,labels,resultFolderPath)
        #print(z)
    elif(dataName in ["flags","FLAGS","flag","Flag","Flags","FLAG"]):
        wholeFlagSpace,wholeCountrySpace,wholeFlagSpaceNames,wholeCountrySpaceNames=loadFlagsData(dataPath) #FLAGS data, full flag view
        featureSpace=wholeFlagSpace
        tagSpace=wholeFlagSpace
        tagNames=wholeFlagSpaceNames
    elif(dataName in ["flagsC","FLAGSC","flagC","FlagC","FlagsC","FLAGC","country","countries","flagCountries","flagsCompet"]):
        wholeFlagSpace,wholeCountrySpace,wholeFlagSpaceNames,wholeCountrySpaceNames=loadFlagsData(dataPath) #FLAGS data, full country view
        featureSpace=wholeCountrySpace
        tagSpace=wholeCountrySpace
        tagNames=wholeCountrySpaceNames
    elif(dataName in ["FlagVSCountries","FlagsVSCountries","FlagVSCountry","flagsVScountries","flagVScountry","fVSc","FVSC"]):
        wholeFlagSpace,wholeCountrySpace,wholeFlagSpaceNames,wholeCountrySpaceNames=loadFlagsData(dataPath) #FLAGS data, full country view
        featureSpace=wholeFlagSpace
        tagSpace=wholeCountrySpace
        tagNames=wholeCountrySpaceNames
    elif(dataName in ["CountriesVSFlag","CountriesVSFlags","CountryVSFlag","countriesVSflags","countryVSflag","cVSf","CVSF"]):
        wholeFlagSpace,wholeCountrySpace,wholeFlagSpaceNames,wholeCountrySpaceNames=loadFlagsData(dataPath) #FLAGS data, full country view
        featureSpace=wholeCountrySpace
        tagSpace=wholeFlagSpace
        tagNames=wholeFlagSpaceNames
    elif(dataName in ["adult","Adult","ADULT"]):
        featureSpace,tagSpace,tagNames=prepareAdultDataset(dataPath) #ADULT data
        #featureSpace,tagSpace,tagNames,precomputedSim=prepareAdultDataset() #ADULT data
    elif(dataName in ["iris","Iris","IRIS","UCIiris","irisCompet","irisTest"]):
        featureSpace,tagSpace,tagNames,groundTruth,classNames=loadUCIDataset([2,4]) #UCI Iris dataset. 2: median, 4: quartiles
    #elif(dataName in ["art","Art","artificial","Artificial","small","Small"]):
    #    featureSpace,tagSpace,tagNames,groundTruth=prepareArtif() #artificial visual dataset
    elif(dataName in ["circles",'Circles','circle','Circle']):
        featureSpace,groundTruth,tagNames,tagSpace=prepareCirclesRandom(precomputed=True)

    elif(dataName in ["halfmoon",'Halfmoon','half','moons']):
        featureSpace,groundTruth,tagNames,tagSpace,clustHalfmoon,colorsPart,shapesPart,sizesPart,discretizedTag,discretizedTagNames=prepareHalfmoonRandom(precomputed=True,showDescr=False)
        #featureSpace,groundTruth,tagNames,tagSpace,clustHalfmoon,colorsPart,shapesPart,sizesPart,discretizedTag,discretizedTagNames=prepareHalfmoonRandom(shapeBias=False,showDescr=True)
        #featureSpace,groundTruth,tagNames,tagSpace,clustHalfmoon,colorsPart,shapesPart,sizesPart,discretizedTag,discretizedTagNames=prepareHalfmoonRandom()
    elif(dataName in ["halfmoonNeg",'HalfmoonNeg','halfNeg','moonsNeg']):
        featureSpace,groundTruth,tagNames,tagSpace,clustHalfmoon=prepareHalfmoonRandom(precomputed=True)
        tagNames,tagSpace=createNegTagSpace(tagNames,tagSpace)
        from skmine.itemsets import LCM
        for k in range(2):
            tagsInC=getTagsIdsForClusterInstances(tagSpace,clustHalfmoon[k])
            lcm = LCM(min_supp=1)
            patterns = lcm.fit_discover(tagsInC)
            patK= patterns.values.tolist()
            patKnames=[[[tagNames[t] for t in pat[0]],pat[1]] for pat in patK]
            print('patK:',patK)
            print(patKnames)

        print(z)
    elif(dataName in ["AWA2","AwA2","awa2","AWA","AwA","aWa","awa","animals"]):
            featureSpace,tagSpace,tagNames,groundTruth,classNames=prepareAWA2() #AWA2 dataset
    elif(dataName in ["Students","Student","student","students"]):
        featureSpace,tagSpace,tagNames=prepareStudentPerfDataset(dataPath)#0) #Student Performance dataset
    elif(dataName in ['SecondaryMushroom','secondaryMushroom','smushroom','sMushroom']):
        featureSpace,tagSpace,tagNames,gt=prepareSecondaryMushroom()
    elif(dataName in ["Zoo","zoo"]):
        featureSpace,tagSpace,K,groundTruth,featureNames,tagNames,instanceNames=prepareZoo()
    elif(dataName in ['artif','Artificial','artif0']):
        featureSpace,tagSpace,tagNames,gtAttrMat=artifDataGeneration(overlapOpt=0)
        #print(featureSpace,tagSpace,tagNames)
        #print(z)
    elif(dataName in ['artif1','artfiv1']):
        featureSpace,tagSpace,tagNames,gtAttrMat=artifDataGeneration(overlapOpt=1)
    elif(dataName in ['artif2','artifv2']):
        featureSpace,tagSpace,tagNames,gtAttrMat=artifDataGeneration(overlapOpt=2)
    elif(dataName in ['artif3','artifv3']):
        featureSpace,tagSpace,tagNames,gtAttrMat=artifDataGeneration(overlapOpt=3)
    elif(dataName in ['artif4','artifv4']):
        featureSpace,tagSpace,tagNames,gtAttrMat=artifDataGeneration(overlapOpt=4)

    elif(dataName in ['NC','NewCaledonia','nc','NC1014','NC5690','Thibaut'] or ('NC' in dataName)):
        if(ncEx==None or ncEx>=8 or ncEx<0):
            featureSpace,tagSpace,tagNames, gt=prepareNCdataset(dataPath)
        else:
            featureSpace,tagSpace,tagNames, gt=prepareNCexercice(dataPath,ncEx)


    #Add noise:
    if(gaussianNoise!=(0,0)):
        featureSpace=np.array(featureSpace)
        noise = np.random.normal(loc=0, scale=1.5, size=featureSpace.shape)
        featureSpace = featureSpace + noise
        print('Gaussian noise added:',gaussianNoise)

    #Comparison
    if(treeCompare):
        #testDream(np.array(tagSpace),np.array(tagSpace),[0 for i in range(len(tagSpace))],6,tagNames,dataName,resultFolderPath+"_testDreaM",n_init = 1,plot=False)
        #dreamTime,part,convRules,convRulesId=testDream(np.array(tagSpace),np.array(featureSpace),[0 for i in range(len(tagSpace))],3,tagNames,dataName,resultFolderPath+"_testDreaM",n_init = 1,plot=False)
        #analyseDreaM(dataName,3,len(tagSpace),part,convRulesId)
        #print(z)

        K=finalKmin
        kmList=[]
        #_,BaseP=genBaseKmeans(featureSpace,2,3,200)
        #kmList=BaseP[0]
        kmeans_model = KMeans(n_clusters=K,n_init=10).fit(featureSpace) #Added n_init=10 because it is the default value, that need to be explicitlfy specified due to update 1.4 of sklearn
        labels = kmeans_model.labels_
        #kmList.append(list(labels))
        #kmList.append(kmeans_model) #ValueError: X has 8 features, but KMeans is expecting 2 features as input.

        IMMres=TreePipeline(tagSpace,[0 for i in range(len(tagSpace))],K,dataName,kmList=kmList,plotTree=False)
        #IMMres=TreePipeline(featureSpace,[0 for i in range(len(tagSpace))],K,dataName,kmList=kmList,plotTree=True)
        kmlab,treelab,Expl,qualityMeasures,exkmcTime=IMMres[0]
        #print(IMMres[0])
        showPartition(K,featureSpace,treelab)
        #print(z)
        opt=[]
        if(dataName in ["halfmoon",'Halfmoon','half','moons'] or dataName in ["circles",'Circles','circle','Circle']):
            opt=(featureSpace,groundTruth,tagNames,tagSpace)
        TKMpart,Tdescr,Tlens,Tmetrics=analyseDreaM(dataName,K,len(tagSpace),kmlab,Expl,opt=opt)
        treeKMARI=round(metrics.adjusted_rand_score(TKMpart,groundTruth),2)
        treeARI=round(metrics.adjusted_rand_score(treelab,groundTruth),2)
        print('treeARI:',treeARI)
        treeRes=[treelab,Tdescr,Tlens,Tmetrics,treeARI]
        print('--- End tree baseline ---')
        print(z)
    else:
        treeRes=[]

    maxClustSize=len(featureSpace)
    maxCoveredOutOfClusterOption=len(featureSpace)-1 #Dataset-wise discr.
    kmeansmin=bpKmin
    kmeansmax=bpKmax

    if(precompFiltered==[]):
        filteringRes=main(baseAlgorithms,featureSpace,tagSpace,repeatBPgen,maxClustSize,patType,patternCoveragePercentage,maxCoveredOutOfClusterOption,kmeansmin,kmeansmax,precomputedSim=precomputedSim,precompBC=precompBC,enforcedClusters=enforcedClusters,allowInclusion=allowInclusion)
    else:
        filteringRes=precompFiltered

    resultTextPath=resultFolderPath+"/output-"+resultFileName+".txt"

    if((filteringRes!=None) and filteringRes!=[]) :#Handle case where no candidate clusters where found
        #basePartitions,baseClustersIds,N,V,nonEmptyClustersIds,instanceClusterMat,allIntDescr,clustPatDF,patternDF,listPat,selectedClustersSize,times
        BasePartitions,BP,N,V,clusterIds,instanceClusterMat,D,clustPatDF,instPatDF,listPat,clustSizes,times,endEnforcedInd=filteringRes
        if(precompFiltered!=[]): #remove time of previous exec
            if(times!=[]):
                times.pop()
    else:
        params=[baseAlgorithms,bpKmin,bpKmax,repeatBPgen,maxClustSize,patternCoveragePercentage,patType]
        paramsNames=['baseAlgorithms','bpKmin','bpKmax','repeatBPgen','maxClustSize','Coverage','Pattern type']
        res=["No candidate clusters found."]
        resNames=['times','Error type: ']
        writeResults(resultTextPath,params,res,paramsNames,resNames)
        return None

    Kmin=finalKmin
    Kmax=finalKmax
    if(nbMaxMoreThan1==None or nbMaxLessThan1==None):
        nbMaxDiffThan1=None
    else:
        nbMaxDiffThan1=nbMaxMoreThan1+nbMaxLessThan1 #600

    if(endEnforcedInd!=[] and endEnforcedInd!=None):
        print('Enforced clusters among the candidates:',endEnforcedInd)

    start_time_CPmodel=time.time()
    if modelExec:
        print('Start CP model with obj',idObj)
        if(idObj==6):
            clustWCSSs=computeWCSSs(featureSpace,clusterIds)
            res=launchCPmodelWCSS(idObj,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,clustWCSSs,[],[])
        else:
            if(precomputedObjValue==None):
                #print('instPatDF.columns:',instPatDF.columns,len(instPatDF.columns))
                #print(listPat,len(listPat))
                #print(z)
                #res=launchCPmodel(idObj,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,MSs=endEnforcedInd)
                #res=launchCPmodel(idObj,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,nbMinMoreThan1,nbMinLessThan1,MSs=endEnforcedInd)
                #res=-(3,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,instPatDF,listPat,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,nbMinMoreThan1,nbMinLessThan1,MSs=endEnforcedInd)
                #res=launchCPmodelNewObj(idObj,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,instPatDF,listPat,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,nbMinMoreThan1,nbMinLessThan1,MSs=endEnforcedInd,maxNbPatPerClust=maxNbPatPerClust,patPreference=patPreference)
                res=launchCPmodelNewObjRephrased(idObj,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,instPatDF,listPat,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,nbMinMoreThan1,nbMinLessThan1,MSs=endEnforcedInd,maxNbPatPerClust=maxNbPatPerClust,patPreference=patPreference)
            else:
                resMultiAnswer=launchCPmodelMultiAnswer(idObj,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,nbMinMoreThan1,nbMinLessThan1,MSs=endEnforcedInd,objectiveValue=precomputedObjValue)
                nbrInstNoClustList,nbrInstOneClustList,nbrClustSelectedList,nbrOverallPatList,F_List,nbOfClusterForEachInstance_List,candidateDescr_List=resMultiAnswer

                allClust_List=[]
                allClustLen_List=[]
                PCRs_List=[]
                DCs_List=[]
                IPCs_List=[]
                patPCRs_List=[]
                patDCs_List=[]
                patIPCs_List=[]
                descrNames_List=[]
                descrPatIds_List=[]
                descrTagsIds_List=[]
                for sol_id in range(len(F_List)):
                    F_t=F_List[sol_id]
                    candidateDescr_t=candidateDescr_List[sol_id]
                    allClust,allClustLen=findSelectedClustersSizes(clusterIds,F_t)
                    allClust_List.append(allClust)
                    allClustLen_List.append(allClustLen)
                    descrNames,descrPatIds,descrTagsIds=genDescriptionNames(D,F_t,candidateDescr_t,tagNames,listPat)
                    descrNames_List.append(descrNames)
                    descrPatIds_List.append(descrPatIds)
                    descrTagsIds_List.append(descrTagsIds)
                    selClustIds=[i for i in range(len(F_t)) if F_t[i]==1] #ids of the selected clusters
                    PCRs,DCs,IPSs,SINGs,IPCs,patPCRs,patDCs,patIPSs,patSINGs,patIPCs=computeNovelDescrQuality(len(allClust),N,allClust,instPatDF.to_numpy(),clustPatDF.to_numpy(),selClustIds,descrPatIds,allClustLen,patternCoveragePercentage)
                    PCRs_List.append(PCRs)
                    DCs_List.append(DCs)
                    IPCs_List.append(IPCs)
                    patPCRs_List.append(patPCRs)
                    patDCs_List.append(patDCs)
                    patIPCs_List.append(patIPCs)
                    #clusters,descr,kin,tagNames,PCRs,DCs,IPCs,unattr,groupAndFam,BP,afs,ufs,
                print('DEBUG MultiAnswer with '+str(len(F_List))+' different solutions.')#:',F_List)
                #optionalRes=(allKins,allKinsName,patPCRs,patIPCs,featureSpace,assignedFeatureSpace,objValue)
                allKins=[]
                allKinsName=[]
                return N,BP,allClust_List,allClustLen_List,descrNames_List,descrPatIds_List,descrTagsIds_List,nbOfClusterForEachInstance_List,PCRs_List,DCs_List,IPCs_List,patPCRs,patDCs_List,patIPCs_List,precomputedObjValue,allKins,allKinsName
                print(z)
                return


        if res is None:
            params=[baseAlgorithms,repeatBPgen,maxClustSize,patternCoveragePercentage,patType,V,clustPatDF.shape,Kmin,Kmax,minAppar,maxAppar,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,idObj]
            paramsNames=['baseAlgorithhms','repeatBPgen','maxClustSize','patternCoveragePercentage','patType','V','clustPatDF.shape','Kmin','Kmax','minAppar','maxAppar','maxCovOutPer','nbMaxMoreThan1','nbMaxLessThan1','nbMaxDiffThan1','idObj']
            res=[times,"No solution found by the CP model"]
            resNames=['times','Error type: ']
            writeResults(resultTextPath,params,res,paramsNames,resNames)
            return None

        F,nbOfClusterForEachInstance,candidateDescr,obj=res#F,nbOfClusterForEachInstance,candidateDescr=launchCPmodelZeta(idObj,N,V,instanceClusterMat,Kmin,Kmax,minAppar,maxAppar,D,clustPatDF,clustSizes,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,Zeta,ZetaType,[],[])
        F=F.value()
        nbOfClusterForEachInstance=nbOfClusterForEachInstance.value()
        objValue=obj.value()
        candidateDescr=[candidateDescr[i].value() for i in range(len(candidateDescr))]
    else:
        F=[1]*len(clusterIds)
        nbOfClusterForEachInstance=[]
        for i in range(len(featureSpace)):
            aI=0
            for c in clusterIds:
                if i in c:
                    aI=aI+1
            nbOfClusterForEachInstance.append(aI)
        #nbOfClusterForEachInstance=[1]*len(featureSpace)
        candidateDescr=[[1]*len(Di) for Di in D]
    model_execution_time=time.time()-start_time_CPmodel
    print("      -- END CP model --")
    print("Model execution time : ",round(model_execution_time,2))
    times.append(round(model_execution_time,2))

    print()
    nbNotAttrib=0
    nbMultiAttrib=0
    if(modelExec):
        nbNotAttrib=(nbOfClusterForEachInstance==0).sum()
        nbMultiAttrib=N-((nbOfClusterForEachInstance==1).sum()+(nbOfClusterForEachInstance==0).sum())
        print("Analysis: ") #Compute stats on instance apparitions
        print("Number of instances without clusters : ",(nbOfClusterForEachInstance==0).sum())
        print("Number of instances in 1 cluster : ",(nbOfClusterForEachInstance==1).sum())
        print("Number of instances in more than 1 cluster : ",N-((nbOfClusterForEachInstance==1).sum()+(nbOfClusterForEachInstance==0).sum()))
        print("Stats on number of attribution of instances:  min: ",min(nbOfClusterForEachInstance),'; mean : ',round(np.mean(nbOfClusterForEachInstance),3),"; median ",np.median(nbOfClusterForEachInstance),'; max : ',max(nbOfClusterForEachInstance))

    #verif WCSS
    if(idObj==6):
        print("Verif WCSS: ")
        postWCSS=int(sum([clustWCSSs[c] for c in range(len(F)) if F[c]==1]))
        print(postWCSS,postWCSS/sum(F)/100)

    #Compute stats on cluster length
    allClust,allClustLen=findSelectedClustersSizes(clusterIds,F)
    print("Overall number of clusters: ",len(allClustLen))
    print("Selected cluster size stats:  min: ",min(allClustLen),'; mean : ',round(np.mean(allClustLen),3),"; median ",np.median(allClustLen),'; max : ',max(allClustLen))

    #Compute stats on pattern length
    allPat,allPatLen=findSelectedPatternStats(D,listPat,F,candidateDescr)
    print("Overall number of patterns: ",len(allPatLen))
    print("Selected pattern size stats:  min: ",min(allPatLen),'; mean : ',round(np.mean(allPatLen),3),"; median ",np.median(allPatLen),'; max : ',max(allPatLen))

    #Compute textual descriptions
    descrNames,descrPatIds,descrTagsIds=genDescriptionNames(D,F,candidateDescr,tagNames,listPat)
    if(dataName in ['NC','NewCaledonia','nc','NC1014','NC5690','Thibaut'] or ('NC' in dataName)):
        descrNames=[patternConcisionNC(descr) for descr in descrNames]
    if(verbose):
        print("Cluster descriptions:")
        for d in range(len(descrNames)):
            print(d," : ",[" ".join(p) for p in descrNames[d]])

    #Clustering quality measures
    print(len(allClustLen)," found clusters with following sizes ; mean : ",round(np.mean(allClustLen),2),"; median ",round(np.median(allClustLen),2),'; std : ',round(np.std(allClustLen),2))

    #Description quality measures
    selClustIds=[i for i in range(len(F)) if F[i]==1] #ids of the selected clusters
    PCRs,DCs,IPSs,SINGs,IPCs,patPCRs,patDCs,patIPSs,patSINGs,patIPCs=computeNovelDescrQuality(len(allClust),N,allClust,instPatDF.to_numpy(),clustPatDF.to_numpy(),selClustIds,descrPatIds,allClustLen,patternCoveragePercentage)
    print("PCRs : ",PCRs) #print(patPCRs)
    print("DCs : ",DCs)
    #print("IPSs : ",IPSs)
    #print("SINGs : ",SINGs)
    print("IPCs : ",IPCs)

    #test covered eval
    if(True):#obj==7 or obj==9):
        #Note: in last ver for app, forgot to remove this measure
        uncovNorSinglePoints=findUncoveredOrNotSingleAssignedPoints(descrPatIds,allClust,nbOfClusterForEachInstance,instPatDF.to_numpy())
        print('Number of points not assigned to a single cluster or uncovered by their cluster pat :',len(uncovNorSinglePoints))

        uncovPoints=findUncoveredPoints(descrPatIds,allClust,instPatDF.to_numpy())
        #print('Number of assigned uncovered points (by pat of same clust):',len(uncovPoints))
        print('total number of uncovered points (by pat of their clusters):',len(uncovPoints)+nbNotAttrib)
        uncovPointsL=findUncoveredPointsLight(descrPatIds,allClust,instPatDF.to_numpy())
        #print('Number of assigned uncovered points:',len(uncovPointsL))
        print('total number of uncovered points (by any pat):',len(uncovPointsL)+nbNotAttrib)
        #print(z)

    #optional results
    optionalRes=[]

    if(dataName in ["AWA2","AwA2","awa2","AWA","AwA","aWa","awa","animals"]): #the composition of the clusters in terms of Ground truth labels for AWA.
        optionalRes=(None,None,showClusterComposition(allClust,groundTruth,classNames))

    elif(dataName in ["iris","Iris","IRIS","UCIiris","irisCompet","irisTest"]): #the composition of the clusters in terms of Ground truth labels.
        optionalRes=showClusterComposition(allClust,groundTruth,classNames)
        print("Iris classes: ",optionalRes)
        ARI=None
        WCSSval=None
        if(idObj==6):
            if(nbMaxMoreThan1==0 and nbMaxLessThan1==0):
                labels=genLabels(allClust,N,nbOfClusterForEachInstance)
                ARI=round(metrics.adjusted_rand_score(labels,groundTruth),2)
                WCSSval=round(postWCSS/sum(F)/100,2)
                #optionalRes=(ARI,WCSSval,optionalRes)
                print("ARI: ",ARI)
            else:
                WCSSval=round(postWCSS/sum(F)/100,2)
                #optionalRes=(WCSSval,optionalRes)
        elif(nbMaxMoreThan1==0 and nbMaxLessThan1==0):
            labels=genLabels(allClust,N,nbOfClusterForEachInstance)
            ARI=round(metrics.adjusted_rand_score(labels,groundTruth),2)
            #optionalRes=(ARI,optionalRes)
            print("ARI: ",ARI)
        optionalRes=(ARI,WCSSval,optionalRes)

    elif(("moon" in dataName) or (dataName in ["circles",'Circles','circle','Circle'])):
        labels=genLabels(allClust,N,nbOfClusterForEachInstance)
        ARI=round(metrics.adjusted_rand_score(labels,groundTruth),2)
        print("ARI: ",ARI)
        optionalRes+=[ARI]
        #showArtifDataSplit(featureSpace,labels,colorsPart,shapesPart,sizesPart)

    elif("flag" in dataName or "Flag" in dataName):
        countryNames=showFlagComp(dataPath,allClust)
        optionalRes=countryNames
        print(countryNames)

    elif(dataName in ["Automobile","automobile","auto","Auto","AUTOMOBILE","car","cars"]):
        #featureSpace,tagSpace,tagNames=prepareAutomobileData(readAutomobileData(dataPath,False)) #AUTOMOBILE data, deprecated
        optionalRes=[[round(np.mean([prices[i] for i in clust]),2) for clust in allClust],[round(np.std([prices[i] for i in clust]),2) for clust in allClust]]
        print('Mean automobile prices:',optionalRes)
        print('Cluster lengths:',[len(clust) for clust in allClust])

    elif(dataName in ["Students","Student","student","students"]):
        #TODO compute stats of results of students in each clusters
        ClusterStudentGradesMean=[[np.mean(featureSpace[s]) for s in clust] for clust in allClust]
        for k in range(len(ClusterStudentGradesMean)):
            cg=ClusterStudentGradesMean[k]
            #print('DEBUG cg:',cg)
            print('Grades of C'+str(k)+'\'s '+str(len(allClust[k]))+' students:','mean:',round(np.mean(cg),2),'std:',round(np.std(cg),2),'median:',round(np.median(cg),2),'min:',round(min(cg),2),'max:',round(max(cg),2))
        optionalRes=[np.mean(ClusterStudentGradesMean[k]) for k in range(len(ClusterStudentGradesMean))]
        #print('Mean average student score:',optionalRes)

    elif('artif' in dataName):
        #Create attribution matrix (only works if no unassigned !)
        attrMat=[[int(i in C) for C in allClust] for i in range(len(featureSpace))]
        if(dataName in ['artif0']):
            evalclustering_ari2 = round(overlapping_ari_with_alignment(attrMat,gtAttrMat),2)
            evalclustering_jaccard = round(accurate_jaccard(attrMat,gtAttrMat),2)
        else: #artif1 artif2 artif3
            #Overlapping GT
            evalclustering_ari2 = round(overlapping_ari_with_alignment(attrMat,gtAttrMat),2)
            evalclustering_jaccard = round(accurate_jaccard(attrMat,gtAttrMat),2)
        #print(allClust)
        #print('Clustering ARI evaluation value:',evalclustering_ari)
        print('Clustering ARI evaluation value:',evalclustering_ari2)
        print('Clustering Jaccard evaluation value:',evalclustering_jaccard)
        optionalRes=[(evalclustering_ari2,evalclustering_jaccard,nbNotAttrib)]

    elif(dataName in ['SecondaryMushroom','secondaryMushroom','smushroom','sMushroom']):
        ClusterEdibleNumber=[np.sum([gt[i] for i in allClust[k]]) for k in range(len(allClust))]
        for k in range(len(allClust)):
            print('Number of edible mushroom in clust',k,':',ClusterEdibleNumber[k])
        optionalRes=ClusterEdibleNumber

    #Write parameters and results
    resultTextPath=resultFolderPath+"/output-"+resultFileName+".txt"
    params=[baseAlgorithms,repeatBPgen,maxClustSize,patternCoveragePercentage,patType,V,clustPatDF.shape,Kmin,Kmax,minAppar,maxAppar,maxCovOutPer,nbMaxMoreThan1,nbMaxLessThan1,nbMaxDiffThan1,idObj]
    paramsNames=['baseAlgorithms','repeatBPgen','maxClustSize','Coverage','patType','V','clustPatDF.shape','Kmin','Kmax','minAppar','maxAppar','Discriminativeness','nbMaxMoreThan1','nbMaxLessThan1','nbMaxDiffThan1','idObj']
    res=[times,allClust,allClustLen,descrNames,descrPatIds,descrTagsIds,nbNotAttrib,nbMultiAttrib,nbOfClusterForEachInstance,"----",PCRs,DCs,IPSs,SINGs,IPCs,patPCRs,patDCs,patIPSs,patSINGs,patIPCs,optionalRes]
    resNames=['times','allClust','allClustLen','descrNames','descrPatIds','descrTagsIds','nbNotAttrib','nbMultiAttrib','nbOfClusterForEachInstance',"----",'PCRs','DCs','IPSs','SINGs','IPCs','patPCRs','patDCs','patIPSs','patSINGs','patIPCs','Optional results']
    writeResults(resultTextPath,params,res,paramsNames,resNames)

    if(tsne):
        labels=genLabels(allClust,N,nbOfClusterForEachInstance)
        if(dataName in ["Automobile","automobile","auto","Auto","AUTOMOBILE","car","cars"]):
            vizualizePartition1D(len(allClust),featureSpace,labels,resultFolderPath)
        elif('artif' in dataName):
            vizualizePartition(len(allClust),featureSpace,labels,resultFolderPath)
        elif(dataName in ['NC','NewCaledonia','nc','NC1014','NC5690','Thibaut'] or ('NC' in dataName)):
            NCProjPath='NC1014_proj.csv'
            projSpace=readProjSpace(NCProjPath,delimiter=',')
            if(ncEx!=None):
                if(ncEx<8 or ncEx>=0):
                    projSpace=[projSpace[i] for i in range(len(projSpace)) if gt[i]==ncEx]
            vizualizePartition(len(allClust),projSpace,labels,resultFolderPath)
        else:
            lauchTSNE(featureSpace,labels,len(allClustLen)+1,resultFolderPath,"TSNE_"+dataName,10, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)
        #TypeError: __init__() got an unexpected keyword argument 'square_distances

    if(verbose):
        print()
        print(descrNames)

    #TODO return filteringRes avoiding pipeline repetition
    #TODO return lists with the unattributed and the overlapping molecules ?
    return times,BasePartitions,BP,filteringRes,allClust,allClustLen,descrNames,descrPatIds,descrTagsIds,nbOfClusterForEachInstance,PCRs,DCs,IPSs,SINGs,IPCs,optionalRes,treeRes

#---------------------------------
#------ TEST AREA ------

autoPath="imports-85.data"
flagPath="flag.data"
adultPath="/adult.data"
studentpathMat="student-mat.csv"
studentpathPor="student-por.csv"
NCPath='NC1014_clustering.csv'

def normal_mainTestECS():
    #timeAnalysis(dataName,5,resultFolderPath)
    dataName='NC' #'Automobile' #'iris'
    resultFileName=dataName+'rob'#+"ncRandom"#"_NC0rephrasedkm_inc_tag"
    dataPath=NCPath #flagPath #autoPath #invovldPath #not import for iris.
    distMatPath=""
    resultFolderPath="foldername"
    cov=30
    discr=10
    patType="tag" #'tag','pat'
    bpKmin=4
    bpKmax=15
    finalKmin=9#2
    finalKmax=9#8
    nbOv=20
    nbUna=20
    baseAlgorithms=['Kmeans'] #"Kmeans","Spectral","SpectralNN",'DBSCAN'
    idObj=9#3
    treeCompare=not True
    allowInclusion=True
    patPreference=0 #0 : no preference. 1 : favorize more general patterns. 2 : favorize more specific patterns.
    ncEx=0#None
    nbRepeatBP=2
    comparisonRobust(dataName,dataPath,resultFolderPath,resultFileName,cov,discr,patType,bpKmin,bpKmax,finalKmin,finalKmax,
               baseAlgorithms,idObj,nbOv,nbUna,nbRepeatBP,True,ncEx,treeCompare,
               verbose=True,allowInclusion=allowInclusion,patPreference=patPreference,locs=[0],scales=[0,0.5])
    #launchICES(dataName,dataPath,resultFolderPath,resultFileName,cov,discr,patType,bpKmin,bpKmax,finalKmin,finalKmax,
    #           baseAlgorithms,idObj,nbMaxMoreThan1=nbOv,nbMaxLessThan1=nbUna,repeatBPgen=nbRepeatBP,tsne=True,ncEx=ncEx,treeCompare=treeCompare,
    #           verbose=True,allowInclusion=allowInclusion,patPreference=patPreference)

def mainTestECS():
    dataName='artif4' #'Automobile' #'iris'
    resultFileName=dataName #+"ncRandom"#"_NC0rephrasedkm_inc_tag"
    dataPath=NCPath
    distMatPath=""
    resultFolderPath="results"
    cov=50
    discr=30
    patType="pat" #'tag','pat'
    bpKmin=2
    bpKmax=5
    finalKmin=3#2
    finalKmax=3#8
    nbOv=100 #0
    nbUna=5
    baseAlgorithms=['Kmeans','Spectral'] #"Kmeans","Spectral","SpectralNN",'DBSCAN'
    idObj=9#3
    treeCompare=not True
    allowInclusion=True
    patPreference=0 #0 : no preference. 1 : favorize more general patterns. 2 : favorize more specific patterns.
    ncEx=0#None
    nbRepeatBP=4
    launchICES(dataName,dataPath,resultFolderPath,resultFileName,cov,discr,patType,bpKmin,bpKmax,finalKmin,finalKmax,
               baseAlgorithms,idObj,nbMaxMoreThan1=nbOv,nbMaxLessThan1=nbUna,repeatBPgen=nbRepeatBP,tsne=True,ncEx=ncEx,treeCompare=treeCompare,
               verbose=True,allowInclusion=allowInclusion,patPreference=patPreference)

mainTestECS()
