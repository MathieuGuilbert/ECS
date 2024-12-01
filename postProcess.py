import numpy as np
from sklearn.metrics import silhouette_score
from particularDataTreatment import getCountriesNames

def verifIndivCoverage(clusterIds,instDF,D,F,candidateDescr,listPat):
    '''Count number of individual instances covered by their cluster.s descriptors.'''
    coveredInstances=[]
    for c in range(len(F)):
        if F[c]==1:         #for each selected cluster
            for inst in clusterIds[c]: #for each instance
                if(inst not in coveredInstances):
                    if(len(D[c])>1):
                        for d in range(len(D[c])):
                            p=D[c][d]
                            tagName=listPat[p]
                            if(candidateDescr[c][d]==1 and instDF[tagName][inst]==1):
                                coveredInstances.append(inst)
                                break
                    else:
                        p=D[c][0]
                        tagName=listPat[p]
                        if(candidateDescr[c]==1 and instDF[tagName][inst]==1):
                            coveredInstances.append(inst)
                            break
    return coveredInstances

def findSelectedPatternStats(D,listPat,F,candidateDescr):
    '''Create list of all the selected patterns and another list with all selected patterns' lengths.'''
    selectedPat=[]
    selPatLen=[]
    for c in range(len(F)):
        if F[c]==1:
            if(len(D[c])>1):
                for d in range(len(D[c])):
                    if(candidateDescr[c][d]==1):
                        p=D[c][d]
                        tagName=listPat[p]
                        tagList=tagName[1:-1].split(",")
                        selectedPat.append(tagList)
                        selPatLen.append(len(tagList))
            else:
                p=D[c][0]
                tagName=listPat[p]
                tagList=tagName[1:-1].split(",")
                selectedPat.append(tagList)
                selPatLen.append(len(tagList))
    return selectedPat,selPatLen

def findSelectedClustersSizes(clusterIds:list,F:list):
    '''Returns list of all selected cluster lengths.'''
    clusters=[]
    clustLengths=[]
    for c in range(len(F)):
        if F[c]==1:
            clusters.append(clusterIds[c])
            clustLengths.append(len(clusterIds[c]))
    return clusters,clustLengths

def writeResults(path:str,params:list,values:list,paramsNames:list,resNames:list):
    '''Write all results in a txt file.

    params: list of parameter values
    paramsNames: list of parameters names
    values: list of output values
    resNames: list of output value names'''
    h = open(path, "w")
    for p in range(len(params)):
        par=params[p]
        h.write(paramsNames[p]+" : "+str(par)+"\n")
    h.write("\n")
    h.write("--- RESULTS ---\n")
    h.write("\n")
    for v in range(len(values)):
        val=values[v]
        h.write(resNames[v]+" : "+str(val)+"\n")
    h.close()
    print("Results written succefully in ",path)
    return

def genLabels(selectedCluster:list,N:int,nbOfClusterForEachInstance:list):
    '''Create a partition in the fitting format for tsne.'''
    part=[]
    for i in range(N):
        if(nbOfClusterForEachInstance[i]==0): #cluster constituted of outliers
            part.append(len(selectedCluster))
        else:
            for c in range(len(selectedCluster)):
                if i in selectedCluster[c]:
                    part.append(c)
                    break
    return part

def genLabels2(selectedCluster:list,N:int):
    '''Create a partition in the fitting format for tsne.'''
    part=[]
    for i in range(N):
        found=False
        for c in range(len(selectedCluster)):
            #print(i,c,(i in selectedCluster[c]))
            if(i in selectedCluster[c] and(not found)):
                part.append(c)
                found=True
                break
        if(not found):
            part.append(len(selectedCluster))
    return part

def getNonUniquelyClusteredInstances(attrib:list):
    '''Get non-uniquely clusterd instances.

    Parameters
    --------
    attrib: list of the number of clusters to which each instance is attributed

    Returns
    ---------
    unattr: list of the ids of the unattributed instances
    overlapped: list of the ids of the instances attributed to multiple clusters
    '''
    unattr=[]
    overlapped=[]
    for i in range(len(attrib)):
        if(attrib[i]<1):
            unattr.append(i)
        elif(attrib[i]>1):
            overlapped.append(i)
    return unattr,overlapped

def getSelectedPointsSpace(attrib:list,X):
    '''Get Selected Points (appearing more than once) space

    Parameters
    --------
    attrib: list of the number of clusters to which each instance is attributed
    '''
    X2=[]
    for i in range(len(X)):
        #print('i:',i)
        if(int(attrib[i])>0):
            #line=[]
            line=X[i]
            #for j in range(len(X[0])):
                #print('j:',j)
            #    if(int(attrib[j])>0):
            #        line.append(X[i][j])
                    #print('X[i][j]:',X[i][j])
            X2.append(line)
    return X2

#------

#TODO: instead of pattern id, return list of list with ids of descriptors in each patterns ?
def genDescriptionNames(D,F,candidateDescr,tagNames,listPat):
    '''Compute textual descriptions.

    PARAMETERS
    --------
    D: the set of all candidate descriptions.
    F: the final partition/clustering.
    candidateDescr : returned by the CP model, 0 or 1 if the corresponding pattern is selected.
    tagNames: the name of the individual tags.
    listPat : list were each sublist contains the ids of the descriptors in corresponding pattern.'''
    patNamesPerClust=[]
    patIdsPerClust=[]
    tagsIdsPerClust=[]
    for c in range(len(F)):
        if F[c]==1:
            clustPatNames=[]
            clustPatIds=[]
            clustTagsIds=[]
            if(len(D[c])>1):
                for d in range(len(D[c])):
                    if(candidateDescr[c][d]==1):
                        p=D[c][d]
                        patTags=listPat[p]
                        tagList=patTags[1:-1].split(",")
                        #print(tagNames,len(tagNames),tagList)
                        clustPatNames.append([tagNames[int(t)] for t in tagList])
                        clustPatIds.append(p)
                        clustTagsIds.append(patTags)
            else:
                p=D[c][0]
                patTags=listPat[p]
                tagList=patTags[1:-1].split(",")
                clustPatNames.append([tagNames[int(t)] for t in tagList])
                clustPatIds.append(p)
                clustTagsIds.append(patTags)
            patNamesPerClust.append(clustPatNames)
            patIdsPerClust.append(clustPatIds)
            tagsIdsPerClust.append(clustTagsIds)
    return patNamesPerClust,patIdsPerClust,tagsIdsPerClust

#------------------------------------------------
# GET MORE INFORMATION ON CLUSTER OF SPECIFIC DATASETS
#------------------------------------------------

def showClusterComposition(clusters:list,labels:list,classNames:list):
    '''Find the composition of the clusters in terms of Ground truth labels.
    #Note: for now, only applied on AWA2'''
    allCP=[]
    res=[]
    for clust in clusters:
        classPresence=[0 for cl in classNames]
        for inst in clust:
            classPresence[labels[inst]-1]+=1
        allCP.append(classPresence)
    for i in range(len(allCP)):
        stri=""
        cpi=allCP[i]
        for v in range(len(cpi)):
            if cpi[v]!=0:
                if(stri!=""):
                    stri+=" ; "
                stri=stri+str(cpi[v])+" "+classNames[v]
        res.append(stri)
    return res

def showFlagComp(path,clusters):
    '''Get names of countries in each clusters.'''
    names=getCountriesNames(path)
    clustNames=[]
    for c in clusters:
        cNames=[]
        for i in c:
            cNames.append(names[i])
        clustNames.append(cNames)
    return clustNames

#------------------------------------------------
# CLUSTERING QUALITY MEASURES
#------------------------------------------------

#Note: silhouette index do not handle unattributed index: should thus remove them from X and P
def evalSilhouette(X,P):
    #X: array of pairwise distances between samples, or a feature array.
    #metric: str or callable, default=’euclidean’. If X is the distance array itself, use metric="precomputed".
    res=silhouette_score(X, P)
    return res

#------------------------------------------------
# NOVEL DESCRIPTION QUALITY MEASURES
#------------------------------------------------

def computeNovelDescrQuality(K:int,N:int,clusters:list,instPatMat,clustPatMat,selClustIds,descr,clustLengths,patternCoveragePercentage):
    '''Compute all our novel descriptions quality criterion.

    PARAMETERS
    -------
    K: number of clusters in the partition
    N: number of instances
    clusters: list of list where sublist i contains ids of instances in Ci
    instPatMat: matrix with activity of each instance on each pattern
    clustPatMat: matrix with activity of each cluster on each pattern
    selClustIds: Clusters, in the form of lists of instance ids
    descr: Descriptions of the clusters
    clustLengths: length of all the clusters
    patternCoveragePercentage: Coverage parameter value

    RETURN
    --------
    XXXs: lists with the values of the measures for all clusters
    patXXXs: detailed measures values for all the patterns of the clusters.
    '''
    patPCRs=[]
    patDCs=[]
    patIPSs=[]
    patSINGs=[]
    patIPCs=[]
    PCRs=[]
    DCs=[]
    IPSs=[]
    SINGs=[]
    IPCs=[]
    for i in range(K):
        #print('pattern mat:',type(instPatMat))
        #print('descr:',descr[i],type(descr[i]),type(descr[i][0]))
        #print(z)

        cPCR,patPCR=clustPCR(descr[i],selClustIds[i],clustPatMat,clustLengths[i]) #D,Cid,clustPatMat,clustSize
        PCRs.append(cPCR)
        patPCRs.append(patPCR)

        DCs.append( DC(descr[i],clusters[i],instPatMat)) #D,C,instPatMat

        cIPS,patIPS=clustIPS(descr[i],clusters[i],N,instPatMat)
        IPSs.append(cIPS)
        patIPSs.append(patIPS)
        cSING,patSING=clustSING(descr[i],selClustIds[i],K,selClustIds,clustPatMat,clustLengths,patternCoveragePercentage)
        SINGs.append(cSING)
        patSINGs.append(patSING)
        cIPC,patIPC=clustIPC(descr[i],selClustIds[i],K,selClustIds,clustPatMat,clustLengths)
        IPCs.append(cIPC)
        patIPCs.append(patIPC)
    return PCRs,DCs,IPSs,SINGs,IPCs,patPCRs,patDCs,patIPSs,patSINGs,patIPCs

#--
def patPCR(p:int,Cid:int,clustPatMat:list,clustLen:int):
    '''Pattern Coverage rate (PCR) of a particular cluster. Domain between 0 and 1, 1 being the best results where p covers all of the instances of the cluster.

    PARAMETERS
    --------
    p: pattern id
    Cid: cluster id
    clustPatMat: matrix of dataframe where M C p is the number of instances in C covered by p. this matrix is computed in the first part of the approach.
    clustSize: size of the cluster
    '''
    return clustPatMat[p][Cid]/clustLen

def clustPCR(D,Cid,clustPatMat,clustLen):
    '''Get PCR of a particular cluster.'''
    patPCRs=[]
    for p in D:
        #print(len(D),p,Cid,len(clustPatMat),len(clustPatMat[0]),clustLen)
        patPCRs.append(patPCR(p,Cid,clustPatMat,clustLen))
    return (round(np.mean(patPCRs),2),round(np.std(patPCRs),2)),patPCRs

#--
def DC(D,C,instPatMat):
    '''cluster Description Coverage (DC) measuring if instances are covered by at least one of their cluster’s descriptive patterns'''
    notCovered=[]
    for o in C:
        cov=False
        for p in D:
            if instPatMat[o][p]==1:
                cov=True
                break
        if cov==False:
            notCovered.append(o)
    return round((len(C)-len(notCovered))/len(C),2)

#--
def IPS(p,Clust,N,instPatMat):
    '''IPS: Inverse Pattern Specificity (dataset-wise discrimination).'''
    s=0
    for o in range(N): #count number of instances outside the cluster that are covered by p
        if o not in Clust:
            if instPatMat[o][p]==1:
                s+=1
    return 1-(s/(N-len(Clust)))

def clustIPS(D,Clust,N,instPatMat):
    '''Compute IPS for a certain cluster with its description D.'''
    patIPSs=[]
    for p in D:
        patIPSs.append(IPS(p,Clust,N,instPatMat))
    return (round(np.mean(patIPSs),2),round(np.std(patIPSs),2)),patIPSs

#--
def patSING(p,idclusti,K,selClustIds,clustPatMat,clustLen,per):
    '''pattern SING : SINGularity

    p: pattern id
    i: cluster id
    K: number of clusters
    Descriptions: list of all selected cluster descriptions
    Part: list of all selected clusters Ids
    clustPatMat: matrix of dataframe where M C p is the number of instances in C covered by p. this matrix is computed in the first part of the approach.
    per: threshold on number of instances
    '''
    occ=0
    for j in range(K):
        th=int((per*clustLen[j])/100)
        idclustj=selClustIds[j]
        if(idclustj!=idclusti):
            if(clustPatMat[p][idclustj]>=th):
                occ+=1
    return 1-occ/(K-1) #1-val to make 1 the desirable outcome

def clustSING(D,idclusti,K,selClustIds,clustPatMat,clustLen,per):
    '''compute a certain cluster SING with its description D.'''
    patSINGs=[]
    for p in D:
        patSINGs.append(patSING(p,idclusti,K,selClustIds,clustPatMat,clustLen,per))
    return (round(np.mean(patSINGs),2),round(np.std(patSINGs),2)),patSINGs

#--
def patIPC(p,idclusti,K,selClustIds,clustPatMat,clustLen):
    '''IPC: Inverse Pattern Contrastivity (instance wise).'''
    s=0 #sum
    for j in range(K):
        idclustj=selClustIds[j]
        if(idclustj!=idclusti):
            jLen=clustLen[j]
            s+=1-(clustPatMat[p][idclustj]/jLen)
    return s/(K-1)

def clustIPC(D,idclusti,K,selClustIds,clustPatMat,clustLengths):
    '''Compute IPC for a certain cluster with its description D.'''
    patIPCs=[]
    for p in D:
        patIPCs.append(patIPC(p,idclusti,K,selClustIds,clustPatMat,clustLengths))
    return (round(np.mean(patIPCs),2),round(np.std(patIPCs),2)),patIPCs

#------
def findUncoveredPoints(Ds,Cs,instPatMat):
    '''Find the points not covered by any pattern of clusters they belong to.

    PARAMETERS
    ------
    Cs: list of clusters
    Ds: list of cluster explanations
    instPatMat:

    RETURN
    ------
    notCovered: lsit of the ids of the points not being covered by at least one of their cluster(s)'s pattern.
    '''
    notCovered=[]
    Covered=[]
    for k in range(len(Cs)):
        C=Cs[k]
        D=Ds[k]
        for o in C:
            cov=(o in Covered)
            if cov==False:
                for p in D:
                    if instPatMat[o][p]==1:
                        cov=True
                        if(o in notCovered):
                            notCovered.remove(o)
                        Covered.append(o)
                        break
                if cov==False:
                    notCovered.append(o)

    return notCovered

def findUncoveredOrNotSingleAssignedPoints(Ds,Cs,nbOfClusterForEachInstance,instPatMat):
    '''Find the points not covered by any pattern of clusters they belong to.

    PARAMETERS
    ------
    Cs: list of clusters
    Ds: list of cluster explanations
    nbOfClusterForEachInstance: list
    instPatMat: np array of np array

    RETURN
    ------
    notCovered: lsit of the ids of the points not being covered by at least one of their cluster(s)'s pattern.
    '''
    notCovered=[]
    Covered=[]
    for k in range(len(Cs)):
        C=Cs[k]
        D=Ds[k]
        for o in C:
            cov=(o in Covered)
            if cov==False:
                for p in D:
                    if instPatMat[o][p]==1:
                        cov=True
                        if(o in notCovered):
                            notCovered.remove(o)
                        Covered.append(o)
                        break
                if cov==False:
                    notCovered.append(o)

    notCovNorSingleAssigned=notCovered
    for o in range(len(nbOfClusterForEachInstance)):
        if(nbOfClusterForEachInstance[o]!=1) and (o not in notCovNorSingleAssigned):
            notCovNorSingleAssigned.append(o)

    return notCovNorSingleAssigned

def findUncoveredPointsLight(Ds,Cs,instPatMat):
    '''Find the points not covered by any pattern.

    PARAMETERS
    ------
    Cs: list of clusters
    Ds: list of cluster explanations
    instPatMat:

    RETURN
    ------
    notCovered: lsit of the ids of the points not being covered by at least one of their cluster(s)'s pattern.
    '''
    notCovered=[]
    Covered=[]
    for k in range(len(Cs)):
        C=Cs[k]
        for o in C:
            cov=(o in Covered)
            for D in Ds:
                if cov==False:
                    for p in D:
                        if instPatMat[o][p]==1:
                            cov=True
                            if(o in notCovered):
                                notCovered.remove(o)
                            break
            if cov==False:
                notCovered.append(o)

    return notCovered

def patternConcisionNC(patList:list):
    '''Enforce pattern concision in NewCaledonia results by removing rendondant information. patList is a list of string list.'''
    resList=[]
    for p in patList:
        #print(p)
        pat=list(p)
        if(('HasLoop' in pat)):
            if(('HasWhileLoop' in pat) and ('HasForLoop' in pat) or ('HasIfInLoop' in pat) or ('HasLoopInIf' in pat)):
                pat.remove('HasLoop')

        if('HasIf' in pat):
            if(('HasIfElse' in pat) or ('HasIfElif' in pat) or ('HasIfElifElse' in pat) or ('HasIfInLoop' in pat) or ('HasLoopInIf' in pat)):
                pat.remove('HasIf')
            if(('HasIfElse' in pat)):
                if(('HasIfElif' in pat) or ('HasIfElifElse' in pat)):
                    pat.remove('HasIfElse')
                if(('HasIfElif' in pat)):
                    if(('HasIfElifElse' in pat)):
                        pat.remove('HasIfElif')
        resList.append(pat)

    return resList
