import subprocess
from clusterSelection import genClustPatternDFColumn
from postProcess import *
from dataTreatment import createDF,createNegDF
from partitionCreation import launchICES
import numpy as np

from dataTreatment import writeMat

def evaluationPipeline(dataNames,dataPaths,Algorithms,covValues,patTypes,minClustSizes,maxClustSizes,
                       discrValues,overlapValues,unatValues,finalKsMin,finalKsMax,objs,rBP,resPath,
                       maxNbPatPerClustList=[5],patPreferences=[-1],ncEx=None,skipDiscr=True):
    """Function for performing a pipeline of tests with different parameters for the same dataset."""
    BP=[]
    noRes=[]
    Skipped=[]
    allRes=[]
    allRes.append(['Coverage','Discr.','patType','ov','un','finKmin','finKmax','maxNbPat','patPreference','obj','PCR','DC','IPC','Size','Other'])
    nbBP=1
    nbLaunches=nbBP*len(covValues)*len(patTypes)*len(minClustSizes)*len(maxClustSizes)*len(discrValues)*len(overlapValues)*len(unatValues)*len(finalKsMin)*len(finalKsMax)*len(patPreferences)*len(maxNbPatPerClustList)*len(objs)
    print("MAX TOTAL NUMBER OF EXPERIMENTS: ",nbLaunches)
    #for each dataset
    for dts in range(len(dataNames)):
        #with 3 base partition sets
        for i in range(0,nbBP):
            print(i,"th launch for dataset ",dataNames[dts])
            cpt=0
            BP=[]
            #for each Algorithm option
            #for each coverage value
            for cov in covValues:
                #for each pattern type
                isAnyPatRes=False #TODO: si pas de res avec pat, alors on en aura pas avec tags
                for patType in patTypes:
                    if(patType=='tag' and ('pat' in patTypes and isAnyPatRes==False) ):
                        break
                    #for each minimum cluster size
                    for minK in minClustSizes:
                        #for each maximum cluster size
                        for maxK in maxClustSizes:
                            Filtered=[]
                            #for each discr value
                            for discr in discrValues:
                                if(discr>=cov and skipDiscr):
                                    print(discr,'>',cov,' : PASS')
                                    Skipped.append((cov,discr))
                                    #TODO increase cpt ?
                                    break
                                #for each overlap parameters
                                for ov in overlapValues:
                                    #for each unattributed instances parameters
                                    for un in unatValues:
                                        #for each number of final clusters
                                        for finKmin in finalKsMin:
                                            for finKmax in finalKsMax:
                                                for pP in patPreferences:
                                                    needToTestNbpat=True
                                                    #for each maxNbPatPerClust
                                                    for maxNbPat in maxNbPatPerClustList:
                                                        if(not needToTestNbpat): #no need to test for other max number of patterns (WARNING works only if list start by None)
                                                            break
                                                        #for each objective criteria
                                                        for obj in objs:
                                                            #Launch
                                                            path=resPath+"/"+str(dataNames[dts])
                                                            if(pP>=0):
                                                                fileName=str(dataNames[dts])+"_"+str(i)+"_"+str(cpt)+"_"+patType+"_"+str(cov)+"_"+str(discr)+'_patPref'+str(pP)+'_maxNbPat'+str(maxNbPat)+"_obj"+str(obj)
                                                            else:
                                                                fileName=str(dataNames[dts])+"_"+str(i)+"_"+str(cpt)+"_"+patType+"_"+str(cov)+"_"+str(discr)+'_patPrefiltered_'+'_maxNbPat'+str(maxNbPat)+"_obj"+str(obj)
                                                            allowInclusion=(pP>=0)
                                                            if(pP<0):
                                                                patPreference=0
                                                            else:
                                                                patPreference=pP
                                                            res=launchICES(dataNames[dts],dataPaths[dts],path,fileName,cov,discr,
                                                                            patType,minK,maxK,finKmin,finKmax,Algorithms,obj,
                                                                        nbMaxMoreThan1=ov,nbMaxLessThan1=un,repeatBPgen=rBP,maxNbPatPerClust=maxNbPat,ncEx=ncEx,
                                                                        precompBC=BP,precompFiltered=Filtered,allowInclusion=allowInclusion,patPreference=patPreference)
                                                            #IF NO RESULTS, NO NEED TO TEST OTHER OBJECTIVES
                                                            if(res==None):
                                                                noRes.append(cpt)
                                                                allRes.append([cov,discr,patType,ov,un,finKmin,finKmax,maxNbPat,pP,obj,None,None,None,None,None])
                                                                needToTestNbpat=False #no need to test for other max number of patterns (WARNING works only if list start by None)
                                                                break
                                                            else:
                                                                _,_,_,_,_,_,_,_,_,nbOfClusterForEachInstance,PCRs,DCs,IPSs,SINGs,IPCs,optionalRes=res
                                                                #TODO keep relevent info
                                                                if(len(optionalRes)==len(PCRs)):
                                                                    allRes.append([cov,discr,patType,ov,un,finKmin,finKmax,maxNbPat,pP,obj,round(np.mean([o[0] for o in PCRs]),2),round(np.mean(DCs),2),round(np.mean([o[0] for o in IPCs]),2),len(PCRs),optionalRes[0]])
                                                                else:
                                                                    allRes.append([cov,discr,patType,ov,un,finKmin,finKmax,maxNbPat,pP,obj,round(np.mean([o[0] for o in PCRs]),2),round(np.mean(DCs),2),round(np.mean([o[0] for o in IPCs]),2),len(PCRs),optionalRes])
                                                                if(patType=='pat' and isAnyPatRes==False):
                                                                    isAnyPatRes=True
                                                                if(BP==[]):
                                                                    print("EMPTY BP")
                                                                    BasePartitions=res[1] #save the base partitions
                                                                    BasePartPath=path+"/BPmat"+str(i)+".txt"
                                                                    writeMat(BasePartPath,BasePartitions) #write the base partitions (matrix)

                                                                    BP=res[2] #ensures that all experiments are done with the same base partitions/clusters
                                                                    BCpath=path+"/BCmat"+str(i)+".txt"
                                                                    writeMat(BCpath,BP) #write the base partitions (matrix)
                                                                if(Filtered==[]):
                                                                    print("EMPTY FILTERED")
                                                                    Filtered=res[3]
                                                                #allResults.append(res)
                                                            cpt=cpt+1
    print(len(noRes)," configurations with no results.")
    print(len(Skipped)," configurations skipped.")
    print("--- END ---")
    return allRes

def testEvalPipeline(data,optEx=None,optKmeansBL=False,optAlgorithms=[]):
    '''Launch the evaluation pipeline for a particular dataset.'''
    skipDiscr=True
    if(data=="iris"):
        dataNames=["iris"]
        dataPaths=["irisPath"]
        Algorithms=["Kmeans",'Spectral']
        covValues=[70]
        patTypes=["pat"]
        minClustSizes=[2]
        maxClustSizes=[6]
        discrValues=[30]
        overlapValues=[0]#,15
        unatValues=[0]#,15
        finalKsMin=[3]
        finalKsMax=[3]
        objs=[3,5]
        rBP=3
    elif(data in ["Automobile","automobile","auto","Auto","AUTOMOBILE","car","cars"]):
        dataNames=["Automobile"]
        dataPaths=["/datasets/Automobile/imports-85.data"]
        Algorithms=["Kmeans",'Spectral']
        covValues=[50,70]
        patTypes=['pat','tag']
        minClustSizes=[2]
        maxClustSizes=[8]
        discrValues=[30,50]
        overlapValues=[15]#,15
        unatValues=[15]#,15
        finalKsMin=[3]
        finalKsMax=[3]
        objs=[9]
        rBP=1
        if(optKmeansBL):
            Algorithms=["Kmeans"]
            covValues=[70]
            patTypes=['pat','tag']
            minClustSizes=[3]
            maxClustSizes=[4]
            discrValues=[100]
            overlapValues=[0]#,15
            unatValues=[0]#,15
            finalKsMin=[3]
            finalKsMax=[3]
            objs=[9]
            rBP=1
            skipDiscr=False
    elif(data in ["Students","Student","student","students"]):
        dataNames=["Students"]
        dataPaths=["student-mat.csv"]
        Algorithms=["Kmeans"]
        covValues=[70]
        patTypes=['pat',"tag"]
        minClustSizes=[2]
        maxClustSizes=[10]
        discrValues=[10,30,40,50]
        overlapValues=[100]#,15
        unatValues=[100]#,15
        finalKsMin=[3]
        finalKsMax=[8]
        objs=[9]
        rBP=1
        if(optKmeansBL):
            Algorithms=["Kmeans"]
            covValues=[70]
            patTypes=['pat','tag']
            minClustSizes=[3]
            maxClustSizes=[4]
            discrValues=[100]
            overlapValues=[0]#,15
            unatValues=[0]#,15
            finalKsMin=[3]
            finalKsMax=[3]
            objs=[9]
            rBP=1
            skipDiscr=False
    elif(data in ["StudentsPor","StudentPor","StudentsPer"]):
        dataNames=["Students"]
        dataPaths=["student-por.csv"]
        Algorithms=["Kmeans"]
        covValues=[30,50,70]
        patTypes=['pat',"tag"]
        minClustSizes=[2]
        maxClustSizes=[10]
        discrValues=[10,30,40,50]
        overlapValues=[40]#,15
        unatValues=[40]#,15
        finalKsMin=[2]
        finalKsMax=[8]
        objs=[9]
        rBP=1
    elif(data=="AWA2"):
        dataNames=["AWA2"]
        dataPaths=["AWA2Path"]
        Algorithms=["Kmeans"]
        covValues=[70,90]
        patTypes=["pat","tag"]
        minClustSizes=[2]
        maxClustSizes=[11]
        discrValues=[10,30]
        overlapValues=[0]
        unatValues=[30]
        finalKsMin=[2]
        finalKsMax=[2,5]
        objs=[4,5]
        rBP=2
    elif(data=="flagsC"):
        dataNames=["flagsC"]
        dataPaths=["flag.data"]
        Algorithms=["Kmeans"]
        covValues=[30,50,70,90]
        patTypes=["pat","tag"]
        minClustSizes=[2]
        maxClustSizes=[16]
        discrValues=[10,30,50]
        overlapValues=[15]
        unatValues=[15]
        finalKsMin=[2]
        finalKsMax=[6]
        objs=[5,6]
        rBP=2
    elif(data=="Halfmoon" or data=="HalfmoonNeg"):
        dataNames=[data]
        dataPaths=["HalfmoonPath"]
        if optAlgorithms!=[]:
            Algorithms=optAlgorithms
        else:
            Algorithms=["SpectralNN"]
        covValues=[30,40,50,70]
        patTypes=["pat","tag"]
        minClustSizes=[2]
        maxClustSizes=[10]
        discrValues=[10,30,50]
        overlapValues=[0]
        unatValues=[0]
        finalKsMin=[2]
        finalKsMax=[2]
        objs=[3]
        rBP=2
    elif(data=="artif0"):
        dataNames=[data]
        dataPaths=[""]
        Algorithms=['Kmeans']
        covValues=[30,50,60,70]
        patTypes=["pat","tag"]
        minClustSizes=[2]
        maxClustSizes=[6]
        discrValues=[10,30,50]
        overlapValues=[0]
        unatValues=[0,30,50]
        finalKsMin=[3]
        finalKsMax=[3]
        objs=[9]
        rBP=5
    elif(data=="artif1"):
        dataNames=[data]
        dataPaths=[""]
        Algorithms=['Kmeans']
        covValues=[30,50,60,70]
        patTypes=["pat","tag"]
        minClustSizes=[2]
        maxClustSizes=[6]
        discrValues=[10,30,50]
        overlapValues=[50]
        unatValues=[0,50]
        finalKsMin=[3]
        finalKsMax=[3]
        objs=[9]
        rBP=5
    elif(data=="artif2"):
        dataNames=[data]
        dataPaths=[""]
        Algorithms=['Kmeans']
        covValues=[30,50,60,70]
        patTypes=["pat","tag"]
        minClustSizes=[2]
        maxClustSizes=[6]
        discrValues=[10,30,50]
        overlapValues=[50]
        unatValues=[0,50]
        finalKsMin=[3]
        finalKsMax=[3]
        objs=[9]
        rBP=5
    elif("NC" in data):
        NCPath='NC1014_clustering.csv'
        dataNames=[data]
        dataPaths=[NCPath]
        Algorithms=["Kmeans"]
        covValues=[50,70,90]
        patTypes=['pat',"tag"]
        minClustSizes=[6]
        maxClustSizes=[12]
        discrValues=[10,30,50]
        overlapValues=[200]#,15
        unatValues=[200]#,15
        finalKsMin=[8]
        finalKsMax=[8]
        objs=[9]
        rBP=1
        if(optKmeansBL):
            Algorithms=["Kmeans"]
            covValues=[70]
            patTypes=['pat','tag']
            minClustSizes=[8]
            maxClustSizes=[9]
            discrValues=[100]
            overlapValues=[0]#,15
            unatValues=[0]#,15
            finalKsMin=[8]
            finalKsMax=[8]
            objs=[9]
            rBP=1
            skipDiscr=False
    maxNbPatPerClustList=[5] #None,5
    patPreferences=[0] #-1,0,1,2,3
    resPath="../results/Pipeline"
    print('Start evaluationPipeline')
    allRes=evaluationPipeline(dataNames,dataPaths,Algorithms,covValues,patTypes,minClustSizes,maxClustSizes,discrValues,
                       overlapValues,unatValues,finalKsMin,finalKsMax,objs,rBP,resPath,
                       maxNbPatPerClustList=maxNbPatPerClustList,patPreferences=patPreferences,ncEx=optEx,skipDiscr=skipDiscr)
    writeMat(resPath+'/'+dataNames[0]+'/Results.csv',allRes)

testEvalPipeline("artif0")
testEvalPipeline("artif1")
testEvalPipeline("artif2")
#testEvalPipeline("iris")
#testEvalPipeline("StudentsPor")
#testEvalPipeline("Students")
#testEvalPipeline("Halfmoon",optAlgorithms=['Kmeans',"SpectralNN"])
#testEvalPipeline("Automobile")

#testEvalPipeline("Automobile",optKmeansBL=True)
#testEvalPipeline("Students",optKmeansBL=True)
#testEvalPipeline("NC",optKmeansBL=True)

#testEvalPipeline("NC",optEx=None)
#testEvalPipeline("NC0",optEx=0)
#testEvalPipeline("NC1",optEx=1)
#testEvalPipeline("NC2",optEx=2)
#testEvalPipeline("NC3",optEx=3)
#testEvalPipeline("NC4",optEx=4)
#testEvalPipeline("NC5",optEx=5)
#testEvalPipeline("NC6",optEx=6)
#testEvalPipeline("NC7",optEx=7)
