import csv
import sklearn.utils
import sklearn.datasets
import random
from ucimlrepo import fetch_ucirepo
from dataTreatment import *
from vizual import *
import numpy as np

#---- TWITTER DATA ----
def readTwitterPoliticalData(file):
    print("Read Twitter dataset")
    res=[]
    with open(file, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
                res.append(list(map(int, row)))#row)
    return res

def readTwitterPoliticalHashtags(file):
    print("Read Twitter Hashtags")
    res=[]
    Header=True
    with open(file, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            if(Header):
                Header=False
            else:
                res.append(row[0])
    return res

#Read a file containing the communities associated to each instances
#returns 2 lists, the first containing the usernames and the second a the communities in the form of a partition
def readTwitterCommunities(file):
    print("Read Twitter Communities")
    usernames=[]
    partition=[]
    Header=True
    with open(file, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            if(Header):
                Header=False
            else:
                usernames.append(row[0])
                partition.append(int(row[2])-1) #-1 to start at 0
    return usernames,partition

#Find accounts with less than minAct descriptors
def findInactiveAccounts(data,minAct):
    idx=[]
    for i in range(len(data)):
        if (sum(data[i])<minAct):
            idx.append(i)
    return idx

#return a list l without the elements at the indexes in list idx.
def removeIndexes(l,idx):
    return [l[i] for i in range(len(l)) if i not in idx]

def createTwitterDescrSpace(data,minAct):
    res=[]
    for inst in data:
        instDescr=[]
        for val in inst:
            if val>=minAct:
                instDescr.append(1)
            else:
                instDescr.append(0)
        res.append(instDescr)
    return res

#General process of reading data for 1 of the 3 twitter political data
def loadTwitterData(dataChoice):
    if(dataChoice==0): #USA 1000, 5 communities
        print("USA data")
        twitterDataPath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/StructuralUserHashtagUSA.csv" #how many times users used all hashtags (n x t)
        twitterHashPath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/nodesHashtagUSA.csv"
        twitterCommunitiePath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/comUSA.csv"
    elif(dataChoice==1): #USA 880, 2 communities #TODO : too many attr in nodesHashtag, need to select
        print("USA data (small)")
        twitterDataPath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/StructuralUserHashtagIUR.csv" #how many times users used all hashtags (n x t)
        twitterHashPath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/nodesHashtagUSA.csv"
        twitterCommunitiePath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/Comm1Indicies.csv"
    elif(dataChoice==2): #FRANCE
        print("FRANCE data")
        twitterDataPath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/StructuralHashtagsFrance.csv" #how many times users used all hashtags (n x t)
        twitterHashPath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/nodesHashtagsFrance.csv"
        twitterCommunitiePath="/home/mathieu/Documents/Travail/These/datasets/TwitterElection/comFrance.csv"

    twitterData=readTwitterPoliticalData(twitterDataPath)
    twitterHash=readTwitterPoliticalHashtags(twitterHashPath)
    twitterUsers,twitterCommunities=readTwitterCommunities(twitterCommunitiePath)
    #print(twitterData[0])
    print(len(twitterData)," instances, ",len(twitterData[0])," attributes")
    print(len(twitterHash)," hashtags")
    print(len(twitterUsers)," users")
    print(max(twitterCommunities)+1," communities")

    #Delete users having small amount of posts
    minAct=10
    idx=findInactiveAccounts(twitterData,minAct)
    print(len(idx)," user active with less than ",minAct," hashtags")
    twitterData=removeIndexes(twitterData,idx)
    twitterUsers=removeIndexes(twitterUsers,idx)
    twitterCommunities=removeIndexes(twitterCommunities,idx)
    print(len(twitterData)," instances, ",len(twitterData[0])," attributes")
    print("After there removal :")
    print(len(twitterData)," instances")

    #Convert data to create binary tag space 
    twitterTagSpace=createTwitterDescrSpace(twitterData,minAct)
    return twitterData,twitterTagSpace,twitterHash,twitterUsers,twitterCommunities


#---- FLAGS DATA ----

#read datafile of the flag dataset specificly
def readFlags(dataFile):
    print("Read Flags dataset")
    res=[]
    with open(dataFile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            res.append(row)
    return res

#get the list of all countries names
def getCountriesNames(path):
    data=readFlags(path)
    return [data[i][0] for i in range(len(data))]

#After reading the lfag dataset, separate its attributes to be conform to our requierements
def prepareFlagData(data,attributeNames):
    landmass=["N.America", "S.America", "Europe", "Africa", "Asia", "Oceania"] 
    zone=["NE","SE","SW","NW"]
    language=["English","Spanish","French","German","Slavic","Other Indo-European","Chinese","Arabic","Japanese/Turkish/Finnish/Magyar","Others"]
    religion=["Catholic","Other Christian","Muslim","Buddhist","Hindu","Ethnic","Marxist","Others"]
    tagNames=[landmass,zone,language,religion]
    countryTagIds=[1,2,5,6]
    colourNames=["red","green","blue","gold","white","black","orange"]

    allCountryDescr=[]
    countryTagNames=landmass+zone+language+religion
    for i in range(len(data)):
        iDescr=[]
        for j in range(len(countryTagIds)):
            if(countryTagIds[j]!=6): #because religions are the only category starting at 0
                for v in range(len(tagNames[j])):
                    if(int(data[i][countryTagIds[j]])==v+1):
                        iDescr.append(1)
                    else:
                        iDescr.append(0)
            else:
                for v in range(len(tagNames[j])):
                    if(int(data[i][countryTagIds[j]])==v):
                        iDescr.append(1)
                    else:
                        iDescr.append(0)
        allCountryDescr.append(iDescr)

    countryNumerical=[3,4]
    flagNumerical=[9]
    flagShapes=[7,8,18,19,20,21,22]
    flagBinary=[10,11,12,13,14,15,16,23,24,25,26,27]
    flagColours=[17,28,29] #28,29
    allCountryNum=[]
    allFlagNum=[]
    allFlagBin=[]
    allFlagHue=[]
    allFlagShapes=[]
    allConvertedFlagVal=[]
    allConvertedCountryVal=[]

    medians={}
    convertedFlagValsNames=[]
    convertedCountryValsNames=[]
    hueNames=[]
    flagShapesNames=[attributeNames[i] for i in flagShapes]
    for i in range(len(data[0])-1):
        if i in flagNumerical or i in countryNumerical:
            iData=[float(sublist[i]) for sublist in data]
            print(i,np.mean(iData),np.median(iData),min(iData),max(iData),sum([int(a==0) for a in iData]))
            medians[i]=np.median([float(sublist[i]) for sublist in data])
            if i in flagNumerical:
                convertedFlagValsNames.append(attributeNames[i]+"_inf")
                convertedFlagValsNames.append(attributeNames[i]+"_sup")
            else:
                convertedCountryValsNames.append(attributeNames[i]+"_inf")
                convertedCountryValsNames.append(attributeNames[i]+"_sup")
        if i in flagColours:
            hueNames=hueNames+[attributeNames[i]+"_"+col for col in colourNames]
    print(medians)

    for i in range(len(data)):
        iCountryNum=[]
        iFlagNum=[]
        iFlagBin=[]
        iFlagHue=[]
        iFlagShapes=[]
        convertedFlagVal=[]
        convertedCountryVal=[]
        for j in flagNumerical:
            val=int(data[i][j])
            iFlagNum.append(val)
            if(val>=medians[j]): #convert in descriptive attributes
                convertedFlagVal.append(0)
                convertedFlagVal.append(1)
            else:
                convertedFlagVal.append(1)
                convertedFlagVal.append(0)
        allFlagNum.append(iFlagNum)
        for j2 in flagBinary:
            iFlagBin.append(int(data[i][j2]))
        allFlagBin.append(iFlagBin)

        for j3 in countryNumerical:
            val=int(data[i][j3])
            iCountryNum.append(val)
            if(val>=medians[j3]): #convert in descriptive attributes
                convertedCountryVal.append(0)
                convertedCountryVal.append(1)
            else:
                convertedCountryVal.append(1)
                convertedCountryVal.append(0)
        for j4 in flagColours:
            iFlagHue=iFlagHue+[int(nameCol==data[i][j4]) for nameCol in colourNames]

        for j5 in flagShapes:
            val=int(data[i][j5])
            if(val>=1): #if the shape is present
                iFlagShapes.append(1)
            else:
                iFlagShapes.append(0)
        allCountryNum.append(iCountryNum)
        allFlagNum.append(iFlagNum)
        allFlagBin.append(iFlagBin)
        allFlagHue.append(iFlagHue)
        allFlagShapes.append(iFlagShapes)
        allConvertedFlagVal.append(convertedFlagVal)
        allConvertedCountryVal.append(convertedCountryVal)

    flagTagNames=[attributeNames[t] for t in flagBinary ]
    wholeFlagSpace=[allConvertedFlagVal[i]+allFlagShapes[i]+allFlagBin[i]+allFlagHue[i] for i in range(len(data))]
    wholeFlagSpaceNames=convertedFlagValsNames+ flagShapesNames+flagTagNames+hueNames

    wholeCountrySpace=[allConvertedCountryVal[i]+allCountryDescr[i] for i in range(len(data))]
    wholeCountrySpaceNames=convertedCountryValsNames+countryTagNames

    return wholeFlagSpace,wholeCountrySpace,wholeFlagSpaceNames,wholeCountrySpaceNames

#load the flag dataset given a path to its csv datafile
def loadFlagsData(path):
    raw=readFlags(path) #shape : 194 30
    attributeNames=["name","landmass","zone","area","population","language","religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft  colour","botright Colour"]
    instNames=[raw[i][0] for i in range(len(raw))]
    wholeFlagSpace,wholeCountrySpace,wholeFlagSpaceNames,wholeCountrySpaceNames=prepareFlagData(raw,attributeNames)
    return wholeFlagSpace,wholeCountrySpace,wholeFlagSpaceNames,wholeCountrySpaceNames

#---- AUTOMOBILE DATA ----

#Read the datafile corresponding to the Automobile dataset
def readAutomobileData(dataFile,keepIncompleteData):
    #print("Read Automobile dataset")
    res=[]
    with open(dataFile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            if(not keepIncompleteData):
                if '?' not in row:
                    res.append(row)
            else:
                res.append(row)
    return res

#convert the raw autombile data into a format conform to what is neeeded for our approach
def prepareAutomobileData(data):
    #TODO explains accronyms
    attributeNames=['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type',' num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

    make=["alfa-romero","audi","bmw","chevrolet","dodge","honda","isuzu","jaguar","mazda","mercedes-benz","mercury","mitsubishi","nissan","peugot","plymouth","porsche","renault","saab","subaru","toyota","volkswagen","volvo"]
    fuel_type=["diesel","gas"]
    aspiration=["std","turbo"]
    num_of_doors=["four","two"]
    body_style=["hardtop","wagon","sedan","hatchback","convertible"]
    drive_wheels=["4wd","fwd","rwd"]
    engine_location=["front","rear"]
    engine_type=["dohc","dohcv","l","ohc","ohcf","ohcv","rotor"]
    num_of_cylinders=["eight","five","four","six","three","twelve","two"]
    fuel_system=["1bbl","2bbl","4bbl","idi","mfi","mpfi","spdi","spfi"]
    binDataNames=[make,fuel_type,aspiration,num_of_doors,body_style,drive_wheels,engine_location,engine_type,num_of_cylinders,fuel_system]
    binCoord=[2,3,4,5,6,7,8,14,15,17] #coords of the values to convert to binary data

    prices=[int(sublist[len(sublist)-1]) for sublist in data]

    #find medians for numerical datas
    #They will be of use to convert numerical data into descriptors
    #(does not take into account price)
    medians={}
    convertedValsNames=[]
    for i in range(len(data[0])-1):
        if i not in binCoord:
            medians[i]=np.median([float(sublist[i]) for sublist in data])
            convertedValsNames.append(attributeNames[i]+"<="+str(medians[i]))
            convertedValsNames.append(attributeNames[i]+">"+str(medians[i]))
            #convertedValsNames.append(attributeNames[i]+"_inf")
            #convertedValsNames.append(attributeNames[i]+"_sup")
            #print(attributeNames[i],'median :',medians[i])
    #print(z)
    print(medians)

    numVals=[]
    binVals=[]
    convertedVals=[]
    for inst in data: #for each instances
        numVal=[]
        binVal=[]
        convertedVal=[]
        for i in range(len(inst)-1): #for each attribute
            if (i not in binCoord): #if the attribute is not binary
                val=float(inst[i])
                numVal.append(val)
                if(val>=medians[i]): #convert in descriptive attributes
                     convertedVal.append(0)
                     convertedVal.append(1)
                else:
                     convertedVal.append(1)
                     convertedVal.append(0)
            else:
                index=binCoord.index(i)
                for o in binDataNames[index]:
                    if inst[i]==o:
                        binVal.append(1)
                    else:
                        binVal.append(0)
        numVals.append(numVal)
        binVals.append(binVal)
        convertedVals.append(convertedVal)

    binValsNames=[item for sublist in binDataNames for item in sublist] #compute the novel names of the attributes
    priceDistances=[[abs(e1-e2) for e2 in prices] for e1 in prices] #compute a distance matrix only based on the price attribute
    wholeDescriptorSpaceNames=convertedValsNames+binValsNames
    wholeDescriptorSpace=[convertedVals[i]+binVals[i] for i in range(len(prices))]
    return numVals,binVals,binValsNames,prices,convertedVals,convertedValsNames,priceDistances,wholeDescriptorSpace,wholeDescriptorSpaceNames

#get statistics on the prices of cars of the Automobile dataset organised in different clusters
def getClusterPrices(part):
    autoPath="/home/mathieu/Documents/Travail/These/datasets/Automobile/imports-85.data"
    numVals,binVals,binValsNames,prices,convertedVals,convertedValsNames,priceDistances,wholeDescriptorSpace,wholeDescriptorSpaceNames=prepareAutomobileData(readAutomobileData(autoPath,False)) #Done only to get all prices
    for clust in part:
        clustPrices=[]
        for i in clust:
            clustPrices.append(prices[i])
        print(len(clust)," : ",round(np.mean(clustPrices),2),round(np.median(clustPrices),2),round(np.std(clustPrices),2))


def postOccAutoAnalysis(Clustering,nbOfClusterForEachInstance):
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
    numVals,binVals,binValsNames,prices,convertedVals,convertedValsNames,priceDistances,wholeDescriptorSpace,wholeDescriptorSpaceNames=prepareAutomobileData(readAutomobileData('/home/mathieu/Documents/Travail/These/datasets/Automobile/imports-85.data',False)) #AUTOMOBILE data
    
    resultFolderPath="/home/mathieu/Images"
    labels=genLabels(Clustering,len(nbOfClusterForEachInstance),nbOfClusterForEachInstance)
    #lauchTSNE(wholeDescriptorSpace,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_DescrSpace10",10, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)
    #lauchTSNE(wholeDescriptorSpace,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_DescrSpace20",20, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)  
    #lauchTSNE(wholeDescriptorSpace,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_DescrSpace1",1, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000) 
    #lauchTSNE(wholeDescriptorSpace,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_DescrSpace1",5, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)
    
    lauchTSNE(binVals,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_binVals5",5, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)
    lauchTSNE(binVals,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_binVals10",10, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)
    lauchTSNE(binVals,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_binVals20",20, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)
    lauchTSNE(binVals,labels,len(Clustering)+1,resultFolderPath,"TSNE_auto_binVals30",30, TSNEmetric='euclidean',TSNEinit='random',TSNEiter=5000)
    
    #Prices
    for k in range(len(Clustering)):
        kPrices=[prices[o] for o in Clustering[k]]
        print('Cluster',k,'prices: min=',np.min(kPrices),'max=',np.max(kPrices),'mean=',round(np.mean(kPrices),0),'std=',round(np.std(kPrices),0),'median=',round(np.median(kPrices),0))


#autoRes=[[3, 32, 44, 45, 46, 47, 48], [0, 1, 2, 4, 5, 6, 7, 18, 30, 43, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 90, 93, 94, 95, 96, 131, 137, 138, 139, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 77, 84, 85, 86, 87, 88, 89, 91, 92, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 140, 141, 142, 143, 144, 145, 146, 147]]
#autoNbOfClusterForEachInstance=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#postOccAutoAnalysis(autoRes,autoNbOfClusterForEachInstance)

#-- Adult Dataset --

def prepareAdultDataset(path):

    label=['>50K','<=50K']

    age=['continuous']
    workclass=['Private','Self-emp-not-inc','Self-emp-inc','Federal-gov','Local-gov','State-gov','Without-pay','Never-worked']
    fnlwgt=['continuous']
    education=['Bachelors','Some-college','11th','HS-grad','Prof-school','Assoc-acdm','Assoc-voc','9th','7th-8th','12th','Masters','1st-4th','10th','Doctorate','5th-6th','Preschool']
    education_num=['continuous']
    marital_status=['Married-civ-spouse','Divorced','Never-married','Separated','Widowed','Married-spouse-absent','Married-AF-spouse']
    occupation=['Tech-support','Craft-repair','Other-service','Sales','Exec-managerial','Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv','Protective-serv','Armed-Forces']
    relationship=['Wife','Own-child','Husband','Not-in-family','Other-relative','Unmarried']
    race=['White','Asian-Pac-Islander','Amer-Indian-Eskimo','Other','Black'] #8
    sex=['Female','Male'] #9
    capital_gain=['continuous']
    capital_loss=['continuous']
    hours_per_week=['continuous']
    native_country=['United-States','Cambodia','England','Puerto-Rico','Canada','Germany','Outlying-US(Guam-USVI-etc)','India','Japan','Greece','South','China','Cuba','Iran','Honduras','Philippines','Italy','Poland','Jamaica','Vietnam','Mexico','Portugal','Ireland','France','Dominican-Republic','Laos','Ecuador','Taiwan','Haiti','Columbia','Hungary','Guatemala','Nicaragua','Scotland','Thailand','Yugoslavia','El-Salvador','Trinadad&Tobago','Peru','Hong','Holand-Netherlands']
    dataNames=[age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,label]

    print(" Read Adult dataset :")
    data=readAutomobileData(path,False)
    SensibleIds=[] #8,9,13

    tagIds=[]
    tagNames=[]
    for j in range(len(dataNames)):
        if j not in SensibleIds:
            if len(dataNames[j])>1: #check if is numerical data
                tagIds.append(j)
                tagNames=tagNames+dataNames[j]

    numData=[]
    tagData=[]
    for i in range(len(data)):
        iNum=[]
        iTag=[]
        #print('debug i:',i,data[i])
        for j in range(len(dataNames)):
            if j not in SensibleIds:
                if len(dataNames[j])>1: #check if is numerical data
                    iTag=iTag+[int(data[i][j]==t) for t in dataNames[j]]
                else:
                    iNum.append(int(data[i][j]))
        numData.append(iNum)
        tagData.append(iTag)

    #Sim=computeAdultSimLabel(data)
    return numData,tagData,tagNames#,Sim

def computeAdultSimLabel(data):
    labelInd=len(data[0])-1
    res=[]
    for i in data:
        iVals=[]
        for j in data:
            if i[labelInd]==j[labelInd]:
                iVals.append(1)
            else:
                iVals.append(0)
        res.append(iVals)
    return res

#---- Old Artificial 2D datasets ----
def OldPrepareArtif():
    '''Get the small visual artificial dataset.'''
    artFeatures=[[1,0],[0,1],[0,2],[1,1.75],[2,1.5],[3,0],[2,2.5],
              [0.5,5.5],[1,4.5],[1,6.5],[2,5.5],
                [3.5,3],[4,4],[4,1.75],[4.5,0],[5,3],[4.25,5.25]]
    gt=[0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2]
    artTags=[[1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],
            [0,1,0,0,1],[0,1,1,0,0],[0,1,1,0,0],[0,1,1,0,0],
            [0,1,0,0,1],[0,1,0,0,1],[1,0,0,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,0,1]]
    artTagNames=['round','square','white','red','green']
    #vizualizePartition(3,artFeatures,gt,'testArt')
    return artFeatures,artTags,artTagNames,gt


def prepareHalfmoonRandom(precomputed=False,shapeBias=True,showDescr=False):
    '''Get the halfmoon dataset with random descriptors.'''
    Halfmoon=[[0.81680544,0.5216447],[1.61859642,-0.37982927],[-0.02126953,0.27372826],[-1.02181041,-0.07543984],[1.76654633,-0.17069874],[1.8820287,-0.04238449],[0.97481551,0.20999374],[0.88798782,-0.48936735],[0.89865156,0.36637762],[1.11638974,-0.53460385],[-0.36380036,0.82790185],[0.24702417,-0.23856676],[1.81658658,-0.13088387],[1.2163905,-0.40685761],[-0.8236696,0.64235178],[0.98065583,0.20850838],[0.54097175,0.88616823],[0.348031,-0.30101351],[0.35181497,0.88827765],[-0.77708642,0.82253872],[1.92590813,0.01214146],[0.86096723,-0.47653216],[0.19990695,0.99672359],[1.2895301,-0.37071087],[-0.27847636,1.02538452],[0.24187916,-0.07627812],[1.84988768,-0.09773674],[1.88406869,0.0449402],[0.165822,-0.08613126],[0.13861369,0.89639036],[0.89087024,0.52265882],[-0.22806587,0.84091882],[0.98279208,-0.46457771],[0.04237749,0.19457898],[0.76422612,0.67223332],[1.91108938,0.21178339],[0.43608432,-0.23007221],[0.96186938,0.09923426],[-0.84336684,0.52414334],[-0.04122466,0.35721873],[0.55507653,-0.42493298],[-0.4388286,0.85940389],[0.6532646,0.71235382],[0.10274835,0.06721414],[1.5486824,-0.34012196],[-0.37318371,0.95506411],[1.01706978,0.19210044],[-0.71923685,0.65476676],[0.16135772,-0.10771978],[0.86434045,-0.4594568],[-0.69717533,0.80133734],[0.32791175,-0.19619019],[1.98046734,0.03848682],[-0.90479784,0.05723938],[1.04515397,-0.50020349],[0.7534213,0.65688005],[0.54968577,0.73635744],[1.24038086,-0.47577903],[0.24918868,0.94246199],[-0.20756105,0.99290594],[0.35136403,-0.29065432],[-1.01628753,0.16290244],[1.78137056,-0.1244931],[0.87423825,0.53065346],[1.09997644,-0.46733763],[-1.07022744,0.2365448],[-0.15869858,1.01497482],[1.46569247,-0.3808977],[0.03025209,0.97792142],[-0.9365943,0.45674926],[0.66038307,-0.46576222],[-0.99144728,0.40662094],[0.46339847,-0.46605416],[-0.132006,0.52447234],[0.81566997,-0.42821617],[-0.94820947,0.37717096],[0.05300205,0.18597406],[0.92648634,0.40988975],[0.60689997,0.78279323],[0.72961391,-0.37215252],[1.9796026,0.12425417],[-0.02053902,0.97601558],[0.63818364,-0.49916763],[2.00639179,0.44597642],[0.02315539,0.24035667],[-0.35883877,1.02716833],[0.95414653,0.04177433],[-0.33921532,0.96308888],[0.59950492,-0.39774852],[1.99019644,0.39360049],[0.33125729,0.9365782],[0.99460422,0.35063363],[1.98845457,0.2628361],[-0.67473718,0.76419738],[2.00751107,0.3651166],[1.78298331,-0.11490401],[1.73616653,-0.22781554],[0.40646216,-0.25422904],[-1.02505346,0.24337404],[0.06414296,0.07759793],[1.30092145,-0.58089757],[1.97425572,0.30889897],[0.03228388,1.07937745],[1.03086156,-0.02389082],[-0.90062492,0.30653639],[0.08068561,0.29131373],[-0.98807765,0.1039765],[-0.47394435,0.96143212],[1.54651932,-0.35008497],[0.23332453,0.89648984],[-0.58481687,0.80318956],[0.0374878,1.02322111],[-0.01943215,1.07001032],[-0.85323667,0.39896937],[0.92635535,0.37695326],[1.43250553,-0.50148981],[0.60622756,0.66229531],[1.94401554,0.13685573],[0.57984414,-0.39868907],[0.74317519,0.50998316],[0.87116686,0.54105191],[-0.71045745,0.57281877],[-0.03081568,0.33644614],[-0.0298505,0.99553114],[-0.06313347,0.42194174],[-0.79223214,0.68354165],[0.92098434,0.04171051],[0.17794377,0.04536893],[1.34934828,-0.3941652],[1.98387143,0.50898445],[1.00104892,0.27158454],[-0.5425424,0.76257612],[-0.9969011,0.47226403],[0.23408511,-0.15381658],[1.21437019,-0.40862022],[1.60101745,-0.17940652],[1.15844202,-0.40408591],[-1.00922523,0.2161359],[2.01865957,0.50313426],[0.88839866,0.39017093],[0.10170896,-0.01206481],[-0.01241966,0.47064905],[0.44566504,0.94595998],[-0.3569344,0.98319206],[-0.43845037,0.88374167],[1.01534178,0.06687469],[0.2310607,0.01153495],[1.35098772,-0.44520507],[0.25423421,1.0205525],[-0.00586456,0.24919627],[0.4752852,-0.37028432],[1.68071768,-0.34775296],[0.84564282,0.45629647],[0.34218757,0.90613948],[0.58741368,-0.35078742],[-0.17818292,0.96641541],[1.25865528,-0.4740009],[0.33542814,-0.18023343],[0.52630774,0.94876068],[0.6424051,0.77717105],[0.15770292,0.04709417],[1.11178863,-0.5065278],[0.60370903,0.83759912],[1.48247118,-0.32721961],[0.39793421,-0.36876588],[1.67240934,-0.09328043],[0.47551295,0.85547255],[0.70605116,-0.42241887],[1.56418943,-0.34860626],[0.94012854,-0.57508877],[0.61400301,0.83833823],[-1.07139757,0.02669316],[-0.91308996,0.52626435],[-0.74824469,0.51823742],[0.14688241,0.0297201],[0.94362014,-0.44829425],[1.84489829,0.40601924],[-0.66827347,0.69085682],[-0.7362418,0.59951884],[0.60146482,0.72551706],[1.47437703,-0.37541022],[-0.88760005,0.50864517],[1.92892164,0.18201791],[1.78673422,-0.27470711],[1.95130228,0.26574549],[0.33471666,0.98057089],[-0.16884749,0.89206411],[0.77063994,-0.51750338],[-0.88700503,0.36696366],[-0.62886492,0.79087211],[-0.93006783,0.38754885],[0.42447858,0.93268774],[0.80861392,0.53599924],[0.94000928,0.27111431],[-0.01609181,0.37369612],[-0.53633385,0.86026837],[1.88281749,0.24435589],[0.17575161,-0.007231],[0.12423604,1.00790161],[1.62152568,-0.22328525]]
    HalfmoonGT=[0,1,1,0,1,1,0,1,0,1,0,1,1,1,0,0,0,1,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,1,0,0,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,1,0,1]
    #Halfmoon,HalfmoonGT=sklearn.datasets.make_moons(n_samples=200,noise=0.05)
    N=len(Halfmoon)
    K=2
    halfmoonTagsNames=['blue','red','white','round','square','triangle','small','big']
    
    clustHalfmoon=[[i for i in range(N) if HalfmoonGT[i]==k] for k in range(K)]
    if(precomputed):
        descrSpace=readDataFile('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/halfmoonRandomTag.csv')
        #descrSpace=readDataFile('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/halfmoonLessRandomTag.csv')
        descrSpace=[[int(i) for i in line] for line in descrSpace]
    else:    
        colorsC0=select_random_Ns(clustHalfmoon[0], int(0.8*len(clustHalfmoon[0]))) #sklearn.utils.random.sample_without_replacement(len(clustHalfmoon[0]), 0.8*len(clustHalfmoon[0]), method='auto', random_state=None)  
        blueC0=colorsC0[0]
        colorsC1=select_random_Ns(clustHalfmoon[1], int(0.8*len(clustHalfmoon[1]))) #sklearn.utils.random.sample_without_replacement(len(clustHalfmoon[1]), 0.8*len(clustHalfmoon[1]), method='auto', random_state=None)  
        redC1=colorsC1[0]
        white=colorsC0[1]+colorsC1[1]
        if(not shapeBias):
            #select randomly 75% of elements in C0 to have blue tag, 75% of elements in C1 to have red tag, and designate others as green 
            allSquare=[]
            allRound=[]
            allTri=[]
            for c in range(K):
                cSquare,cRound,cTri=select_random_Ns(clustHalfmoon[c], 45)
                allSquare+=[i for i in cSquare]
                allRound+=[i for i in cRound]
                allTri+=[i for i in cTri]
        else:
            bsq=select_random_Ns(blueC0, int(0.55*len(clustHalfmoon[0])))
            blueSquare=bsq[0]
            bluenotsquare=bsq[1]
            bss=select_random_Ns(bluenotsquare, int(0.15*len(clustHalfmoon[0])))
            blueRound=bss[0]
            blueTriangle=bss[1]
            w0s=select_random_Ns(colorsC0[1], int(0.1*len(clustHalfmoon[0])))
            white0sqr=w0s[0]
            white0round=w0s[1]

            rsq=select_random_Ns(redC1, int(0.55*len(clustHalfmoon[1])))
            redRound=rsq[0]
            rednotRound=rsq[1]
            rss=select_random_Ns(rednotRound, int(0.15*len(clustHalfmoon[1])))
            redSquare=rss[0]
            redTriangle=rss[1]
            w1s=select_random_Ns(colorsC1[1], int(0.1*len(clustHalfmoon[1])))
            white1round=w1s[0]
            white1sqr=w1s[1]

            allSquare=blueSquare+white0sqr+redSquare+white1sqr
            allRound=blueRound+white0round+redRound+white1round
            allTri=blueTriangle+redTriangle

        instSizes=select_random_Ns([i for i in range(N)], int(N/2))

        #create descr space
        descrSpace=[]
        for i in range(N):
            line=[]
            #Colors
            if(i in blueC0):
                line.append(1)
            else:
                line.append(0)
            if(i in redC1):
                line.append(1)
            else:
                line.append(0)
            if(i in white):
                line.append(1)
            else:
                line.append(0)
            #Shapes
            if(i in allSquare):
                line.append(1)
            else:
                line.append(0)
            if(i in allRound):
                line.append(1)
            else:
                line.append(0)
            if(i in allTri):
                line.append(1)
            else:
                line.append(0)
            if(i in instSizes[0]):
                line.append(1)
                line.append(0)
            else:
                line.append(0)
                line.append(1)
            descrSpace.append(line)
        #print(len(descrSpace[0]),len(halfmoonTagsNames))
        #print(allSquare)
        #print(allRound)
        #print(allTri)
        #print(descrSpace[0])
        #print(descrSpace[1])
        #writeMat('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/halfmoonLessRandomTag.csv',descrSpace)
        writeMat('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/halfmoonRandomTag.csv',descrSpace)
        #writeMat('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/halfmoonRandom.csv',Halfmoon)
        #writeMat('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/halfmoonRandomGT.csv',HalfmoonGT)
    
    colorsPart=[]
    shapesPart=[]
    sizesPart=[]
    for i in range(N): #'blue','red','white','round','square','triangle','small','big'
        if(descrSpace[i][0]==1):
            colorsPart.append(0)
        elif(descrSpace[i][1]==1):
            colorsPart.append(1)
        elif(descrSpace[i][2]==1):
            colorsPart.append(2)

        if(descrSpace[i][3]==1):
            shapesPart.append(0)
        if(descrSpace[i][4]==1):
            shapesPart.append(1)
        if(descrSpace[i][5]==1):
            shapesPart.append(2)

        if(descrSpace[i][6]==1):
            sizesPart.append(0)
        if(descrSpace[i][7]==1):
            sizesPart.append(1)

    if(showDescr):
        showArtifData(Halfmoon,colorsPart,shapesPart,sizesPart)
        #print(z)
    
    #cluster composition (use LCM ?)
    from skmine.itemsets import LCM
    for k in range(K):
        tagsInC=getTagsIdsForClusterInstances(descrSpace,clustHalfmoon[k])
        #print('tagsInC:',tagsInC)
        lcm = LCM(min_supp=1)
        patterns = lcm.fit_discover(tagsInC)
        patK= patterns.values.tolist()
        patKnames=[[[halfmoonTagsNames[t] for t in pat[0]],pat[1]] for pat in patK]
        #print('patK:',patK)
        print('possible pat for C'+str(k)+':',patKnames)

    discretizedTag,discretizedTagNames=convertNumToBin(Halfmoon,['X','Y'],2)
    #print(discretizedTag)
    #print(z)

    return Halfmoon,HalfmoonGT,halfmoonTagsNames,descrSpace,clustHalfmoon,colorsPart,shapesPart,sizesPart,discretizedTag,discretizedTagNames

def prepareCirclesRandom(precomputed=False):
    '''Get the circles dataset with random descriptors.'''
    circles,circlesGT=noisy_circles = sklearn.datasets.make_circles(n_samples=200, factor=0.5, noise=0.01)
    N=len(circles)
    K=2
    circlesTagsNames=['blue','red','white','round','square','triangle','small','big']
    #vizualizePartition(K,circles,circlesGT,'moons')
    
    if(precomputed):
        descrSpace=readDataFile('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/circlesRandomTag.csv')
        descrSpace=[[int(i) for i in line] for line in descrSpace]
    else:
        clustcircles=[[i for i in range(N) if circlesGT[i]==k] for k in range(K)]
        #TODO select randomly 75% of elements in C0 to have blue tag, 75% of elements in C1 to have red tag, and designate others as green 
        allSquare=[]
        allRound=[]
        allTri=[]
        for c in range(K):
            cSquare,cRound,cTri=select_random_Ns(clustcircles[c], 45)
            allSquare+=[i for i in cSquare]
            allRound+=[i for i in cRound]
            allTri+=[i for i in cTri]
        colorsC0=select_random_Ns(clustcircles[0], int(0.9*len(clustcircles[0]))) #sklearn.utils.random.sample_without_replacement(len(clustHalfmoon[0]), 0.8*len(clustHalfmoon[0]), method='auto', random_state=None)  
        blueC0=colorsC0[0]
        colorsC1=select_random_Ns(clustcircles[1], int(0.9*len(clustcircles[1]))) #sklearn.utils.random.sample_without_replacement(len(clustHalfmoon[1]), 0.8*len(clustHalfmoon[1]), method='auto', random_state=None)  
        redC1=colorsC1[0]
        white=colorsC0[1]+colorsC1[1]
        instSizes=select_random_Ns([i for i in range(N)], int(N/2))

        #create descr space
        descrSpace=[]
        for i in range(N):
            line=[]
            #Colors
            if(i in blueC0):
                line.append(1)
            else:
                line.append(0)
            if(i in redC1):
                line.append(1)
            else:
                line.append(0)
            if(i in white):
                line.append(1)
            else:
                line.append(0)
            #Shapes
            if(i in allSquare):
                line.append(1)
            else:
                line.append(0)
            if(i in allRound):
                line.append(1)
            else:
                line.append(0)
            if(i in allTri):
                line.append(1)
            else:
                line.append(0)
            if(i in instSizes[0]):
                line.append(1)
                line.append(0)
            else:
                line.append(0)
                line.append(1)
            descrSpace.append(line)
        print(len(descrSpace[0]),len(circlesTagsNames))
        print(allSquare)
        print(allRound)
        print(allTri)
        print(descrSpace[0])
        print(descrSpace[1])
        writeMat('/home/mathieu/Documents/Travail/These/git/stage-involvd-mathieu-guilbert-2021/Devs/data/circlesRandomTag.csv',descrSpace)

    return circles,circlesGT,circlesTagsNames,descrSpace


def select_random_Ns(l, k):
    '''randomly put all elements of list l into sublist of size k.

    Source: https://www.geeksforgeeks.org/python-select-random-value-from-a-list'''
    random.shuffle(l)
    res = []
    for i in range(0, len(l), k):
        res.append(l[i:i + k])
    return res

#prepareHalfmoonRandom()

#---- IRIS datasets ----

#load the UCI dataset iris
#nbDec : option on how to discretize the data. 2 will do according to the mediane, and 4 according to quartiles
def loadUCIDataset(discretizeValueList:list):
    from sklearn.datasets import load_iris
    iris = load_iris()
    X=iris.data
    y=iris.target
    attributeNames=["sepal_length","sepal_width","petal_length","petal_width"]
    attributeShortNames=["sl","sw","pl","pw"]
    
    descriptors=[]
    descriptorsNames=[]
    for nbDec in discretizeValueList:
        d,n=convertNumToBin(X,attributeShortNames,nbDec)
        if(descriptors!=[]):
            descriptors=[d[i]+descriptors[i] for i in range(len(descriptors))]
        else:
            descriptors=d
        descriptorsNames=n+descriptorsNames

    return X,descriptors,descriptorsNames,y,iris.target_names


def convertNumToBin(X:list,attributeNames:list,nbDecom:int):
    '''Discretization
    
    PARAMETERS
    --------
    X: dataset
    attributeNames: list of the names of the attributes
    nbDecom: option on how to discretize the data. 2 = depending on the mediane, 4 = quartiles.'''
    if(nbDecom==2):
        medians={}
        convertedValsNames=[]
        for i in range(len(X[0])):
            medians[i]=np.median([float(sublist[i]) for sublist in X])
            convertedValsNames.append(attributeNames[i]+"_m1") #"_inf")
            convertedValsNames.append(attributeNames[i]+"_m2") #"_sup")
        print(medians)
        descriptors=[]
        for i in range(len(X)):
            descr=[]
            for j in range(len(X[0])):
                if(X[i][j]>=medians[j]):
                    descr.append(0)
                    descr.append(1)
                else:
                    descr.append(1)
                    descr.append(0)
            descriptors.append(descr)
    elif(nbDecom==3):
        quar33={}
        quar66={}
        convertedValsNames=[]
        for i in range(len(X[0])):
            iX=[float(sublist[i]) for sublist in X]
            quar33[i]=np.percentile(iX,33)
            quar66[i]=np.percentile(iX,55)
            convertedValsNames.append(attributeNames[i]+"1")
            convertedValsNames.append(attributeNames[i]+"2")
            convertedValsNames.append(attributeNames[i]+"3")
        descriptors=[]
        for i in range(len(X)):
            descr=[]
            for j in range(len(X[0])):
                if(X[i][j]<quar33[j]):
                    descr.append(1)
                    descr.append(0)
                    descr.append(0)
                elif(X[i][j]<quar66[j]):
                    descr.append(0)
                    descr.append(1)
                    descr.append(0)
                else:
                    descr.append(0)
                    descr.append(0)
                    descr.append(1)
            descriptors.append(descr)
    elif(nbDecom==4):
        quar25={}
        quar50={}
        quar75={}
        convertedValsNames=[]
        for i in range(len(X[0])):
            iX=[float(sublist[i]) for sublist in X]
            quar25[i]=np.percentile(iX,25)
            quar50[i]=np.percentile(iX,50)
            quar75[i]=np.percentile(iX,75)
            convertedValsNames.append(attributeNames[i]+"1")
            convertedValsNames.append(attributeNames[i]+"2")
            convertedValsNames.append(attributeNames[i]+"3")
            convertedValsNames.append(attributeNames[i]+"4")
        descriptors=[]
        print(quar50)
        for i in range(len(X)):
            descr=[]
            for j in range(len(X[0])):
                if(X[i][j]<quar25[j]):
                    descr.append(1)
                    descr.append(0)
                    descr.append(0)
                    descr.append(0)
                elif(X[i][j]<quar50[j]):
                    descr.append(0)
                    descr.append(1)
                    descr.append(0)
                    descr.append(0)
                elif(X[i][j]<quar75[j]):
                    descr.append(0)
                    descr.append(0)
                    descr.append(1)
                    descr.append(0)
                else:
                    descr.append(0)
                    descr.append(0)
                    descr.append(0)
                    descr.append(1)
            descriptors.append(descr)
    
    return descriptors,convertedValsNames

#---- Secondary Mushroom ----

#TODO convert categorical attributes
def prepareSecondaryMushroom():
    # fetch dataset 
    secondary_mushroom  = fetch_ucirepo(id=848) 
    #first mushroom is = fetch_ucirepo(id=73) , but it does not have numerical features

    capshape=['b', 'c', 'x', 'f','s', 'p', 'o']
    capsurface=['i', 'g','y','s','h', 'l','k','t','w', 'e']
    capcolor=['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k']
    gillattachment=['a', 'x', 'd', 'e', 's','p', 'f', '?']
    gillspacing=['c', 'd', 'f']
    gillcolor=['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k','f']
    stemroot=['b','s', 'c', 'u','e','z', 'r']
    stemsurface=['i', 'g','y','s','h', 'l','k','t','w', 'e','f']
    stemcolor=['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k','f']
    veiltype=['p', 'u']
    veilcolor=['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k','f']
    ringtype=['c', 'e', 'r', 'g','l', 'p', 's', 'z', 'y', 'm', 'f', '?']
    sporeprintcolor=['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k']
    habitat=['g', 'l', 'm', 'p', 'h', 'u', 'w', 'd']
    season=['s', 'u', 'a', 'w']
    
    # data (as pandas dataframes) 
    X = secondary_mushroom.data.features 
    y = secondary_mushroom.data.targets
    N=len(X)

    propNames=secondary_mushroom.variables.name.to_numpy().tolist()[1:]
    #print('debug x0',X[0])
    #print('debug proNames mushroom:',propNames)
    tagNames=propNames.copy()

    #2 classes: edible and poisous
    gt=[int(i=='e') for i in y.to_numpy().flatten()] #y #[int(x[0]=='e') for x in X]
    #print('debug gt:',gt)
    
    data=X.to_numpy().tolist()

    numIds=[1,9,10]#starting from 0 counting the calss column
    featureSpace=[]

    for i in range(N):
        featureLine=[]
        for id in numIds:
            featureLine.append(data[i][id-1])
        featureSpace.append(featureLine)

    binIds=[5,16]
    nominalIds=[2,3,4,6,7,8,11,12,13,14,15,17,18,19,20]
    nominal=[capshape,capsurface,capcolor,gillattachment,gillspacing,gillcolor,stemroot,stemsurface,stemcolor,veiltype,veilcolor,ringtype,sporeprintcolor,habitat,season]

    tagNames=[]
    tagSpace=[]
    for feature in binIds:
        tagLine=[]
        tagNames.append(propNames[feature-1])
        for i in range(N):
            tagLine.append(int(data[i][feature-1]=='t'))
        tagSpace.append(tagLine)
    #print('debug bin passed',len(tagSpace),len(tagSpace[0]))
            
    for f in range(len(nominalIds)):
        feature=nominalIds[f]
        #print(f,'out of',len(nominalIds))
        for attr in nominal[f]: #for each possible attribute
            #print('attr:',attr,'for',propNames[feature-1])
            tagLine=[]
            tagNames.append(propNames[feature-1]+'_'+attr)
            for i in range(N):
                tagLine.append(int(data[i][feature-1]==attr)) #if empty attr then all will be set to 0.
            tagSpace.append(tagLine)
    #print('debug categorical passed;',len(tagSpace),len(tagSpace[0]))

    transposed_tagSpace = np.transpose(tagSpace)
    #print('debug transposed;',len(transposed_matrix),len(transposed_matrix[0]))
    #print('debug names',tagNames,len(tagNames))
    print('- End read SecondaryMushroom -')
    #print(len(featureSpace),len(featureSpace[0]),len(tagNames),len(gt))
    return featureSpace,transposed_tagSpace,tagNames,gt

#test mushroom
#prepareSecondaryMushroom()
#print(z)


#---- Zoo ----

def prepareZoo():
    zoo = fetch_ucirepo(id=111) # fetch dataset (as pandas dataframes) 
    X = zoo.data.features 
    y = zoo.data.targets
    propNames=zoo.variables.name.to_numpy().tolist()[1:]
    tagNames=propNames.copy()
    #zoo.target_names
    binAtIds=[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15]
    #binAtIds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    data=X.to_numpy().tolist()
    gt=[i-1 for i in y.to_numpy().flatten()]
    P=gt
    K=7
    instanceNames=zoo.data.ids.to_numpy().flatten().tolist()
    binData=[data[i].copy() for i in range(len(data))]#convertNumToBin(X,propNames,2)
    for d in binData:
        del d[12]
    del tagNames[12]
    return data,binData,K,P,propNames,tagNames,instanceNames


#---- Student Performance ----

def prepareStudentPerfDataset(datapath,op=None):
    print("Read Student Performance dataset")
    if(op==None):
        path=datapath
    elif(op!=1):
        path="/home/mathieu/Documents/Travail/These/datasets/StudentPerf/student-mat.csv"
    else:
        path="/home/mathieu/Documents/Travail/These/datasets/StudentPerf/student-por.csv"
    gradeId=[30,31,32] #G1,G2,G3
    data=readDataFile(path)
    attributeNames=data.pop(0)

    school=["GP","MS"] #attribute 0
    sex=["F","M"] #1
    address=["U","R"] #3
    famsize=["LE3","GT3"] #4
    Pstatus=["T","A"] #5
    Medu=["none","primary education (4th grade)","5th to 9th grade","secondary education","higher education"] #6
    Fedu=["none","primary education (4th grade)","5th to 9th grade","secondary education","higher education"] #7
    Mjob=["teacher","health","services","at_home","other"] #8
    Fjob=["teacher","health","services","at_home","other"] #9
    reason=["home","reputation","course","other"] #10
    guardian=["mother","father","other"] #11
    traveltime=["<15 min","15 to 30 min","30 min. to 1 hour",">1 hour"] #12
    studytime=["<2 hours","2 to 5 hours","5 to 10 hours",">10 hours"] #13

    numtobinIds0=[6,7]
    numtobinIds1=[12,13]
    numtobin0=[Medu,Fedu]
    numtobin1=[traveltime,studytime] #WARNING : starts from 1 and not 0
    binIds=[15,16,17,18,19,20,21,22]
    numIds=[2,14,23,24,25,26,27,28,29]
    nominalIds=[0,1,3,4,5,8,9,10,11]
    nominal=[school,sex,address,famsize,Pstatus,Mjob,Fjob,reason,guardian]

    medians={}
    convertedValsNames=[]
    for i in numIds:
        medians[i]=np.median([float(sublist[i]) for sublist in data])
        convertedValsNames.append(attributeNames[i]+"<="+str(medians[i]))#+"_inf")
        convertedValsNames.append(attributeNames[i]+">"+str(medians[i]))#+"_sup")

    gradeData=[]
    tagData=[]
    for inst in data:
        iGradeData=[]
        iTagData=[]
        for j in range(len(numtobin0)):
            attr=numtobin0[j]
            attrId=numtobinIds0[j]
            for val in range(len(attr)):
                if inst[attrId]==val :
                    iTagData.append(1)
                else:
                    iTagData.append(0) 

        for j in range(len(numtobin1)):
            attr=numtobin1[j]
            attrId=numtobinIds1[j]
            for val in range(len(attr)):
                if inst[attrId]==(val+1) :
                    iTagData.append(1)
                else:
                    iTagData.append(0) 

        for j in range(len(nominalIds)):
            attr=nominal[j]
            attrId=nominalIds[j]
            for val in attr:
                if inst[attrId]==val :
                    iTagData.append(1)
                else:
                    iTagData.append(0) 

        for j in binIds:
            if inst[j]=="yes" :
                iTagData.append(1)
            else:
                iTagData.append(0) 

        for j in numIds:
            if float(inst[j])>medians[j] :
                iTagData.append(0) #inf
                iTagData.append(1) #sup
            else:
                iTagData.append(1)
                iTagData.append(0)

        for j in gradeId:
                iGradeData.append(int(inst[j]))
    
        gradeData.append(iGradeData)
        tagData.append(iTagData)

    #numbtobin0,numbtoBin1,nominal,bin,converted
    binNames=[attributeNames[j] for j in binIds]
    tagNames=Medu+Fedu+traveltime+studytime+school+sex+address+famsize+Pstatus+Mjob+Fjob+reason+guardian+binNames+convertedValsNames
    
    return gradeData,tagData,tagNames


#------------ AWA2 ----------------

#read only the data at the specified rows in list ids
def readSubsetData(path,ids):
    res=[]
    cpt=0
    with open(path, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            if cpt in ids:
                res.append(row)
            cpt=cpt+1
    return res

#read data file with space as delimitor
def readSpacedData(path):
    res=[]
    with open(path, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            res.append(row)
    return res

#aread data file with \t as delimitor
def readTData(path):
    res=[]
    with open(path, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in datareader:
            res.append(row[1])
    return res

#prepare the AWA dataset to be lauched on our dataset
def prepareAWA2():
    print("READ AWA 2 DATASET")
    pathClassAttr="/home/mathieu/Documents/Travail/These/datasets/AwA2/AwA2-base/Animals_with_Attributes2/predicate-matrix-binary.txt"
    pathClassNames="/home/mathieu/Documents/Travail/These/datasets/AwA2/AwA2-base/Animals_with_Attributes2/classes.txt"
    pathTagNames="/home/mathieu/Documents/Travail/These/datasets/AwA2/AwA2-base/Animals_with_Attributes2/predicates.txt"
    pathLabels="/home/mathieu/Documents/Travail/These/datasets/AwA2/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
    pathFeatures="/home/mathieu/Documents/Travail/These/datasets/AwA2/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
    N=37322

    #Read files
    labels=readDataFile(pathLabels)
    tagNames=readTData(pathTagNames)
    allclassNames=["antelope","grizzly+bear","killer+whale","beaver","dalmatian","persian+cat","horse","german+shepherd","blue+whale"
                ,"siamese+cat","skunk","mole","tiger","hippopotamus","leopard","moose","spider+monkey","humpback+whale","elephant"
                ,"gorilla","ox","fox","sheep","seal","chimpanzee","hamster","squirrel","rhinoceros","rabbit","bat","giraffe","wolf"
                ,"chihuahua","rat","weasel","otter","buffalo","zebra","giant+panda","deer","bobcat","pig","lion"
                ,"mouse","polar+bear","collie","walrus","raccoon","cow","dolphin"]

    #Define the classes we want to keep
    classTags=readSpacedData(pathClassAttr)
    classesIds=[1,2,3,4,5,6,7,8,9,10]
    classesNames=[allclassNames[i-1] for i in classesIds]
    #classesIds=[i for i in range(51)]

    #derive the instances we keep
    instIds=[]
    for i in range(N):
        if int(labels[i][0]) in classesIds:
            instIds.append(i)
    print("NUMBER OF INSTANCES : ",len(instIds))

    #extract features for animals in those classes
    featureSpace=readSubsetData(pathFeatures,instIds)
    print("NUMBER OF FEATURES : ",len(featureSpace),len(featureSpace[0]))

    #Extract tags for animals in those classes
    tagSpace=[]
    groundTruth=[]
    for i in instIds: #labels in classTags start at 1, not 0 ! so we put -1 
        #tagSpace.append(classTags[int(labels[i-1][0])])
        tagSpace.append([int(t) for t in classTags[int(labels[i][0])-1] ])
        groundTruth.append(int(labels[i][0]))

    return featureSpace,tagSpace,tagNames,groundTruth,classesNames

#--------- Treecut ---------

def readFinaData(featurefile,descrfile):
    print("Read  Fina's treecut data")
    groundTruth=[]
    fSpace=[]
    CT=[]
    with open(featurefile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            fSpace.append(row[1:])
    with open(descrfile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            groundTruth.append(int(row[0]))
            CT.append(row[1:])
    thresholds=[0.11533889477407566, 0.1203077098366005, 0.11511143406008541, 0.10949628381590712, 0.1104818764095347, 0.08652794627493725, 0.0859585635282215, 0.0896137509532199, 0.12247916373037748, 0.08693955076534798]
    for i in range(len(CT)):
        print(len(CT[i]),len(thresholds))
        CT[i]=[int(float(CT[i][j])>thresholds[j]) for j in range(len(CT[0]))]
        fSpace[i]=[float(fSpace[i][j]) for j in range(len(fSpace[0]))]
    return groundTruth,CT,fSpace


def readTreecutData(featurefile,groundTruthFile):
    '''Read the Treecut dataset.'''

    print("Read treecut data")
    groundTruth=[]
    fSpace=[]
    fNames=[]
    dSpace=[]
    #Read
    init=True
    with open(featurefile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            if(init):
                fNames=row
                init=False
            else:
                #CreateCT
                prevVal=None
                fSpaceRow=[]
                dSpaceRow=[]
                for v in row:
                    currentVal=float(v)
                    if(prevVal!=None):
                        diff=prevVal-currentVal
                        dSpaceRow.append(diff)
                    fSpaceRow.append(currentVal)
                    prevVal=currentVal

                fSpace.append(fSpaceRow)
                dSpace.append(dSpaceRow)

    with open(groundTruthFile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datareader:
            groundTruth.append(int(row[0]))

    #Discretize
    fDiscr,convertedValsNames=convertNumToBin(fSpace,fNames,2)
    print(fSpace[0])
    print('fDiscr: ',fDiscr[0])
    print(convertedValsNames)
    
    return groundTruth,fSpace,dSpace,fNames,fDiscr,convertedValsNames

#treecutpath="/home/mathieu/Documents/Travail/These/datasets/treecut/tree_cut_data.csv"
#treecutGTpath="/home/mathieu/Documents/Travail/These/datasets/treecut/ground_truth.csv"
#t=readTreecutData(treecutpath,treecutGTpath)
#print(t)


#NewCaledonia dataset
def prepareNCdataset(path):
    '''Prepare the NewCaledonida dataset used and given by Thibaut.'''
    rawdata=readDataFile(path,delimiter=',')
    attributeNames=rawdata.pop(0)
    N=len(rawdata)
    exercisePart=[]

    #print(attributeNames)

    featureSpace=[]
    tagSpace=[[] for i in range(N)]
    tagNames=[]

    numToBin=['NbLines','NbVars','NbParams','InstDepth','NbReturns','NbTokens']
    catToBin=['Exercise','Student']

    #feature space construction
    for i in range(N):
        featureSpace.append([float(v) for v in rawdata[i][0:20]]) #20 numerical features

    for featureInd in range(20,len(rawdata[0])):
        #Exercises: Ground Truth
        if(attributeNames[featureInd]=='Exercise'):
            exVals=[]
            for i in range(N):
                iVal=rawdata[i][featureInd]
                if (iVal not in exVals):
                    exVals.append(iVal)
            for i in range(N):
                iVal=rawdata[i][featureInd]
                for v in range(len(exVals)):
                    if(iVal==exVals[v]):
                        exercisePart.append(v)

        #Categorical
        if (attributeNames[featureInd] in catToBin):
            #print(attributeNames[featureInd],'categories conversion')
            catVals=[]
            for i in range(N):
                iVal=rawdata[i][featureInd]
                if (iVal not in catVals):
                    catVals.append(iVal)
            for i in range(N):
                iVal=rawdata[i][featureInd]
                for v in range(len(catVals)):
                    tagSpace[i].append(int(iVal==catVals[v]))#tagSpace.append(catVals.index(iVal))
            tagNames+=catVals

        #Numerical
        elif(attributeNames[featureInd] in numToBin):
            #discretization (median ?)
            #print(attributeNames[featureInd],'discetization')
            median=np.median([float(sublist[featureInd]) for sublist in rawdata])
            for i in range(N):
                tagSpace[i].append(int(float(rawdata[i][featureInd])<=median))
                tagSpace[i].append(int(float(rawdata[i][featureInd])>median))
            tagNames.append(attributeNames[featureInd]+"<="+str(median))
            tagNames.append(attributeNames[featureInd]+">"+str(median))

        #Boolean
        else:
            #print(attributeNames[featureInd],'boolean')
            tagNames.append(attributeNames[featureInd])
            for i in range(N):
                tagSpace[i].append(int(rawdata[i][featureInd]=='True'))
    
    return featureSpace,tagSpace,tagNames, exercisePart

#CSpath='/home/mathieu/Documents/Travail/These/datasets/Thibaut/NC1014_clustering.csv'
#prepareNCdataset(CSpath)

#TODO NewCaledonia dataset - specific exercise
def prepareNCexercice(path,specific_ex):
    '''Prepare a cluster from the NewCaledonida dataset used and given by Thibaut.'''
    rawdata=readDataFile(path,delimiter=',')
    attributeNames=rawdata.pop(0)
    N=len(rawdata)

    if(specific_ex!=None and specific_ex>=0 and specific_ex<8):
        print('Get only the points belonging to exercice',specific_ex)
    else:
        print('Error: unknown exercice number',specific_ex)

    featureSpace=[]
    tagNames=[]
    exVals=[]
    exercisePart=[]

    numToBin=['NbLines','NbVars','NbParams','InstDepth','NbReturns','NbTokens']
    catToBin=['Exercise','Student']

    for featureInd in range(20,len(rawdata[0])):
        #Exercises
        if(attributeNames[featureInd]=='Exercise'):
            for i in range(N):
                iVal=rawdata[i][featureInd]
                if (iVal not in exVals):
                    exVals.append(iVal)
            for i in range(N):
                iVal=rawdata[i][featureInd]
                for v in range(len(exVals)):
                    if(iVal==exVals[v]):
                        exercisePart.append(v)

    #feature space construction
    for i in range(N):
        if(specific_ex==exercisePart[i]):
            featureSpace.append([float(v) for v in rawdata[i][0:20]]) #20 numerical features

    N_ex=len(featureSpace)
    tagSpace=[[] for i in range(N)]

    for featureInd in range(20,len(rawdata[0])):
        #Categorical
        if (attributeNames[featureInd] in catToBin and attributeNames[featureInd]!='Exercise'):
            catVals=[]
            for i in range(N):
                if(specific_ex==exercisePart[i]):
                    iVal=rawdata[i][featureInd]
                    if (iVal not in catVals):
                        catVals.append(iVal)
            for i in range(N):
                if(specific_ex==exercisePart[i]):
                    iVal=rawdata[i][featureInd]
                    for v in range(len(catVals)):
                        tagSpace[i].append(int(iVal==catVals[v]))#tagSpace.append(catVals.index(iVal))
            tagNames+=catVals

        #Numerical
        elif(attributeNames[featureInd] in numToBin):
            #discretization (median ?)
            #print(attributeNames[featureInd],'discetization')
            median=np.median([float(sublist[featureInd]) for sublist in rawdata])
            for i in range(N):
                if(specific_ex==exercisePart[i]):
                    tagSpace[i].append(int(float(rawdata[i][featureInd])<=median))
                    tagSpace[i].append(int(float(rawdata[i][featureInd])>median))
            tagNames.append(attributeNames[featureInd]+"<="+str(median))
            tagNames.append(attributeNames[featureInd]+">"+str(median))

        #Boolean
        else:
            #print(attributeNames[featureInd],'boolean')
            tagNames.append(attributeNames[featureInd])
            for i in range(N):
                if(specific_ex==exercisePart[i]):
                    tagSpace[i].append(int(rawdata[i][featureInd]=='True'))
                    
    #N_ex=len(featureSpace)
    tagSpace=[tagSpace[i] for i in range(N) if specific_ex==exercisePart[i]]

    return featureSpace,tagSpace,tagNames,exercisePart

#CSpath='/home/mathieu/Documents/Travail/These/datasets/Thibaut/NC1014_clustering.csv'
#featureSpace,tagSpace,tagNames,exercisePart=prepareNCexercice(CSpath,specific_ex=0)
#print(tagNames)
#print(range(len(tagNames)))
#print(len(tagSpace),len(tagSpace[0]),len(tagNames))
#print(tagNames[66])

#---- 2D Artificial datasets ----
def artifDataGeneration(overlapOpt=0):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def extract_cluster_data(dataframe):
        """
        Prend un DataFrame d'un cluster et retourne deux listes de listes :
        - Une contenant les coordonnées (x, y) de tous les points.
        - Une contenant leurs autres propriétés (couleur, forme, taille, etc.).
        
        Args:
            dataframe (pd.DataFrame): DataFrame contenant les colonnes 'x', 'y', et les propriétés (e.g., 'color', 'shape', 'size').
            
        Returns:
            tuple: (coordinates_list, properties_list)
                - coordinates_list (list of list): Liste des coordonnées [(x1, y1), (x2, y2), ...].
                - properties_list (list of list): Liste des propriétés [[prop1, prop2, ...], [prop1, prop2, ...], ...].
        """
        # Liste des coordonnées (x, y)
        coordinates_list = dataframe[['x', 'y']].values.tolist()
        
        # Liste des autres propriétés
        properties_columns = [col for col in dataframe.columns if col not in ['x', 'y']]
        properties_list = dataframe[properties_columns].values.tolist()
        
        return coordinates_list, properties_list

    def genFormatedDataAttrMat(c1,c2,c3,ov=[],ovOpt=False):
        '''Returns feature Space, Description Space, gt attribution matrix and GT list of list where sublists are ids of points in corresp clusters.'''
        attrMat=[]
        clustIds=[]
        l=0
        fc1,dc1=extract_cluster_data(c1)
        attrMat+=[[1,0,0] for i in range(len(c1))]
        clustIds+=[[i for i in range(len(c1))]]
        l=len(clustIds[0])
        fc2,dc2=extract_cluster_data(c2)
        attrMat+=[[0,1,0] for i in range(len(c2))]
        clustIds+=[[l+i for i in range(len(c1))]]
        l+=len(clustIds[1])
        fc3,dc3=extract_cluster_data(c3)
        attrMat+=[[0,0,1] for i in range(len(c3))]
        clustIds+=[[l+i for i in range(len(c1))]]
        l+=len(clustIds[2])
        
        if(ovOpt!=False):
            fov,dov=extract_cluster_data(ov)
            attrMat+=[[1,0,1] for i in range(len(ov))]
            for i in range(len(ov)):
                clustIds[0].append(l)
                clustIds[2].append(l)
                l+=1

        else:
            fov=[]
            dov=[]

        ff=fc1+fc2+fc3+fov
        fd=dc1+dc2+dc3+dov
        return ff,fd,attrMat,clustIds

    # Fonction pour générer des propriétés selon des proportions
    def generate_properties(num_points, color_prop, shape_prop, size_prop):
        colors = np.random.choice(['blue', 'red', 'green'], size=num_points, p=color_prop)
        shapes = np.random.choice(['square', 'triangle', 'circle'], size=num_points, p=shape_prop)
        sizes = np.random.choice(['small', 'large'], size=num_points, p=size_prop)
        return colors, shapes, sizes

    # Fonction pour générer un cluster
    def generate_cluster(center, num_points, std, color_prop, shape_prop, size_prop):
        x = np.random.normal(center[0], std, num_points)
        y = np.random.normal(center[1], std, num_points)
        colors, shapes, sizes = generate_properties(num_points, color_prop, shape_prop, size_prop)
        return pd.DataFrame({'x': x, 'y': y, 'color': colors, 'shape': shapes, 'size': sizes})

    # Cluster 1
    cluster1 = generate_cluster(
        center=[2, 2], num_points=100, std=0.2,
        color_prop=[0.8, 0.2, 0.0], shape_prop=[0.6, 0.3, 0.1], size_prop=[0.3, 0.7]
    )
    # Cluster 1
    cluster15 = generate_cluster(
        center=[2, 2], num_points=100, std=0.5,
        color_prop=[0.8, 0.2, 0.0], shape_prop=[0.6, 0.3, 0.1], size_prop=[0.3, 0.7]
    )

    # Cluster 2
    cluster2 = generate_cluster(
        center=[5, 5], num_points=100, std=0.2,
        color_prop=[0.2, 0.7, 0.1], shape_prop=[0.3, 0.5, 0.2], size_prop=[0.6, 0.4]
    )

    # Cluster 3 (allongé, fusion de sous-clusters)
    centers = [[4.2, 2.2], [3.8, 2.1], [3.4, 2.1], [3.0, 2.1]]
    cluster3_parts = []
    for center in centers:
        cluster3_parts.append(generate_cluster(
            center=center, num_points=30, std=0.1,
            color_prop=[0.1, 0.1, 0.8], shape_prop=[0.2, 0.2, 0.6], size_prop=[0.3, 0.7]
        ))
    cluster3 = pd.concat(cluster3_parts, ignore_index=True)

    # Cluster 35 (allongé, fusion de sous-clusters)
    centers = [[4.2, 2.2], [3.8, 2.1], [3.4, 2.1], [3.0, 2.1]]
    cluster35_parts = []
    for center in centers:
        cluster35_parts.append(generate_cluster(
            center=center, num_points=30, std=0.3,
            color_prop=[0.1, 0.1, 0.8], shape_prop=[0.2, 0.2, 0.6], size_prop=[0.3, 0.7]
        ))
    cluster35 = pd.concat(cluster35_parts, ignore_index=True)

    # Cas 0 : pas chevauchement
    if(overlapOpt==0 or overlapOpt>4):
        data = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)

        ff,fd,attrMat,clustIds=genFormatedDataAttrMat(cluster1, cluster2, cluster3,[],ovOpt=False)

    else:
        # Cas 2 : Données avec chevauchement
        # Chevauchement
        if(overlapOpt==1):
            # Sous-cas 1 : chevauchement majoritairement bleu/carré
            overlap1 = generate_cluster(
                center=[2.6, 2.1], num_points=30, std=0.05,
                color_prop=[0.8, 0.1, 0.1], shape_prop=[0.8, 0.1, 0.1], size_prop=[0.3, 0.7]
            )
            data = pd.concat([cluster1, cluster2, cluster3, overlap1], ignore_index=True)
            ff,fd,attrMat,clustIds=genFormatedDataAttrMat(cluster1, cluster2, cluster3,overlap1,ovOpt=True)
        elif(overlapOpt==2):
            # Sous-cas 2 : chevauchement majoritairement vert/carré
            overlap2 = generate_cluster(
                center=[2.6, 2.1], num_points=30, std=0.05,
                color_prop=[0.1, 0.1, 0.8], shape_prop=[0.8, 0.1, 0.1], size_prop=[0.3, 0.7]
            )
            data = pd.concat([cluster1, cluster2, cluster3, overlap2], ignore_index=True)
            ff,fd,attrMat,clustIds=genFormatedDataAttrMat(cluster1, cluster2, cluster3,overlap2,ovOpt=True)
        elif(overlapOpt==3):
            # Sous-cas 3 : chevauchement totallement bleu/carré
            overlap3 = generate_cluster(
                center=[2.6, 2.1], num_points=50, std=0.06,
                color_prop=[1.0, 0, 0], shape_prop=[1, 0, 0], size_prop=[0.3, 0.7]
            )
            data = pd.concat([cluster1, cluster2, cluster3, overlap3], ignore_index=True)
            ff,fd,attrMat,clustIds=genFormatedDataAttrMat(cluster1, cluster2, cluster3,overlap3,ovOpt=True)
        elif(overlapOpt==4):
            # Sous-cas 4 : chevauchement totallement vert/carré
            overlap4 = generate_cluster(
                center=[2.6, 2.1], num_points=50, std=0.06,
                color_prop=[0, 0, 1], shape_prop=[1, 0, 0], size_prop=[0.3, 0.7]
            )
            data = pd.concat([cluster1, cluster2, cluster3, overlap4], ignore_index=True)
            ff,fd,attrMat,clustIds=genFormatedDataAttrMat(cluster15, cluster2, cluster35,overlap4,ovOpt=True)

    def convFD(fd):
        '''Create binary descr space and list of tags'''
        colors = ['blue', 'red', 'green']
        shapes = ['square', 'triangle', 'circle']
        sizes = ['small', 'large']
        descrspace=[]
        tagnames=colors+shapes+sizes
        for i in range(len(fd)):
            descrspace.append([int(tagnames[t] in fd[i]) for t in range(len(tagnames))])
        return descrspace,tagnames

    descrspace,tagnames=convFD(fd)

    # Calcul des supports pour les combinaisons de propriétés
    def calculate_support(data):
        combinations = data.groupby(['color', 'shape', 'size']).size().reset_index(name='support')
        return combinations

    # Calcul des supports
    supports_datascale = calculate_support(data)
    suppC1= calculate_support(cluster1)
    suppC2= calculate_support(cluster2)
    suppC3= calculate_support(cluster3)
    if(overlapOpt==1):
        sppOv=calculate_support(overlap1)
    if(overlapOpt==2):
        sppOv=calculate_support(overlap2)
    print()

    return ff,descrspace,tagnames,attrMat

artifDataGeneration(overlapOpt=0)