import csv
import sklearn.utils
import sklearn.datasets
import random
from ucimlrepo import fetch_ucirepo
from dataTreatment import *
import numpy as np

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

#--------- NewCaledonia dataset -----------
def prepareNCdataset(path):
    '''Prepare the NewCaledonida dataset used and given by Thibaut Martinet.'''
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


#TODO NewCaledonia dataset - specific exercise
def prepareNCexercice(path,specific_ex):
    '''Prepare a cluster from the NewCaledonida dataset used and given by Thibaut Martinet.'''
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