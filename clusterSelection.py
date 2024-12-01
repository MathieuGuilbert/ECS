import numpy as np
#from sklearn import metrics

#from sklearn.metrics import pairwise_distances
from sklearn import datasets

from skmine.itemsets import LCM

import time

from particularDataTreatment import *
from dataTreatment import *
from basePartitions import *
from clusterQuality import *
from constraints import verifyMustLink,verifyCannotLink
from tests import *

#notes:
#For our model, clusters have to be represented by a list of list. Each sublist represents one of the cluster elements, in the form of a list of their covering tags.

#---------------------------

def clusterElem(Z,T,K):
	'''
	PARAMETERS
	--------
	Z = partition where Zik=1 signify that instance i is in cluster k
	K = number of clusters
	T = tag space.

	RETURNS
	--------
	returns a list of tuple coords.
	First element of the tuples is an array of instances belonging to the cluster, the second element is the array with all other elements.
	List of List ids contains the ids of the instances belonging to each cluster in separated sublists.
	Instances are represented in thoses array by a list of their tags.
	'''
	coords=[]
	ids=[[] for i in range(len(Z))]
	for i in range(len(Z)):
		ids[Z[i]].append(i)

	for i in range(0,K):
		mask = Z == i
		i_cluster = T[mask]
		mask = Z != i
		ineg_cluster = T[mask]
		coords.append((i_cluster,ineg_cluster))
	return coords,ids

#--------------------------

#TODO: add multiple possibilities (and possibility to use multiple measures combined ?)
def evaluateCluster(C):
	'''Evaluate the quality of a cluster C according to a given measure.'''
	return WCSS(C)

def clusterSelection(ClustIds:list,minClustQuality:int,minClustSize:int,MLs:list,CLs:list):
	'''Select among the clustIds list the clusters satisfying the specified requierments.

	PARAMETERS
	--------
	ClustIds: list of clusters, which are themselves lists of instance ids.
	minClustQuality: minimal threshold for a cluster to be considered "good" according to the evaluation criteria.
	minClustSize: minimal number of instances requiered in all selected clusters
	MLs: list of Must-Link constraints
	CLs: list of Cannot-Link constraints'''
	selectedClustersIds=[]
	for i in range(len(ClustIds)):
		Cid=ClustIds[i]
		clustVerifiesConstr=True
		#Constraint Satisfaction
		for (a,b) in MLs:
			if not verifyMustLink(a,b,Cid):
				clustVerifiesConstr=False
				break
		for (a,b) in CLs:
			if not verifyCannotLink(a,b,Cid):
				clustVerifiesConstr=False
				break

		if(len(Cid)>=minClustSize and clustVerifiesConstr):
			val=1000 #TODO: modify evaluateCluster(C)
			if(val>=minClustQuality):
				selectedClustersIds.append(ClustIds[i])

	return selectedClustersIds

#---------------------------

def generatePatterns(C:list,minSupp:int,patType:str,allowInclusion:bool=True):
	'''Generate the patterns covering a cluster.

	PARAMETERS
	--------
	C: cluster (list of each instances pattern id)
	minSupp: minimum support on the cluster for a pattern to be considered covering and thus keep it as descriptor.
	patType = "tag" or "pat", depending on if we want the descriptor to be LCM-generated patterns or pattern containing only one tag.

	RETURNS
	--------
	array of tuple (couples), each containing a list (ids of the tags in the pattern) and an int (support)
	'''
	if patType=="tag":
		s1Patterns=generateSize1Patterns(C,minSupp)
		return s1Patterns
	else:
		lcmPatterns=launchLCM(C,minSupp,allowInclusion=allowInclusion)
		return lcmPatterns

def launchLCM(data:list,supp:int,allowInclusion:bool=True):
	'''Launches the LCM algorithm on the given dataset, with a certain parameter of minimum support.

	PARAMETERS
	--------
	data: list of N lists of integer values representing the ids of activated tags for each instance.
	supp: minimum support on the cluster for a pattern to be considered covering and thus keep it as descriptor.
	allowInclusion: indicates if patterns being subparts of other patterns are allowed. If not, a filtering step is needed.
	'''
	lcm = LCM(min_supp=supp)
	patterns = lcm.fit_discover(data) #patterns type = pd.dataframe
	if(allowInclusion):
		patternsValues=patterns.values #type : np.array
	else:
		#-filter out pattern inclusion
		#print('DEBUG no inclusion allowed')
		patterns_filtered=filterLCMinclusions(patterns)
		patternsValues=patterns_filtered.values
	return patternsValues

def filterLCMinclusions(patternsDF:pd.DataFrame):
	'''Filter out LCM patterns that are redundant, i.e. those that are included in others.'''

	# Convert itemsets to sets for efficient comparison
	patternsDF['itemset'] = patternsDF['itemset'].apply(lambda x: set(x))

	# Sort DataFrame by the length of the itemsets (shortest first)
	patternsDF['itemset_len'] = patternsDF['itemset'].apply(len)
	patternsDF.sort_values('itemset_len', inplace=True)

	# Use a boolean array to track which patterns have been removed
	removed = np.zeros(len(patternsDF), dtype=bool)

	# Iterate over all pairs of patterns to check for subset relationships
	for i in range(len(patternsDF)):
		if removed[i]:
			continue
		pattern1 = patternsDF.iloc[i]['itemset']
		for j in range(i + 1, len(patternsDF)):
			if removed[j]:
				continue
			pattern2 =patternsDF.iloc[j]['itemset']
			if pattern1.issubset(pattern2):
				removed[i] = True
				break

	# Drop the redundant patterns
	df_filtered = patternsDF[~removed].reset_index(drop=True)

	# Convert itemsets back to lists
	df_filtered['itemset'] = df_filtered['itemset'].apply(lambda x: list(x))

	# Drop the temporary itemset_len column
	df_filtered.drop(columns=['itemset_len'], inplace=True)

	#print('DEBUG df fitered',df_filtered)
	return df_filtered


def generateSize1Patterns(C:list,minSupp:int):
	'''Search patterns individually covering the cluster C.'''
	treated=[]
	res=[]
	for inst in C:
		for t in inst:
			if t not in treated:
				cpt=0
				for i2 in C:
					if t in i2:
						cpt+=1
				if(cpt>=minSupp):
					res.append([[t],cpt])
				treated.append(t)
	return res

def selectPatterns(lenC,allPat,tagDF,patternDic,patternQualityThreshold,minCovPer,maxCoveredOutOfClusterOption):
	'''Selection and computation of a dataframe of activation stats of each pattern on all instances.

	lenC=size of cluster C'''
	selPat=[]
	minSupport=int(minCovPer*lenC/100) #convert percentage in concrete number #int(20*lenC/100)
	for p in allPat:
		status=evalPattern(p,tagDF,minSupport,patternDic,maxCoveredOutOfClusterOption)
		if(status>=patternQualityThreshold):
			selPat.append(p)
	return selPat


def evalPattern(p,tagDF,minSupport,patternDic,maxCoveredOutOfCluster):
	'''Evaluate the validity of a pattern and add its activity in the pattern Dataframe.'''
	(e,support)=p
	if(support<=minSupport):
		return 0

	#Discriminativeness: Check the number of instances outside of the cluster that are covered by p
	if str(e) not in patternDic:
		patAct=genPatternColumn(e,tagDF)
		inDic=False
	else:
		patAct=patternDic[str(e)]
		inDic=True
	nbrNegativeCov=sum(patAct)-support

	if nbrNegativeCov>maxCoveredOutOfCluster:
		print("PATTERN NOT DISCRIMINATIVE DATASET WISE, ",nbrNegativeCov," instead of maximum ",maxCoveredOutOfCluster," !")
		return 0
	else:
		if(inDic==False): ##add the pattern to the pattern dict if it is not already
			patternDic[str(e)]=patAct
	return 1

def genPatternColumn(p,tagDF):
	'''p: a pattern, i.e. a list of tags ids
	tafDF: a dataframe displaying the reactivity of each instance on each tag
	returns values 0 or 1 depending on if the pattern is active on the object or not.'''
	s=sum([ tagDF[j] for j in p ])
	res=(s-s%len(p))/len(p)
	return res

#with disjonction pattern disP
#and tafDF a dataframe displaying the reactivity of each instance on each tag
#NOT APPLIED
def genDisjPatternDFColumn(disP,tagDF):
	subPats=[]
	(e,s)=disP
	for p in e:
		subPats.append(genPatternColumn(p,tagDF))
	res=sum(subPats)
	for i in range(len(res)):
		if res[i]>=1:
			res[i]=1
		else:
			res[i]=0
	print(res)
	print(sum(res))
	return res

def genClustPatternDFColumn(clustIds,instPatDF):
	'''Generate cluster per pattern dataframe.'''
	patClustDF = pd.DataFrame()
	patClustDic={}
	for p in instPatDF:
		patClustDic[p]=[ sum([instPatDF[p][i] for i in Cid]) for Cid in clustIds ]
	patClustDF=pd.DataFrame(patClustDic)
	return patClustDF.transpose(),patClustDF #.transpose to inverse and have the format clusterPatDF[cluster][pattern]

def assembleDesc(selPat:list):
	'''Create the description of the cluster from selected patterns in list selPat.'''
	descr=[]
	for p in selPat:
		descr.append(p)
	return descr

#TODO remove Quality Threshold ?
def generateClusterDescriptionDF(C:list,tagDF:pd.DataFrame,patternDic:dict,patternQualityThreshold:float,minCovPer:float,
				 maxCoveredOutOfClusterOption:float,patType:str,allowInclusion:bool=True):
	'''Complete process to create the description of a cluster C.

	Parameters
	--------
	C: cluster (list of each instances pattern id)
	tagDF: tag Dataframe
	patternDF: pattern Dataframe
	minCovPer: minimal coverage percentage for a pattern to be accepted as a candidate descriptor.
	maxCoveredOutOfClusterOption : discriminativeness Cluster VS All threshold, beyond which a pattern is not considered discriminative.
	patType: "tag" or "pat"
	allowInclusion: indicates if LCM patterns being subparts of other patterns are allowed. If not, a filtering step is needed.

	Returns
	--------
	descr: the description of the entry cluster.
	'''
	minSupp=max(int(minCovPer*len(C)/100),1) #convert percentage in concrete number
	patterns=generatePatterns(C,minSupp,patType,allowInclusion=allowInclusion)
	selectedPatterns=selectPatterns(len(C),patterns,tagDF,patternDic,patternQualityThreshold,minCovPer,maxCoveredOutOfClusterOption)
	descr=assembleDesc(selectedPatterns)
	return descr

#------------------------------------

def generateIdPatterns(patClustDF,descr):
	'''Generate the Ids of Patterns.'''
	res=patClustDF.columns.tolist()
	patClustDF.columns=[i for i in range(len(patClustDF.columns))]

	#convert the descriptions in list of ints
	allIntDescr=[]
	for d in descr:
		intDescr=[]
		for (e,s) in d:
			intDescr.append(res.index(str(e)))
		allIntDescr.append(intDescr)
	return res,allIntDescr

def instanceClusterMatrix(N:int,Cids:list):
	'''Return a matrix where M[i][c] == 1 if instance i is in cluster c and 0 otherwise.'''
	res=np.array([[np.multiply((i in Cid), 1) for i in range(N)] for Cid in Cids])
	res=res.transpose()
	return res

def verifyAttrib(minAttrib:int,N:int,listClustersIds:list):
	'''Find all instances that are attributed to a number of clusters inferior a parameter minAttrib.

	#minAttrib : mimimum number of clusters attributed to each instances
	#N: number of instances
	#listClustersIds : list of lists, each sublist corresponding to a particular cluster. They contain ids of instances.'''
	notSufficientlyAttributedInstances=[]
	for i in range(0,N):
		cpt=0
		for clust in listClustersIds:
			if i in clust:
				cpt+=1
			if cpt>=minAttrib:
				break
		if cpt<minAttrib:
			notSufficientlyAttributedInstances.append(i)
	#Send Warning
	if(len(notSufficientlyAttributedInstances)>0):
		print("WARNING : ",len(notSufficientlyAttributedInstances)," are attributed to less than ",minAttrib," cluster.s")
	return notSufficientlyAttributedInstances

#-- Overall BC generation --

def genBC(featureSpace,kmin,kmax,simMat,distMat,baseAlgorithms,repeatBPgen,maxClustSize):
	'''Overall Base Clusters generation

	RETURNS
	--------
	baseClusterIds: base clusters, in the form of a list of list where each sublist represents a cluster and contains the ids of instances.
	basePartitions: base partitions, in the form of a list of list of list. Each sublist represents a partitions, itself containg subsublist representing its clusters.
	genBasePart_execution_time: execution time of the all Base clusters generation step.
	'''
	print("Generate Base partitions : ")
	start_time_genBasePart=time.time()
	baseClustersIds=[]
	basePartitions=[]
	for b in range(repeatBPgen):
		if("Kmeans" in baseAlgorithms):
			KmeansClust,KmeansPart=genBaseKmeans(np.array(featureSpace),kmin,kmax,maxClustSize)
			baseClustersIds=baseClustersIds+KmeansClust #Kmeans
			basePartitions=basePartitions+KmeansPart
		if("Spectral" in baseAlgorithms):
			if simMat!=[]:
				SpectralRes,SpectralPart=genSpectralPrecomp(simMat,kmin,kmax) #needs similarity matrix
				baseClustersIds=baseClustersIds+SpectralRes
				basePartitions=basePartitions+SpectralPart
		if("SpectralNN" in baseAlgorithms):
			SpectralRes,SpectralPart=genSpectralNN(featureSpace,kmin,kmax)
			baseClustersIds=baseClustersIds+SpectralRes
			basePartitions=basePartitions+SpectralPart
		if("Hierarchical" in baseAlgorithms):
			if distMat!=[]:
				HierarchicalRes,HierarchicalPart=genBaseAgglo(distMat,kmin,kmax) #needs distance matrix
				baseClustersIds=baseClustersIds+HierarchicalRes
				basePartitions=basePartitions+HierarchicalPart
	genBasePart_execution_time=round(time.time()-start_time_genBasePart,2)
	return baseClustersIds,basePartitions,genBasePart_execution_time

def genDistAndSim(featureSpace,precomputedSim,precomDist,baseAlgorithms):
	'''Generate Distance and/or Similarity matrices if needed.'''
	distMat=precomDist
	simMat=precomputedSim
	if("Hierarchical" in baseAlgorithms or "DBSCAN" in baseAlgorithms): #needs distance matrix
		if(distMat==[]):
			print("Def sim matrix")
			distMat=defineMatrixEuclidean(featureSpace)
	if("Spectral" in baseAlgorithms): #needs similarity matrix
		if simMat==[]:
			print("Gen sim matrix")
			if(distMat==[]):
				distMat=defineMatrixCosine(featureSpace) #defineMatrixEuclidean(featureSpace)
			simMat=convertDistInSim(distMat)
	return distMat,simMat

def launchClusterSelection(baseClustersIds):
	'''First cluster filtering/selection.
	(In pipeline applied in our current experiments, not really considered)'''
	minClustQuality=1
	minClustSize=1
	start_time_clustQualitySelection=time.time()
	selectedClustersIds=clusterSelection(baseClustersIds,minClustQuality,minClustSize,[],[])
	clusterQualitySelection_execution_time=time.time()-start_time_clustQualitySelection

	print("Number of clusters BEFORE filtering : ",len(baseClustersIds))
	print("Number of clusters AFTER filtering : ",len(selectedClustersIds))
	return selectedClustersIds,round(clusterQualitySelection_execution_time,2)

def displayResults(N,V,nonEmptyClustersIds,baseClustersIds,selectedClustersIds,selectedClustersSize,emptyDescr,emptyDescrClustLen,descrSize,positiveDescrSize,clustPatDF,genBP_time,descr_time,clustPatDF_time):
	'''Print in the terminal differents results and statistics.'''
	print()
	print("-------------------")
	print("Generation and selection results :")
	print()
	#print("Number of clusters BEFORE first filtering : ",len(baseClustersIds))
	#print("Number of clusters AFTER first filtering  : ",len(selectedClustersIds))
	print("Number of cluster having empty descriptions : ",emptyDescr,"They are of size : ",emptyDescrClustLen)
	print("Number of cluster NOT having empty descriptions : ",len(nonEmptyClustersIds))
	print("Selected cluster Size stats: min : ",min(selectedClustersSize)," ; median : ",np.median(selectedClustersSize)," ; mean : ",np.mean(selectedClustersSize)," ; max : ",max(selectedClustersSize))
	print("Description Size stats: min : ",min(descrSize)," ; median : ",np.median(descrSize)," ; mean : ",np.mean(descrSize)," ; max : ",max(descrSize))
	if(len(positiveDescrSize)!=0):
		print("Positive Descrs stats: min : ",min(positiveDescrSize)," ; median : ",np.median(positiveDescrSize)," ; mean : ",np.mean(positiveDescrSize)," ; max : ",max(positiveDescrSize))

	#print("Verif attrib after description selection:")
	notSufficientlyAttributedInstancesAfterDescrSelection=verifyAttrib(1,N,nonEmptyClustersIds)

	print("Times :")
	print("Base partition generation : ",genBP_time)
	print("Generation of the descriptions : ",descr_time)
	print("Generation of clustPat Dataframe : ",clustPatDF_time)
	print()

	print("-- Launch CP model --")
	print("Entry: ",V," clusters, having ",len(notSufficientlyAttributedInstancesAfterDescrSelection)," instances not attributed.")
	print("Entry: ",clustPatDF.shape," individual patterns.")
	print()

	return notSufficientlyAttributedInstancesAfterDescrSelection

def genDescrStep(tagSpace,tagSpaceDF,selectedClustersIds,minCovPer,maxCoveredOutOfClusterOption,patType,allowInclusion:bool=True):
	'''#Step where the description of each cluster is built.

	It also generates a DataFrame displaying coverage status between each pattern and instances.

	If patType="tag" then patterns will be constituted of only 1 tag of any size. Else, pattern are conjunction of tags of size 3.'''
	emptyDescr=0
	emptyDescrClustLen=[]
	descrSize=[]
	positiveDescrSize=[]
	selectedClustersDescr=[]
	selectedClustersNonEmptyDescr=[]
	selectedClustersSize=[]
	nonEmptyClustersIds=[]
	patternDic={}
	start_time_description=time.time()
	print("- Start Generation of Descriptions step. -")
	for c in range(0,len(selectedClustersIds)):
		if(len(selectedClustersIds)<2):
			return None
		tagsInC=getTagsIdsForClusterInstances(tagSpace,selectedClustersIds[c])
		#print(tagsInC)
		r=generateClusterDescriptionDF(tagsInC,tagSpaceDF,patternDic,1,minCovPer,maxCoveredOutOfClusterOption,patType,allowInclusion=allowInclusion)
		#print("Number of cluster",c," patterns : ",len(r))
		descrSize.append(len(r))
		selectedClustersDescr.append(r)

		if(len(r)==0):
			emptyDescr+=1
			emptyDescrClustLen.append(len(tagsInC))
		else:
			#print(c,len(r))
			positiveDescrSize.append(len(r))
			selectedClustersNonEmptyDescr.append(r)
			nonEmptyClustersIds.append(selectedClustersIds[c])
			selectedClustersSize.append(len(selectedClustersIds[c]))

	clusterDescriptionGeneration_execution_time=time.time()-start_time_description
	patternDF=pd.DataFrame(patternDic)
	return nonEmptyClustersIds,patternDF,selectedClustersSize,emptyDescr,emptyDescrClustLen,descrSize,positiveDescrSize,selectedClustersNonEmptyDescr,round(clusterDescriptionGeneration_execution_time,2)

#----------- MAIN ------------------

def main(baseAlgorithms:list,featureSpace:list,tagSpace:list,repeatBPgen:int,maxClustSize:int,patType:str,minCovPer,maxCoveredOutOfClusterOption,kmin:int,kmax:int,precomputedSim=[],precompDist=[],precompBC=[],enforcedClusters=[],allowInclusion:bool=True):
	'''Function handling the base partition generation, cluster selection and description generation steps.

	PARAMETERS
	--------
	baseAlgorithms: list of clustering algorithms names
	featureSpace : feature space, on which the base clusters will be computed
	tagSpace : tag/descriptor space, with which the cluster descriptions will be created
	repeatBPgen : number of repetition of the generation process
	maxClustSize : maximal cluster size
	patType : pattern type, either "tag" or "pat"
	minCovPer : minimal percentage of instances of a cluster a pattern has to cover to be considered in the descriptions
	maxCoveredOutOfClusterOption : discriminativeness Cluster VS all requierement.
	kmin: smallest value of k to apply in the base partition generation step.
	kmax: max value of k to apply in the base partition generation step.
	#precomputedSim : optional, precomputed Similarity matrix.
	precompDist : optional, precomputed Distance matrix.
	precompBC : optional, precomputed Base clusters. If different than [], we do not generate novel cluster.
	enforcedClusters : optional, list of lists each representing a cluster to add to the pool of candidates.
	allowInclusion: optional, default=True. Indicates if returned LCM patterns are allowed to be subsets of each other.

	RETURNS
	--------
	baseClustersIds: Base partitions' clusters
	N: number of instances
	V: number of candidate clusters
	nonEmptyClustersIds: V candidate clusters
	instanceClusterMat: matrix representing the cluster attributions of each instance
	allIntDescr: candidate descriptions
	clustPatDF: Dataframe between clusters and patterns
	patternDF: dataframe between instances and patterns
	listPat: list of all patterns
	selectedClustersSize: size of all the V candidate clusters
	times: list with the execution times of each steps of this function'''

	N=len(featureSpace)
	times=[] # a list that will contain diverse execution times
	#print('start create tag DataFrame')
	tagSpaceDF=createDF(tagSpace)
	#print('end createDF')

	#Evaluate if we need to generate the base partitions
	basePartitions=[]
	if(precompBC==[]):
		#Generate Distance and/or Similarity matrices if needed
		distMat,simMat=genDistAndSim(featureSpace,precomputedSim,precompDist,baseAlgorithms)

		#Generate base partitions and clusters
		baseClustersIds,basePartitions,genBP_time=genBC(featureSpace,kmin,kmax,simMat,distMat,baseAlgorithms,repeatBPgen,maxClustSize)
		times.append(genBP_time)
	else:
		distMat,simMat=genDistAndSim(featureSpace,precomputedSim,precompDist,baseAlgorithms)
		#Use precomputed base clusters
		print("PRECOMPUTED BASE CLUSTERS.")
		baseClustersIds=precompBC
		genBP_time=0.0
		times.append(genBP_time)

	#Cluster selection
	selectedClustersIds,clusterQualitySelection_execution_time=launchClusterSelection(baseClustersIds)
	times.append(clusterQualitySelection_execution_time)

	#TODO enforcedClusters (how to deal when description is empty ?)
	if(enforcedClusters!=[]):
		enforcedInd=[]
		for ec in enforcedClusters:
			if(ec not in selectedClustersIds): #account for cases where cluster is already in the pool of candidates
				print('DEBUG: add enforced cluster to candidate pool:',ec)
				enforcedInd.append(len(selectedClustersIds))
				selectedClustersIds.append(ec)
			else:
				print('DEBUG: Enforced cluster',ec,'was already in the candidate pool.')
				enforcedInd.append(selectedClustersIds.index(ec))

	#Generate descriptions
	descrRes=genDescrStep(tagSpace,tagSpaceDF,selectedClustersIds,minCovPer,maxCoveredOutOfClusterOption,patType,allowInclusion=allowInclusion)
	if(descrRes!=None):
		nonEmptyClustersIds,patternDF,selectedClustersSize,emptyDescr,emptyDescrClustLen,descrSize,positiveDescrSize,selectedClustersNonEmptyDescr,descr_time=descrRes
	else:
		print("ERROR: No cluster with satisfying descriptions where found")
		return None
	times.append(descr_time)

	#TODO enforcedClusters (how to deal when description is empty ?)
	endEnforcedInd=[]
	if(enforcedClusters!=[]):
		for u in range(len(enforcedClusters)):
			ec=enforcedClusters[u]
			ei=enforcedInd[u]
			if(ec not in nonEmptyClustersIds):# in emptyDescr):
				print('DEBUG ERROR: enforced cluster has no covering descriptor;',descrSize[ei])
			else:
				#print('DEBUG nonEmptyClustersIds',nonEmptyClustersIds)
				print('DEBUG: enforced',u,'cluster has',descrSize[ei],'covering descriptors.')
				endEnforcedInd.append(nonEmptyClustersIds.index(ec))
		#for ei in enforcedInd:
		#	print('DEBUG ei descrSize',descrSize[ei])
		#	if(descrSize[ei]==0):# in emptyDescr):
		#		print('DEBUG ERROR: enforced cluster has no covering descriptor.')
		#	else:
		#		print('DEBUG nonEmptyClustersIds',selectedClustersNonEmptyDescr)
		#		endEnforcedInd.append(selectedClustersNonEmptyDescr.index(ei))

	#print('DEBUG nonEmptyClustersIds',nonEmptyClustersIds)
	#print('DEBUG endEnforcedInd:',endEnforcedInd)

	#- Prepare Cluster/Pattern dataframe
	start_time_clustPatDF=time.time()
	clustPatDF,patClustDF=genClustPatternDFColumn(nonEmptyClustersIds,patternDF)
	clustPatDF_execution_time=time.time()-start_time_clustPatDF
	clustPatDF_time=round(clustPatDF_execution_time,2)
	times.append(clustPatDF_time)

	#Prepare other data
	listPat,allIntDescr=generateIdPatterns(patClustDF,selectedClustersNonEmptyDescr)
	clustPatDF=clustPatDF.astype('int')
	instanceClusterMat=instanceClusterMatrix(N,nonEmptyClustersIds)
	V=len(nonEmptyClustersIds)

	#Display results
	#print('selectedClustersSize:',selectedClustersSize)
	notAttrInst=displayResults(N,V,nonEmptyClustersIds,baseClustersIds,selectedClustersIds,selectedClustersSize,emptyDescr,emptyDescrClustLen,descrSize,positiveDescrSize,clustPatDF,genBP_time,descr_time,clustPatDF_time)

	return basePartitions,baseClustersIds,N,V,nonEmptyClustersIds,instanceClusterMat,allIntDescr,clustPatDF,patternDF,listPat,selectedClustersSize,times,endEnforcedInd
