
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from collections import Counter

#def get_class_paths_with_coverage(tree, feature_names, class_names):
def get_class_paths_with_coverage(tree, feature_names, class_names, total_samples_per_class,verbose=False):
    """
    Gathers all paths leading to leaves for each class and calculates their percentage coverage.

    Args:
        tree: Trained DecisionTreeClassifier or DecisionTreeRegressor.
        feature_names: List of feature names used in the tree.
        class_names: List of class names (for classification).
        total_samples_per_class: A dictionary with total sample counts for each class.

    Returns:
        A dictionary where keys are class names and values are lists:
        - The first list contains paths leading to the leaves.
        - The second list contains the percentage coverage (percentage of class points each path covers).
    """
    from sklearn.tree import _tree

    tree_ = tree.tree_
    paths = {class_name: {"paths": [], "coverage": [],"discr" : [],"leafsize" : []} for class_name in class_names}

    def traverse(node_id, path):
        # If it's a leaf node
        if tree_.children_left[node_id] == _tree.TREE_LEAF:
            leaf_class = tree_.value[node_id].argmax()# Get the class index with the majority count in the leaf
            leaf_dts_coverage = int(tree_.n_node_samples[node_id])  # Total number of samples in this leaf
            erros=(sum([tree_.value[node_id][0][i] for i in range(len(tree_.value[node_id][0])) if i!=leaf_class])) #Number of errors in the leaf
            total_class_samples = total_samples_per_class[class_names[leaf_class]]
            coverage_percentage = ((leaf_dts_coverage-erros) / total_class_samples) * 100  # Percentage coverage of the CLASS
            #print([(i,tree_.value[node_id][0][i]) for i in range(len(tree_.value[node_id][0]))])
            #print(leaf_coverage, total_class_samples,coverage_percentage,erros)
            paths[class_names[leaf_class]]["paths"].append(path)
            paths[class_names[leaf_class]]["leafsize"].append(leaf_dts_coverage)  # Rounded percentage
            paths[class_names[leaf_class]]["coverage"].append(round(coverage_percentage, 2))  # Rounded percentage

            pathdiscr=0
            for v in range(len(tree_.value[node_id][0])):
                if (v!=leaf_class):
                    patCovInV=tree_.value[node_id][0][v]
                    if(patCovInV>0):
                        if(verbose):
                            print("Element from class",v,"in leaf",node_id,"of class",leaf_class,":",tree_.value[node_id])
                            #print((patCovInV/total_samples_per_class[class_names[v]]),1-(patCovInV/total_samples_per_class[class_names[v]]))
                        pathdiscr+=(1-(patCovInV/total_samples_per_class[class_names[v]]))#*100
                        #print(patCovInV,"/",total_samples_per_class[class_names[v]])
                    else:
                        pathdiscr+=1
            if(pathdiscr!=0):
                #print(pathdiscr,"/",len(tree_.value[node_id][0])-1)
                pathdiscr=pathdiscr/(len(tree_.value[node_id][0])-1) #Average path/pattern IPC
            else:
                pathdiscr=1
            #print("Path discr:",pathdiscr)
            paths[class_names[leaf_class]]["discr"].append(round(pathdiscr, 2))  # Rounded percentage
        else:
            # Get the splitting feature and threshold
            feature = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]

            # Add the split rule and traverse both branches
            left_path = path + [f"not {feature}"]
            traverse(tree_.children_left[node_id], left_path)

            right_path = path + [f"{feature}"]
            traverse(tree_.children_right[node_id], right_path)

    traverse(0, [])  # Start traversal from the root node
    return paths


def postHocDecisionTree(Clustering,BoolSpace,featureNames,K,max_depth=None,max_leaf_nodes=None,ccp_alpha=0,vizTree=False,verbose=False,pathVerbose=False):
    # Initialize and fit the Decision Tree
    clf = DecisionTreeClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,ccp_alpha=ccp_alpha)
    clf.fit(BoolSpace,Clustering)

    plt.figure()
    plt.clf()
    class_names=["C"+str(c) for c in range(K)]
    plot_tree(clf, filled=True, feature_names=featureNames, class_names=class_names)
    if(vizTree):
        plt.show()
    
    # Calculate total samples per class
    total_samples_per_class = Counter(["C" + str(label) for label in Clustering])

    # Get paths with coverage as percentages
    paths = get_class_paths_with_coverage(clf, featureNames, class_names, total_samples_per_class,verbose=verbose)

    allPaths=[]
    allCov=[]
    AllDiscr=[]
    for class_name, class_data in paths.items():
        classPaths=[]
        for path in class_data["paths"]:
            classPaths.append(path)
        if(verbose or pathVerbose):
            print(f"Paths to leaves for {class_name} ({total_samples_per_class[class_name]} points):")
            for path in class_data["paths"]:
                print(f"- {', '.join(path)}")
        if(verbose):
            print(f"Leaf Size: {class_data['leafsize']}")
            print(f"Coverage: {class_data['coverage']}")
            print(f"Discrimination: {class_data['discr']}")
        allCov.append(round(np.mean([i/100 for i in class_data['coverage']]),2))
        AllDiscr.append(round(np.mean(class_data['discr']),2))
        allPaths.append(classPaths)
        #print(f"Discrimination: {class_data['discrimination']}")
    #print(f"Average Discrimination for {class_name}: {class_data['average_discrimination']}")
    
    # Predict the labels
    y_pred = clf.predict(BoolSpace)

    # Calculate the number of errors (misclassified samples)
    errors = (Clustering != y_pred).sum()
    print('Total number of errors: ',errors)

    errorPerClust=[0 for i in range(len(total_samples_per_class))]
    for i in range(len(y_pred)):
        if(Clustering[i]!=y_pred[i]):
            errorPerClust[Clustering[i]]+=1
    print(total_samples_per_class)
    ecPerClust=[round((1-errorPerClust[i]/total_samples_per_class[class_names[i]]),2) for i in range(len(total_samples_per_class))]
    print('Error per cluster: ',errorPerClust)
    print('EC per cluster: ',ecPerClust)
    print('mean PCR per cluster: ',allCov)
    print('mean IPC per cluster: ',AllDiscr)
    print('Number of pattern:',[len(subl) for subl in allPaths])
    print('size of pattern:',[[len(subsubl) for subsubl in subl] for subl in allPaths])

    return allPaths,allCov,AllDiscr,ecPerClust,errors,errorPerClust


#Clusters results from resp. the artificial dataset and the Automobile dataset
allClust=[[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 77, 84, 85, 86, 87, 88, 89, 91, 92, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 140, 141, 142, 143, 144, 145, 146, 147], [0, 1, 2, 4, 5, 6, 7, 18, 30, 43, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 90, 93, 94, 95, 96, 131, 137, 138, 139, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158], [3,32, 44, 45, 46, 47, 48]] #res from git with 1 unattr; 3, which I added
allClustA=[[200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319], [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]

# Find the maximum index
max_index = max(max(cluster) for cluster in allClust)
max_indexA = max(max(cluster) for cluster in allClustA)

# Initialize the list
L = [-1] * (max_index + 1)
LA = [-1] * (max_indexA + 1)

# Assign the cluster index to each element
for cluster_index, cluster in enumerate(allClust):
    for point in cluster:
        L[point] = cluster_index

for cluster_index, cluster in enumerate(allClustA):
    for point in cluster:
        LA[point] = cluster_index

#----- Post hoc decision tree on Automobile -----
autoPath="datasets/imports-85.data"
from particularDataTreatment import prepareAutomobileData,readAutomobileData
numVals,binVals,binValsNames,prices,convertedVals,convertedValsNames,priceDistances,wholeDescriptorSpace,wholeDescriptorSpaceNames=prepareAutomobileData(readAutomobileData(autoPath,False)) #AUTOMOBILE data
featureSpace=prices
featureSpace=[[f,1] for f in featureSpace]
tagSpace=wholeDescriptorSpace
tagNames=wholeDescriptorSpaceNames

print('--- AUTOMOBILE ---')
print('-- DEFAULT --')
postHocDecisionTree(L,tagSpace,tagNames,3,pathVerbose=True)
print()
print('-- ccp_alpha=0.01 --')
postHocDecisionTree(L,tagSpace,tagNames,3,ccp_alpha=0.01,pathVerbose=True)
print()
print('-- ccp_alpha=0.02 --')
postHocDecisionTree(L,tagSpace,tagNames,3,ccp_alpha=0.02,pathVerbose=True)
print()
print('-- ccp_alpha=0.03 --')
postHocDecisionTree(L,tagSpace,tagNames,3,ccp_alpha=0.03,pathVerbose=True)
print()
print('---------------------')

KmeansCompare=False
if(KmeansCompare):
    from sklearn.cluster import KMeans
    BasePartitions=[]
    k=3
    for i in range(0):
        kmeans_model = KMeans(n_clusters=k,n_init=1,max_iter=2).fit(featureSpace) 
        labels = kmeans_model.labels_
        print(labels)
        BasePartitions.append(labels)
        print('KM',i,':')
        postHocDecisionTree(labels,tagSpace,tagNames,3)
        print()
        print('---------------------')


#----- artif ------
from particularDataTreatment import artifDataGeneration
featureSpace,tagSpace,tagNames,gtAttrMat=artifDataGeneration(overlapOpt=0)
print('--- ARTIFICIAL DATASET ---')
print('-- DEFAULT --')
postHocDecisionTree(LA,tagSpace,tagNames,3,pathVerbose=True)
print()
print('-- ccp_alpha=0.01 --')
postHocDecisionTree(LA,tagSpace,tagNames,3,ccp_alpha=0.01,pathVerbose=True)
print()
print('-- ccp_alpha=0.02 --')
postHocDecisionTree(LA,tagSpace,tagNames,3,ccp_alpha=0.02,pathVerbose=True)
print()
print('-- ccp_alpha=0.03 --')
postHocDecisionTree(LA,tagSpace,tagNames,3,ccp_alpha=0.03,pathVerbose=True)
print()
print('---------------------')

print('---------------------')
KmeansCompare=False
max_leaf_nodes=None
if(KmeansCompare):
    from sklearn.cluster import KMeans
    BasePartitions=[]
    k=3
    for i in range(2):
        kmeans_model = KMeans(n_clusters=k,n_init=1,max_iter=1).fit(featureSpace) 
        labels = kmeans_model.labels_
        print(labels)
        if(True):
            BasePartitions.append(labels)
            print('KM',i,':')
            postHocDecisionTree(labels,tagSpace,tagNames,3,max_leaf_nodes=max_leaf_nodes)
        else:
            print(i,'skipped.')
        print('---------------------')


print('---------------------')