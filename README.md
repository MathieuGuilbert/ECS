# ECS
This repository contains the code associated to the Explainable Clustering (ensemble) Selection (ECS) approach. 

File "imports" is a script importing all necesary libraries.

%----------------

Code is separated as follows:


partitionCreation.py : contains the CP model, and the function launchICES launching the entire pipeline. 
Note: for now, to vary the parameters of the generation, selection and model steps modifying them in launchICES function is needed.

clusterSelection.py: handling the base partition generation, cluster selection and description generation steps

postProcess.py: contains all functions needed after the execution of the model, such as quality measures

compare.py: contains functions used to compare our approach to others.

dataTreatment.py: functions needed to read our different datasets.

clusterQuality.py: Evaluation of individual cluster qualities.


%----------------

Objective criterions of the CP models are as follows:
0 : minimize number of instances attributed to 0 cluster
1 : minimize number of clusters selected
5 : minimize overall number of patterns selected 
2 : maximize number of instances attributed to one and only one cluster
3 : maximize overall number of descriptors selected in cluster descriptions
4 : maximize number of clusters selected
7 : maximize number of instances covered by at least one pattern.
8 : minimize number of instances covered by at least one pattern.
9 : maximize number of instances covered by at least one pattern AND belonging to precisly one cluster.


%----------------

This work was developped by Mathieu Guilbert, and is associated to the ECS approach written by Mathieu Guilbert, Christel Vrain and Thi-Bich-Hanh Dao from the LIFO laboratory in Orléans, France.
 
For any requierement, remark or question, authors can joined by mail at mathieu.guilbert@univ-orleans.fr .
