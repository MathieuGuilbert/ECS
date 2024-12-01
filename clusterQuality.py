#-- This file contains functions to evaluate the quality of clusters --

import numpy as np
from sklearn.metrics import pairwise_distances,jaccard_score,silhouette_score
from dataTreatment import computeFasterDistMat,computeFastestEuclideanDistance
from scipy.special import comb
from scipy.optimize import linear_sum_assignment

import math

def WCSS(C:list):
	'''Compute the WCSS (Within Cluster Sum of Square) value of a particular cluster.'''
	centroid = np.mean(C, axis=0)
	dists = pairwise_distances(C, centroid.reshape(1, -1))
	wcss = np.sum(dists**2)
	return wcss

def computeWCSSs(featureSpace:list,clusterIds:list):
	'''Returns the WCSS values for all entry clusters.'''
	res=[]
	for c in clusterIds:
		clustData=[featureSpace[i] for i in c]
		res.append(int(WCSS(clustData)*100)) #*100 to keep 2 number after comas
	return res

def jaccard_sample(y_true, y_pred):
	'''Calculate the Jaccard similarity score Using 'samples' average for sample-wise evaluation.'''
	jaccard = jaccard_score(y_true, y_pred, average='samples')
	return jaccard

def jaccard_micro(y_true, y_pred):
	'''Calculate the Jaccard similarity score Using Micro-average across all entries.'''
	jaccard = jaccard_score(y_true, y_pred, average='micro')
	return jaccard

def accurate_jaccard(allocation1, allocation2):
    """
    Compute the accurate Jaccard measure between two clustering allocations.

    Parameters:
    - allocation1: ndarray, shape (n_samples, n_clusters1)
        Binary allocation matrix for the first clustering.
    - allocation2: ndarray, shape (n_samples, n_clusters2)
        Binary allocation matrix for the second clustering.

    Returns:
    - jaccard: float
        The accurate Jaccard measure.
    """
    allocation1 = np.array(allocation1)
    allocation2 = np.array(allocation2)

    n_clusters1 = allocation1.shape[1]
    n_clusters2 = allocation2.shape[1]

    # Create a cost matrix for Jaccard similarities (used by Hungarian)
    cost_matrix = np.zeros((n_clusters1, n_clusters2))

    for i in range(n_clusters1):
        for j in range(n_clusters2):
            intersection = np.sum(np.logical_and(allocation1[:, i], allocation2[:, j]))
            union = np.sum(np.logical_or(allocation1[:, i], allocation2[:, j]))
            jaccard_similarity = intersection / union if union > 0 else 0.0
            cost_matrix[i, j] = 1 - jaccard_similarity  # Cost = 1 - Jaccard similarity

    # Use Hungarian algorithm to find the best matching
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Compute the accurate Jaccard measure using the optimal matching
    total_similarity = 0.0
    for i, j in zip(row_indices, col_indices):
        intersection = np.sum(np.logical_and(allocation1[:, i], allocation2[:, j]))
        union = np.sum(np.logical_or(allocation1[:, i], allocation2[:, j]))
        jaccard_similarity = intersection / union if union > 0 else 0.0
        total_similarity += jaccard_similarity

    # Average Jaccard similarity
    return total_similarity / max(n_clusters1, n_clusters2)


def overlapping_ari(allocation1, allocation2):
    """
    Compute Adjusted Rand Index (ARI) for overlapping cluster allocations matrices.

    Parameters:
    - allocation1: ndarray, shape (n_samples, n_clusters1)
        Binary allocation matrix for the first clustering (overlapping).
    - allocation2: ndarray, shape (n_samples, n_clusters2)
        Binary allocation matrix for the second clustering (overlapping).

    Returns:
    - ari: float
        Adjusted Rand Index for overlapping clusters.
    """
    # Convert to NumPy arrays
    allocation1 = np.array(allocation1)
    allocation2 = np.array(allocation2)

    n_samples = allocation1.shape[0]

    # Initialize counts
    total_pairs = comb(n_samples, 2)
    observed_agreements = 0
    observed_disagreements = 0

    # Count agreements/disagreements for each pair of points
    for i in range(n_samples):
        for j in range(i + 1, n_samples):  # Loop over all pairs (i, j)
            clusters1_i = np.where(allocation1[i] == 1)[0]
            clusters1_j = np.where(allocation1[j] == 1)[0]
            clusters2_i = np.where(allocation2[i] == 1)[0]
            clusters2_j = np.where(allocation2[j] == 1)[0]

            # Agreement if clusters overlap in both partitions
            overlap1 = set(clusters1_i).intersection(clusters1_j)
            overlap2 = set(clusters2_i).intersection(clusters2_j)

            if overlap1 and overlap2:
                observed_agreements += len(overlap1.intersection(overlap2))
            else:
                observed_disagreements += 1

    # Compute expected agreements (chance level)
    expected_agreements = (
        np.sum(comb(allocation1.sum(axis=0), 2)) *
        np.sum(comb(allocation2.sum(axis=0), 2))
    ) / total_pairs

    # Compute maximum possible agreements
    max_agreements = 0.5 * (
        np.sum(comb(allocation1.sum(axis=0), 2)) +
        np.sum(comb(allocation2.sum(axis=0), 2))
    )

    # Adjusted Rand Index
    ari = (observed_agreements - expected_agreements) / (max_agreements - expected_agreements)
    return ari


def overlapping_ari_with_alignment(allocation1, allocation2):
    """
    Compute Adjusted Rand Index (ARI) for overlapping clusters with optimal cluster alignment.

    Parameters:
    - allocation1: ndarray, shape (n_samples, n_clusters1)
        Binary allocation matrix for the first clustering (overlapping).
    - allocation2: ndarray, shape (n_samples, n_clusters2)
        Binary allocation matrix for the second clustering (overlapping).

    Returns:
    - ari: float
        Adjusted Rand Index for overlapping clusters.
    """
    allocation1 = np.array(allocation1)
    allocation2 = np.array(allocation2)
    n_samples = allocation1.shape[0]

    # Compute the cost matrix for alignment
    n_clusters1 = allocation1.shape[1]
    n_clusters2 = allocation2.shape[1]
    cost_matrix = np.zeros((n_clusters1, n_clusters2))

    for i in range(n_clusters1):
        for j in range(n_clusters2):
            # Cost is the dissimilarity (1 - Jaccard similarity)
            intersection = np.sum(np.logical_and(allocation1[:, i], allocation2[:, j]))
            union = np.sum(np.logical_or(allocation1[:, i], allocation2[:, j]))
            similarity = intersection / union if union > 0 else 0.0
            cost_matrix[i, j] = 1 - similarity

    # Use Hungarian algorithm for optimal alignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Reorder allocation2 based on the alignment
    aligned_allocation2 = np.zeros_like(allocation1)
    for i, j in zip(row_indices, col_indices):
        aligned_allocation2[:, i] = allocation2[:, j]

    # Compute ARI with aligned clusters
    return overlapping_ari(allocation1, aligned_allocation2)
