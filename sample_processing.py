# Functions to process a set of MCMC samples 
import numpy as np
from typing import List
from sklearn.cluster import AgglomerativeClustering

def calc_hit_matrix_sample(partition:np .array) -> np.array:
    """ calculates a co-occurence matrix for a given sample 
    partition: np.array (1d) of cluster indices of data

    Args:
        partition (np.array): An array (n dim) of hard clustering indices for a set of n data points

    Returns:
        np.array: n x n binary indicator matrix, with 1 when points i,j are in same cluster
    """    
    
    # Form hit matrix
    n_part = len(partition)
    hit_matrix = np.zeros([n_part,n_part])
    for ii in range(n_part):
        for jj in range(n_part):
            hit_matrix[ii,jj] = (partition[ii]==partition[jj])
    return hit_matrix

def calc_hit_matrix(partition_samples:List[np.array],
                    burn_samples:int=0,
                    normalize:bool=True) -> np.array:
    """ From a set of MCMC samples of hard clusterings compute counts of co-occurence (normalize = False) or
    or normalize counts of co-occurence(normalize = True)

    Args:
        partition_samples (List[np.array]): List contatining samples of hard clsuterings
        burn_samples (int, optional): number of samples to discard at the start. Defaults to 0.
        normalize (bool, optional): Whether to normalize. Defaults to True.

    Returns:
        np.array: Counts or normalized counts.
    """    
    # for a given set of partition samples, calculate the number of times 
    n_partitions = len(partition_samples)

    n = len(partition_samples[0])  # No. of data points in each sample
    hit_matrix_overall = np.zeros([n,n])
    n_samples_used = n_partitions - burn_samples

    for i in range(burn_samples,n_partitions):
        hit_matrix_overall += calc_hit_matrix_sample(partition_samples[i])
    
    if normalize:
        return (1/n_samples_used)*hit_matrix_overall
    
    else:
        return hit_matrix_overall


def agglo_cluster(sim_matrix: np.array,n_clust:int,
                  linkage_type:str='average') -> np.array:

    """ Applys hierarchical clustering to a similarity matrix (sim_matrix),
        generating n_clust numbers of clusters

    Args:
        sim_matrix (np.array): similarity matrix, (0-1) range for each element. Eg output from calc_hit_matrix with normalize=True
        n_clust (int): number of clusters wanted
        linkage_type (str, optional): Type of linkage to use, average seems to work best. Defaults to 'average'.

    Returns:
        np.array: _description_
    """    
    model = AgglomerativeClustering(
        affinity='precomputed',
        n_clusters=n_clust,
        linkage=linkage_type).fit(1-sim_matrix)
    
    return model.labels_
