# Contains main code for needed for EPA gibbs sampler

from scipy.stats import norm,multivariate_normal, gamma, beta
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Tuple
import numba 
import pandas as pd

@numba.njit 
def logsumexp(x:np.array)->np.array:
    """ Utility function for log - sum exp trick in normalization of log prob vector

    Args:
        x (np.array): Vector of log probabilities to normalize

    Returns:
        np.array: 
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

@numba.njit 
def exp_sim_func(x1:float, x2:float, tau:float=1) -> float:
    """ Exponential similarity function

    Args:
        x1 (float): first input
        x2 (float): second input

    Returns:
        float: exponential similarity between x1 and x2
    """    
    return np.exp(-tau*np.linalg.norm(x1-x2)**2)


def partition_log_pdf(partition: np.ndarray,
                      sim_mat: np.ndarray,
                      order: np.ndarray,
                      alpha: float,
                      delta: float) -> float:

    """ Calculates the log probability of a partition given (This implementation is pure python and slow, use 
    partition_log_pdf_fast which is optimized with numba, this function is kept primarily for checking/testing)

    Args:
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"


    # keeps track of clusters seen
    clusters_dict = {}
    
    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    log_p = 0
    # Loop over each element (based on order) (think this is necessary)
    for t_1, o in enumerate(order):

        t = t_1 + 1 # to match t in formula

        c_o = partition[o]

        if c_o in clusters_dict.keys():  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_members = clusters_dict[c_o]
            
            total_seen = sum(list(clusters_dict.values()), [])
            
            # calculate p_t
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(sim_mat[o,cluster_members]) / np.sum(sim_mat[o,total_seen]))

            # Update seen clusters
            clusters_dict[c_o].append(o)

        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)
            
            # add new cluster
            clusters_dict[c_o] = [o]

        log_p += np.log(p_t)
        
    return log_p


@numba.njit
def partition_log_pdf_fast(partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0) -> float:
    """ Calculates the log probability of a partition given.

    Args:
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    log_p = 0

    # Loop over each element (based on order) (think this is necessary)
    for t_1, o in enumerate(order):

        t = t_1 + 1 # to match t in formula

        c_o = partition[o]  # cluster membership of current point
        
        if t_1 == 0:
            points_seen = order[0:t_1]
        else:
            points_seen = order[0:t_1-1]

        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))
        
        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                       np.sum(point_pairwise_dist[points_seen]))


        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)


        log_p += np.log(p_t)

    return log_p

def unit_log_likelihood_pdf(y:float,
                            x:np.array,
                            phi:np.array,
                            model:str='Gaussian',
                            sigma_reg:float=1) -> float:
    """_summary_

    Args:
        y (float): y_i value
        x (np.array): X_i value
        phi (np.array): phi_i the parameters of the cluster point i is assigned to
        model (str, optional): Gaussian for gaussian linear model. Defaults to 'Gaussian'.
        sigma_reg (float, optional): sigma of regression likelihood. Defaults to 1.

    Returns:
        float: unit log likelihood
    """

    if model == 'Gaussian':
        return norm.logpdf(y, loc=x@phi, scale=sigma_reg)
    else:
        return 1


def generate_candidate_partitions_alt(i:int,
                                      partition:np.array,
                                      names_used:List[int])->Tuple[List[np.array], List[int], int]:
    """generates candidate partitions but with new clust partitions having +1 of max partition
    note always puts the new clust partition as the first element

    Args:
        i (int): which data point (in partition) to generate candidates for
        partition (np.array):  1 x n dim array of cluster indices for each data points
        names_used (List[int]): list of ids which have been used as cluster names so far

    Returns:
        Tuple(List[np.array], List[int], int): candidate_partitions: list of arrays where each array corresponds to candidate partition
                                               clust_ids: the unique cluster labels from the input partition
                                               new_clust_id: the unique cluster labels from the output partition
    """
    
    clust_ids = list(np.unique(partition))

    # generate label for new cluster as +1 of the largest current label
    new_clust_id = max(names_used) + 1

    # insert at 0 element, the new cluster id name
    clust_ids.insert(0,new_clust_id)
    candidate_partitions = []

    # Loop over each unique index and generate a partition where the i-th entry's cluster id is replaced 
    for c in clust_ids:
        cand_part = partition.copy()
        cand_part[i] = c

        candidate_partitions.append(cand_part)
    return candidate_partitions, clust_ids, new_clust_id



def sample_conditional_i_clust_alt(i:int , partition:np.array,
                                 alpha:float, delta:float,
                                  sim_mat:np.array, order:np.array,
                                   phi:np.array, y:np.array,
                                    x:np.array, sigma_reg:float,
                                  names_used:List[int], phi_base_mean:float,
                                   phi_base_cov:np.array,reordering:bool=True) -> Tuple[np.array, np.array, List[int]]:    
    """
    partition: np.array (1xn) dim array of cluster indices for each data points
    i: which data point to generate candidates for
    sim_mat: n x n matrix of similarities 
    phis: matrix / vector each column is beta_j
    order: array of 0:(n-1) of any order indicating the (randomly sampled order)
    alpha: alpha parameter of distribution
    delta: delta parameter of distribution
    y: y_data
    x: x_data
    names_used: str of cluster ids used thus far
    reordering: whether to re-order cluster labels to prevent cluster labels from getting large
    """
    
    y_i = y[i]
    x_i = x[i]

    # generate candidate partitions
    candidate_partitions, clust_ids, new_clust_id = generate_candidate_partitions_alt(i,
                                                                                      partition,
                                                                                      names_used)
    updated_names = names_used.copy()
    # calc log likelihood of each partition
    log_liks = []
    
    for c_name,cp in zip(clust_ids, candidate_partitions):
        
        # get loglikelihood of the partition
        partition_comp =  partition_log_pdf_fast(cp,
                                        sim_mat, 
                                        order, 
                                        alpha, delta) 

        if c_name == new_clust_id: # if it is the new cluster (Note this is always first)

            phi_new = np.random.multivariate_normal(phi_base_mean,phi_base_cov) # sample something from prior

            ll_comp = unit_log_likelihood_pdf(y_i,
                                    x_i, phi_new,
                                    model='Gaussian',
                                    sigma_reg=sigma_reg)

        else: # it is an existiing cluster
            #print("phi used: ", phi[c_name-1])
            ll_comp = unit_log_likelihood_pdf(y_i,
                                x_i, phi[c_name-1],  # 1 based indexing for cluster but 0 for phi
                                model='Gaussian',
                                sigma_reg=sigma_reg)
        
        # calculate probability of candidate partition and collect
        log_prob_cp = partition_comp + ll_comp
        #print(f"Correct LP {c_name}: {log_prob_cp}")
        #print(f'Correct Partition comp {c_name}: , {partition_comp}')

        #print("Partition: ", cp)
        #print(' ')
        log_liks.append(log_prob_cp)


    # Collect probabilities and normalize with log-sum-exp trick
    log_liks = np.array(log_liks)
    cand_probs = np.exp(log_liks - logsumexp(log_liks))

    # sample outcome partition for candidate partitions
    cand_part_choice = np.random.choice(np.arange(len(cand_probs)), p=cand_probs)
    output_partition = candidate_partitions[cand_part_choice]

    if cand_part_choice == 0:  # if new partition formed then update an extra name and phi
        updated_names.append(new_clust_id)
        phi = np.concatenate([phi,np.array([phi_new])])
    
    
    if reordering == True:
    # Re-indexing step
        # re-index partition
        new_labels, existing_labels, relab_map = gen_reindex(output_partition)
        recoded_partition = remap_partition(output_partition, relab_map)

        # reset phi
        # Drop all empty elements
        phi_new = phi[existing_labels-1]
        #print(existing_labels-1)

        return recoded_partition, phi_new, list(np.sort(new_labels))
    
    else:
        return output_partition, phi,updated_names 

def remap_partition(partition, relab_dict):
    # from https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Basically maps partition to a new set of indexes based on relab_dict.
    u,inv = np.unique(partition,return_inverse = True)
    recoded_partition = np.array([relab_dict[x] for x in u])[inv].reshape(partition.shape)
    return recoded_partition

def gen_reindex(partition):
    # Get the unique labels of the current partition
    unique_labels = pd.Series(partition).value_counts()
    unique_labels = np.array(unique_labels.index)
    #unique_labels = np.unique(partition)

    # Generate new labels from 1 - max label
    new_labels = np.arange(len(unique_labels)) + 1
    return new_labels, unique_labels, dict(zip(unique_labels, new_labels))


def permute_k(current_ordering:np.array ,k:int) -> np.array:
    """ Permutes the first k elements of current_ordering, used to generate prosals for sigma (random ordering)

    Args:
        current_ordering (np.array): Current ordering of variables, array of unique integers with values in 0-(n-1)
        k (int): number of k to permute

    Returns:
        np.array: _description_
    """    
    # permutes first k elements in current_ordering
    # k: int < len(current_ordering)
    # current_ordering np.array of integers signifying order
    
    first_k = current_ordering[0:k]
    remaining_k = current_ordering[k::]

    first_k_permuted = first_k.copy()
    np.random.shuffle(first_k_permuted)

    permuted = np.concatenate([first_k_permuted, remaining_k])
    
    return permuted


def grw_proposal(cur_val: float, sd: float) -> float:
    """ proposal for a gaussian random walk

    Args:
        cur_val (float): current value of the random walk
        sd (float): standard deviation on innovations

    Returns:
        float: a proposed value
    """

    # gaussian random walk proposal distribution
    # proposes normal centered at cur_val with std sd
    return np.random.normal(cur_val, sd)



def delta_prior(delta: float, a_delta:float , b_delta:float , w:float)->float:
    """ Calculates the pdf of a delta prior. The prior is a mixture of a beta(a_delta, b_delta) amd a point mass at 0
    with weights (1-w) and w respectivetly.

    Args:
        delta (float): delta value to calculate pdf for
        a_delta (float): a parameter in beta component
        b_delta (float): b parameter in beta component
        w (float): mixture weight

    Returns:
        float: pdf value evaluted at delta
    """

    # mixture prior for beta parameter
    # mixture of a point mass at 0 and a beta(a_delta, b_delta) distribution with mixing weights w
    
    # point mass pdf
    if delta == 0:
        zero_ind = 1
    else:
        zero_ind = 0
    
    # beta pdf:
    beta_comp = beta.pdf(delta, a_delta, b_delta)
    
    return w*zero_ind + (1-w)*beta_comp
    
def metropolis_step_alpha(alpha_cur, sd, a_alpha, b_alpha,
                         partition,
                             delta,
                             sim_mat,
                             order,bounds=None):
    
    # metropolis step for alpha
    
    # Sample proposal
    alpha_prop = grw_proposal(alpha_cur, sd)
    
    if bounds:
        if (alpha_prop<bounds[0]) or (alpha_prop>bounds[1]):
            return alpha_cur
    
    # Get prior log probs (gamma distribution)
    log_prior_cur = gamma.logpdf(alpha_cur, a=a_alpha, scale=1/b_alpha)
    log_prior_prop = gamma.logpdf(alpha_prop, a=a_alpha, scale=1/b_alpha)
    
    # Likelihood values
    log_ll_cur = partition_log_pdf_fast(partition, sim_mat, order, alpha_cur, delta)
    log_ll_prop = partition_log_pdf_fast(partition, sim_mat, order, alpha_prop, delta)
    
    log_num = log_prior_prop + log_ll_prop
    log_denom = log_prior_cur + log_ll_cur
    # get ratio
    mh_ratio = np.exp(log_num - log_denom)
    
    a = min(1, mh_ratio)
    #print(a)
    
    # Accept proposal
    if np.random.uniform(0,1) < a:
        return alpha_prop
    else: # Reject proposa
        return alpha_cur
    

def metropolis_step_delta(delta_cur, sd, a_delta, b_delta, w,
                         partition,
                             alpha,
                             sim_mat,
                             order,bounds=None):
    
    # Sample proposal
    delta_prop = grw_proposal(delta_cur, sd)
    
    if bounds:
        if (delta_prop<bounds[0]) or (delta_prop>bounds[1]):
            return delta_cur
    
    # Get prior probs
    prior_cur = delta_prior(delta_cur, a_delta=a_delta, b_delta=b_delta, w=w)
    prior_prop = delta_prior(delta_prop, a_delta=a_delta, b_delta=b_delta, w=w)
    
    # Likelihood values
    ll_cur = np.exp(partition_log_pdf_fast(partition, sim_mat, order, alpha, delta_cur))
    ll_prop = np.exp(partition_log_pdf_fast(partition, sim_mat, order, alpha, delta_prop))
    
    # get ratio
    mh_ratio = (prior_prop*ll_prop) / (prior_cur*ll_cur)
    
    a = min(1, mh_ratio)
    #print(a)
    
    # Accept proposal
    if np.random.uniform(0,1) < a:
        return delta_prop
    else: # Reject proposa
        return delta_cur

def metropolis_step_order(order_current:np.ndarray,
                          alpha:float,
                          delta:float,
                          partition:np.ndarray,
                          sim_mat:np.ndarray, k:int) -> np.ndarray:
    """ Metropolis step for sampling the order

    Args:
        order_current (np.ndarray): Current order
        alpha (float): alpha parameter
        delta (float): delta parameter
        partition (np.ndarray): current paratition
        sim_mat (np.ndarray): similarity matrix
        k (int): hyperparameter k, decides how many elements to permute

    Returns:
        np.ndarray: Sampled order
    """    
    
    # calculate log partition prob of curent point
    log_partition_prob_current = partition_log_pdf_fast(partition,
                                                  sim_mat,
                                                  order_current,
                                                  alpha,
                                                  delta) 
    # Sample an order
    order_sample = permute_k(order_current, k)
    
    # calculate log partition prob of proposed point
    log_partition_prob_proposed = partition_log_pdf_fast(partition,
                                                  sim_mat,
                                                  order_sample,
                                                  alpha,
                                                  delta) 
    
    mh_ratio = np.exp(log_partition_prob_proposed - log_partition_prob_current)
    
    # Compare
    a = min(1, mh_ratio)
    
    # Accept proposal
    if np.random.uniform(0,1) < a:
        return order_sample
    else: # Reject proposal
        return order_current

def sample_phi(phi_cur:np.array, y:np.array, x:np.array, partition:np.array,
               phi_mean_prior:np.array,
               phi_cov_prior:np.array,
               sigma_reg:float)->np.array:
    """ Samples phi from the full conditional, note this is for a linear regression problem

    Args:
        phi_cur (np.array): current value for phi
        y (float): n dim array of y values 
        x (np.array): n x p array of x values (note add 1s for intercept)
        partition (np.array): current partition
        phi_mean_prior (np.array): prior mean for coefficients
        phi_cov_prior (np.array): prior covariance for coefficients
        sigma_reg (float): error in linear regression outcome

    Returns:
        np.array: Sample from full conditional of phi
    """

    sigma_sq_reg = sigma_reg**2
    active_clust_ids = np.unique(partition)

    phi_sample = phi_cur.copy()
    
    # Update those in phi cur that are active clusters
    for c_id in active_clust_ids:

        y_vals = y[np.where(partition==c_id)[0]]#.reshape(-1,1)
        x_vals = x[np.where(partition==c_id)[0]]

        # calc quantities
        xtx = x_vals.T @ x_vals
        xty = x_vals.T @ y_vals

        phi_reg_post_cov_inv = (1/sigma_sq_reg)*xtx  + np.linalg.inv(phi_cov_prior)
        phi_reg_post_cov = np.linalg.inv(phi_reg_post_cov_inv)

        phi_reg_post_mean = phi_reg_post_cov @ (((1/sigma_sq_reg)*xty)  +\
                                                np.linalg.inv(phi_cov_prior) @ phi_mean_prior)

        # Sample posterior
        clust_phi_sample = np.random.multivariate_normal(phi_reg_post_mean.reshape(-1),
                                                        phi_reg_post_cov)

        phi_sample[c_id-1] = clust_phi_sample
        
    return phi_sample



def calc_log_joint(partition:np.array, phi:np.array,
                   y:np.array, x:np.array,
                   sim_mat:np.array, order:np.array,
                   alpha:float, delta:float, sigma_reg:float) -> float:
    """ Calculates log joint of the EPA regression model.

    Args:
        partition (np.array): np.array (1xn) dim array of cluster indices for each data points
        phi (np.array): parameters of the sampling model
        y (np.array): y data
        x (np.array): X data
        sim_mat (np.array): n x n matrix of similarities 
        order (np.array): array of 0:(n-1) of any order indicating the (randomly sampled order)
        alpha (float): alpha parameter of distribution
        delta (float): delta parameter of distribution
        sigma_reg (float): sigma parameter of linear regression

    Returns:
        float: log joint
    """
    
    # Partition log prob
    partition_log_prob = partition_log_pdf_fast(partition,
                                        sim_mat, 
                                        order, 
                                        alpha, delta)
  
    # phi log prob
    sampling_log_prob = 0
    for i,c in enumerate(partition):
        sampling_log_prob += unit_log_likelihood_pdf(y[i],
                                                     x[i],
                                                     phi[c-1], model='Gaussian', sigma_reg=sigma_reg)
    
    
    return partition_log_prob + sampling_log_prob

   

# Gibbs optimization code
@numba.njit()
def partition_log_pdf_factors(start_id:int,
                              return_fac: bool=False,
                              partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0) -> float:
    """ Calculates the log probability of with some factors pre-computed. Computes factors in the partition 
    from start_id onwards

    Args:
        start_id (int): Index from with to start computing the partial pdf, note this corresponds to 
        delta_{start_id}
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    
    
    log_p = np.zeros(len(partition))

    # Loop over each element (based on order) (think this is necessary)
    for t_1, o in enumerate(order):
        
        t = t_1 + 1 # to match t in formula

        c_o = partition[o]

        if t_1 == 0:
            points_seen = order[0:t_1]
        else:
            points_seen = order[0:t_1-1]
            
        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))

        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                       np.sum(point_pairwise_dist[points_seen]))


        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)

        log_p[t_1] = np.log(p_t)

    
    return log_p


@numba.njit()
def partition_log_pdf_partial(start_id:int,
                              return_fac: bool=False,
                              partition: np.ndarray=np.array([]),
                      sim_mat: np.ndarray=np.array([[]]),
                      order: np.ndarray=np.array([]),
                      alpha: float=1,
                      delta: float=0 ) -> float:
    """ Calculates the log probability of with some factors pre-computed. Computes factors in the partition 
    from start_id onwards

    Args:
        start_id (int): Index from with to start computing the partial pdf, note this corresponds to 
        delta_{start_id}
        partition (np.ndarray): Cluster indices of each point
        sim_mat (np.ndarray): n x n matrix of similarities
        order (np.ndarray): n dim vector of order indicating the randomly sampled order
        alpha (float): alpha parameter
        delta (float): delta parameter

    Returns:
        float: log probability of partition
    """    

    assert len(order) == len(partition), "Length of order and partition inputs not the same"

    q = 1 # No of clusters in t-1 (? one subset in empty set?)
    
    log_p = 0

    # Loop over each element (based on order) (think this is necessary)
    for t_1, o in enumerate(order[start_id:]):
    
        
        t = t_1 + 1 + start_id # to match t in formula

        c_o = partition[o]

        if t == 1:
            points_seen = order[0:t-1]
        else:
            points_seen = order[0:t-2]
            
        clust_labels_seen = partition[points_seen]
        q = len(np.unique(clust_labels_seen))

        if c_o in clust_labels_seen:  # c_o belongs to existing cluster

            # Extract members of cluster c_o is in 
            cluster_member_position = np.where(clust_labels_seen==c_o)[0]
            cluster_members = order[cluster_member_position]

            # calculate p_t
            point_pairwise_dist = sim_mat[o,:]
            p_t = ((t-1 - delta * q)/(alpha + t-1)) * (np.sum(point_pairwise_dist[cluster_members]) / \
                                                       np.sum(point_pairwise_dist[points_seen]))

        else:  # c_o belongs to new cluster
            # calculate p_t
            p_t = (alpha + delta*q)/(alpha+t-1)

        log_p += np.log(p_t)

    
    return log_p



def sample_conditional_i_clust_gibbs_opti(i:int,
                                          order_place:int,
                                   partition:np.array,
                                  return_fac:bool,
                                 pre_compute_factors:np.array,
                                 alpha:float, delta:float,
                                  sim_mat:np.array, order:np.array,
                                   phi:np.array, y:np.array,
                                    x:np.array, sigma_reg:float,
                                  names_used:List[int], phi_base_mean:float,
                                   phi_base_cov:np.array,reordering:bool=True) -> Tuple[np.array, np.array, List[int]]:    
    """
    partition: np.array (1xn) dim array of cluster indices for each data points
    i: which data point to generate candidates for
    order_place: where in the order the data point is.
    pre_compute_factors: np.array (1xn) dim array of factors calculated in the last term.
    sim_mat: n x n matrix of similarities 
    phis: matrix / vector each column is beta_j
    order: array of 0:(n-1) of any order indicating the (randomly sampled order)
    alpha: alpha parameter of distribution
    delta: delta parameter of distribution
    y: y_data
    x: x_data
    names_used: str of cluster ids used thus far
    reordering: whether to re-order cluster labels to prevent cluster labels from getting large
    """
    
    y_i = y[i]
    x_i = x[i]

    # generate candidate partitions
    candidate_partitions, clust_ids, new_clust_id = generate_candidate_partitions_alt(i,
                                                                                      partition,
                                                                                      names_used)
    updated_names = names_used.copy()
    # calc log likelihood of each partition
    log_liks = []
    
    
    for c_name,cp in zip(clust_ids, candidate_partitions):
        
        if not return_fac:
        # get loglikelihood of the partition
            partition_comp_partial =  partition_log_pdf_partial(order_place,  # start_id
                                                        return_fac,
                                                        cp,
                                                        sim_mat, 
                                                        order, 
                                                        alpha, delta) 
        
        else:
            partition_comp_partial =  partition_log_pdf_factors(order_place,  # start_id
                                                        return_fac,
                                                        cp,
                                                        sim_mat, 
                                                        order, 
                                                        alpha, delta) 

        if not return_fac:
            partition_comp = partition_comp_partial + np.sum(pre_compute_factors[:order_place])
            
        else:
            partition_comp = np.sum(partition_comp_partial)
            #print('Partition comp: ', partition_comp)
        if c_name == new_clust_id: # if it is the new cluster (Note this is always first)

            phi_new = np.random.multivariate_normal(phi_base_mean,phi_base_cov) # sample something from prior

            ll_comp = unit_log_likelihood_pdf(y_i,
                                    x_i, phi_new,
                                    model='Gaussian',
                                    sigma_reg=sigma_reg)

        else: # it is an existiing cluster
            #print("phi used: ", phi[c_name-1])
            ll_comp = unit_log_likelihood_pdf(y_i,
                                x_i, phi[c_name-1],  # 1 based indexing for cluster but 0 for phi
                                model='Gaussian',
                                sigma_reg=sigma_reg)
        
        # calculate probability of candidate partition and collect
        log_prob_cp = partition_comp + ll_comp
        #print(f"New LP {c_name}: {log_prob_cp}")
        #print(f'New Partition comp {c_name}: , {partition_comp}')

        #print("Partition: ", cp)
        #print(' ')
        log_liks.append(log_prob_cp)


    # Collect probabilities and normalize with log-sum-exp trick
    log_liks = np.array(log_liks)
    cand_probs = np.exp(log_liks - logsumexp(log_liks))

    # sample outcome partition for candidate partitions
    cand_part_choice = np.random.choice(np.arange(len(cand_probs)), p=cand_probs)
    output_partition = candidate_partitions[cand_part_choice]

    if cand_part_choice == 0:  # if new partition formed then update an extra name and phi
        updated_names.append(new_clust_id)
        phi = np.concatenate([phi,np.array([phi_new])])
    
    
    if reordering == True:
    # Re-indexing step
        # re-index partition
        new_labels, existing_labels, relab_map = gen_reindex(output_partition)
        recoded_partition = remap_partition(output_partition, relab_map)

        # reset phi
        # Drop all empty elements
        phi_new = phi[existing_labels-1]
        #print(existing_labels-1)
        
        if return_fac:
            return recoded_partition, phi_new, list(np.sort(new_labels)), partition_comp_partial
        else:
            return recoded_partition, phi_new, list(np.sort(new_labels))
    
    else:
        
        if return_fac: 
            return output_partition, phi, updated_names, partition_comp_partial
        else:
            return output_partition, phi,updated_names