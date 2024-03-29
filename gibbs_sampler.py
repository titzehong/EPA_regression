from epa import *
from sample_processing import *
import numpy as np
from typing import List
import concurrent.futures
import time 

def gibbs_sample_regression_single_thread(Ys:np.ndarray,
                            Xs:np.ndarray,
                            Zs:np.ndarray,
                            sim_mat:np.ndarray,
                            partition_init: np.ndarray,
                            phi_init: np.ndarray,
                            phi_mean_prior: np.ndarray,
                            phi_cov_prior: np.ndarray,
                            labels_used: List[int],
                            alpha_init: float,
                            delta_init: float,
                            sigma_reg: float=1,
                            n_gibbs: int=5000,
                            n_burn_in: int=2000,
                            k: int=100,
                            a_alpha: float=1,
                            b_alpha: float=10,
                            a_delta: float=1,
                            b_delta: float=1,
                            w: float=0.5):
                                
    n = len(Ys)
    # intialize 
    names_used = labels_used
    alpha_samp = alpha_init

    delta_samp = delta_init
    order_samp = np.arange(n)
    np.random.shuffle(order_samp)
    phi_samp = phi_init
    partition_samp = partition_init

    #### gibbs sampling hyper parameters
    n_gibbs = n_gibbs
    k = k # no. of numbers to permute order

    # GRW sampler param
    rw_sd = 0.2

    # alpha prior
    a_alpha = a_alpha
    b_alpha = b_alpha
    alpha_bounds = [0,1e99]

    # delta prior
    a_delta = a_delta
    b_delta = b_delta
    
    assert (w<=1) and (w>=0)
    w = w
    delta_bounds = [0,1]


    # phi / regression prior
    phi_mean_prior = phi_mean_prior
    phi_cov_prior = phi_cov_prior

    sigma_reg = sigma_reg

    partition_save = []
    alpha_save = []
    delta_save = []
    phi_save = []
    log_prob_save = []

    # Gibbs loop
    for g in range(n_gibbs+n_burn_in):
        if g%100 == 0:
            print("Gibbs: ", g)
            
        # Sample cluster for each i 
        for i in range(len(Ys)):

            partition_samp,phi_samp, names_used = sample_conditional_i_clust_alt(i,
                                            partition_samp,
                                            alpha_samp,
                                            delta_samp,
                                            sim_mat,
                                            order_samp,
                                            phi_samp, Ys, Xs, sigma_reg,
                                            names_used,
                                            phi_base_mean=phi_mean_prior,
                                                phi_base_cov=phi_cov_prior)
            
        
        # Update phis
        phi_samp = sample_phi(phi_samp, Ys, Xs, partition_samp,
                phi_mean_prior,
                phi_cov_prior,
                sigma_reg)
        
        # Sample ordering 
        #order_samp = permute_k(order_samp, k)
        order_samp = metropolis_step_order(order_current=order_samp,
                                        alpha=alpha_samp,
                                        delta=delta_samp,
                                        partition=partition_samp,
                                        sim_mat=sim_mat,
                                        k=k)
        
            
        #### Sample parameters, alpha, sigma
        
        alpha_samp = metropolis_step_alpha(alpha_samp, rw_sd, a_alpha, b_alpha,
                            partition_samp,
                                delta_samp,
                                sim_mat,
                                order_samp,bounds=alpha_bounds)
        
        
        delta_samp = metropolis_step_delta(delta_samp, rw_sd, a_delta, b_delta, w,
                            partition_samp,
                                alpha_samp,
                                sim_mat,
                                order_samp,bounds=delta_bounds)
        
        #alpha_samp = 0.5
        #delta_samp = 0
        
        
        # Calc log prob of result
        log_prob_samp = calc_log_joint(partition=partition_samp,
                                    phi=phi_samp,
                                    y=Ys,
                                    x=Xs,
                                    sim_mat=sim_mat,
                                    order=order_samp,
                                    alpha=alpha_samp,
                                    delta=delta_samp,
                                    sigma_reg = sigma_reg)
        

        if g>n_burn_in:

            # Save sampled values
            log_prob_save.append(log_prob_samp)
            partition_save.append(partition_samp)
            alpha_save.append(alpha_samp)
            delta_save.append(delta_samp)
            phi_save.append(phi_samp)


    return log_prob_save, partition_save, alpha_save, delta_save, phi_save




def gibbs_sample_regression_single_thread_opt(Ys:np.ndarray,
                            Xs:np.ndarray,
                            Zs:np.ndarray,
                            sim_mat:np.ndarray,
                            partition_init: np.ndarray,
                            phi_init: np.ndarray,
                            phi_mean_prior: np.ndarray,
                            phi_cov_prior: np.ndarray,
                            labels_used: List[int],
                            alpha_init: float,
                            delta_init: float,
                            sigma_reg: float=1,
                            n_gibbs: int=5000,
                            n_burn_in: int=2000,
                            k: int=100,
                            a_alpha: float=1,
                            b_alpha: float=10,
                            a_delta: float=1,
                            b_delta: float=1,
                            w: float=0.5):
                         
    n = len(Ys)
    # intialize 
    names_used = labels_used
    alpha_samp = alpha_init

    delta_samp = delta_init
    order_samp = np.arange(n)
    np.random.shuffle(order_samp)
    phi_samp = phi_init
    partition_samp = partition_init

    #### gibbs sampling hyper parameters
    n_gibbs = n_gibbs
    k = k # no. of numbers to permute order

    # GRW sampler param
    rw_sd = 0.2

    # alpha prior
    a_alpha = a_alpha
    b_alpha = b_alpha
    alpha_bounds = [0,1e99]

    # delta prior
    a_delta = a_delta
    b_delta = b_delta
    
    assert (w<=1) and (w>=0)
    w = w
    delta_bounds = [0,1]


    # phi / regression prior
    phi_mean_prior = phi_mean_prior
    phi_cov_prior = phi_cov_prior

    sigma_reg = sigma_reg

    partition_save = []
    alpha_save = []
    delta_save = []
    phi_save = []
    log_prob_save = []

    # Gibbs loop
    for g in range(n_gibbs+n_burn_in):
        if g%100 == 0:
            print("Gibbs: ", g)
            
        # Get last term in order samp
        #last_term_samp = order_samp[-1]
        last_term_id = order_samp[-1]#np.where(order_samp == np.max(order_samp))[0][0]
        # Draw sample for the final term
        partition_samp, phi_samp, names_used, partition_factors = sample_conditional_i_clust_gibbs_opti(last_term_id,
                                        0,
                                        partition_samp,
                                        True,
                                        [],
                                        alpha_samp,
                                        delta_samp,
                                        sim_mat,
                                        order_samp,
                                        phi_samp, Ys, Xs, sigma_reg,
                                        names_used,
                                        # pre_compute_factors = [],
                                        phi_base_mean=phi_mean_prior,
                                            phi_base_cov=phi_cov_prior)
        
        for o in list(range(len(order_samp)-2,-1,-1)):
            
            term_id = order_samp[o]#np.where(order_samp == o)[0][0]
            
            partition_samp, phi_samp, names_used= sample_conditional_i_clust_gibbs_opti(term_id,
                                            o,
                                            partition_samp,
                                            False,
                                            partition_factors,
                                            alpha_samp,
                                            delta_samp,
                                            sim_mat,
                                            order_samp,
                                            phi_samp, Ys, Xs, sigma_reg,
                                            names_used,
                                            phi_base_mean=phi_mean_prior,
                                                phi_base_cov=phi_cov_prior)

        # Update phis
        phi_samp = sample_phi(phi_samp, Ys, Xs, partition_samp,
                phi_mean_prior,
                phi_cov_prior,
                sigma_reg)
        
        # Sample ordering 
        #order_samp = permute_k(order_samp, k)
        order_samp = metropolis_step_order(order_current=order_samp,
                                        alpha=alpha_samp,
                                        delta=delta_samp,
                                        partition=partition_samp,
                                        sim_mat=sim_mat,
                                        k=k)
        
            
        #### Sample parameters, alpha, sigma
        
        alpha_samp = metropolis_step_alpha(alpha_samp, rw_sd, a_alpha, b_alpha,
                            partition_samp,
                                delta_samp,
                                sim_mat,
                                order_samp,bounds=alpha_bounds)
        
        
        delta_samp = metropolis_step_delta(delta_samp, rw_sd, a_delta, b_delta, w,
                            partition_samp,
                                alpha_samp,
                                sim_mat,
                                order_samp,bounds=delta_bounds)
        
        #alpha_samp = 0.5
        #delta_samp = 0
        
        
        # Calc log prob of result
        log_prob_samp = calc_log_joint(partition=partition_samp,
                                    phi=phi_samp,
                                    y=Ys,
                                    x=Xs,
                                    sim_mat=sim_mat,
                                    order=order_samp,
                                    alpha=alpha_samp,
                                    delta=delta_samp,
                                    sigma_reg = sigma_reg)
        

        if g>n_burn_in:

            # Save sampled values
            log_prob_save.append(log_prob_samp)
            partition_save.append(partition_samp)
            alpha_save.append(alpha_samp)
            delta_save.append(delta_samp)
            phi_save.append(phi_samp)


    return log_prob_save, partition_save, alpha_save, delta_save, phi_save


def gibbs_sample_regression(Ys:np.ndarray,
                            Xs:np.ndarray,
                            Zs:np.ndarray,
                            sim_mat:np.ndarray,
                            partition_init: np.ndarray,
                            phi_init: np.ndarray,
                            phi_mean_prior: np.ndarray,
                            phi_cov_prior: np.ndarray,
                            labels_used: List[int],
                            alpha_init: float,
                            delta_init: float,
                            sigma_reg: float=1,
                            n_gibbs: int=5000,
                            n_burn_in: int=2000,
                            k: int=100,
                            a_alpha: float=1,
                            b_alpha: float=10,
                            a_delta: float=1,
                            b_delta: float=1,
                            w: float=0.5,
                            n_chain: int=2,
                            opti: bool=True):
                                
    gibbs_args = {'Ys':Ys,
                'Xs':Xs,
                'Zs':Zs,
                'sim_mat':sim_mat,
                'partition_init':partition_init,
                'phi_init':phi_init,
                'phi_mean_prior':phi_mean_prior,
                'phi_cov_prior':phi_cov_prior,
                'labels_used':labels_used,
                'alpha_init':alpha_init,
                'delta_init':delta_init,
                'sigma_reg':sigma_reg,
                'n_gibbs':n_gibbs,
                'n_burn_in':n_burn_in,
                'k':k,
                'a_alpha':a_alpha,
                'b_alpha':b_alpha,
                'a_delta':a_delta,
                'b_delta':b_delta,
                'w':0.5}


    if n_chain == 1:

        if opti:
            outputs = gibbs_sample_regression_single_thread_opt(**gibbs_args)
        else:
            outputs = gibbs_sample_regression_single_thread(**gibbs_args)

    else:

        if opti:
            outputs = []
            
            with concurrent.futures.ProcessPoolExecutor() as executor:

                results = [executor.submit(gibbs_sample_regression_single_thread_opt, **gibbs_args) for _ in range(n_chain)]
                
                for f in concurrent.futures.as_completed(results):
                    outputs.append(f.result())
        
        else:
            outputs = []
            
            with concurrent.futures.ProcessPoolExecutor() as executor:

                results = [executor.submit(gibbs_sample_regression_single_thread, **gibbs_args) for _ in range(n_chain)]
                
                for f in concurrent.futures.as_completed(results):
                    outputs.append(f.result())
    
    return outputs



def gibbs_sample_regression_single_thread_opt_parallel(Ys:np.ndarray,
                            Xs:np.ndarray,
                            Zs:np.ndarray,
                            sim_mat:np.ndarray,
                            partition_init: np.ndarray,
                            phi_init: np.ndarray,
                            phi_mean_prior: np.ndarray,
                            phi_cov_prior: np.ndarray,
                            labels_used: List[int],
                            alpha_init: float,
                            delta_init: float,
                            sigma_reg: float=1,
                            n_gibbs: int=5000,
                            n_burn_in: int=2000,
                            k: int=100,
                            a_alpha: float=1,
                            b_alpha: float=10,
                            a_delta: float=1,
                            b_delta: float=1,
                            w: float=0.5):
                         
    n = len(Ys)
    # intialize 
    names_used = labels_used
    alpha_samp = alpha_init

    delta_samp = delta_init
    order_samp = np.arange(n)
    np.random.shuffle(order_samp)
    phi_samp = phi_init
    partition_samp = partition_init

    #### gibbs sampling hyper parameters
    n_gibbs = n_gibbs
    k = k # no. of numbers to permute order

    # GRW sampler param
    rw_sd = 0.2

    # alpha prior
    a_alpha = a_alpha
    b_alpha = b_alpha
    alpha_bounds = [0,1e99]

    # delta prior
    a_delta = a_delta
    b_delta = b_delta
    
    assert (w<=1) and (w>=0)
    w = w
    delta_bounds = [0,1]


    # phi / regression prior
    phi_mean_prior = phi_mean_prior
    phi_cov_prior = phi_cov_prior

    sigma_reg = sigma_reg

    partition_save = []
    alpha_save = []
    delta_save = []
    phi_save = []
    log_prob_save = []
    
    s1 = time.time()
    # Gibbs loop
    for g in range(n_gibbs+n_burn_in):
        if g%100 == 0:
            e1 = time.time()
            print("Gibbs: ", g, " Time Taken: ", (e1-s1)/60)
            s1 = time.time()
            
        # Get last term in order samp
        #last_term_samp = order_samp[-1]
        last_term_id = order_samp[-1]#np.where(order_samp == np.max(order_samp))[0][0]
        # Draw sample for the final term
        partition_samp, phi_samp, names_used, partition_factors = sample_conditional_i_clust_gibbs_opti_parallel(last_term_id,
                                        0,
                                        partition_samp,
                                        True,
                                        np.zeros(1),
                                        alpha_samp,
                                        delta_samp,
                                        sim_mat,
                                        order_samp,
                                        phi_samp,
                                        Ys, Xs, sigma_reg,
                                        names_used,
                                        # pre_compute_factors = [],
                                        phi_base_mean=phi_mean_prior,
                                            phi_base_cov=phi_cov_prior)



        for o in list(range(len(order_samp)-2,-1,-1)):

            term_id = order_samp[o]#np.where(order_samp == o)[0][0]

            partition_samp, phi_samp, names_used= sample_conditional_i_clust_gibbs_opti_parallel(term_id,
                                            o,
                                            partition_samp,
                                            False,
                                            partition_factors,
                                            alpha_samp,
                                            delta_samp,
                                            sim_mat,
                                            order_samp,
                                            phi_samp, Ys, Xs, sigma_reg,
                                            names_used,
                                            phi_base_mean=phi_mean_prior,
                                                phi_base_cov=phi_cov_prior)

        partition_samp = partition_samp.astype('int')

        # Update phis
        phi_samp = sample_phi(phi_samp, Ys, Xs, partition_samp,
                phi_mean_prior,
                phi_cov_prior,
                sigma_reg)
        
        # Sample ordering 
        #order_samp = permute_k(order_samp, k)
        order_samp = metropolis_step_order(order_current=order_samp,
                                        alpha=alpha_samp,
                                        delta=delta_samp,
                                        partition=partition_samp,
                                        sim_mat=sim_mat,
                                        k=k)
        
            
        #### Sample parameters, alpha, sigma
        
        alpha_samp = metropolis_step_alpha(alpha_samp, rw_sd, a_alpha, b_alpha,
                            partition_samp,
                                delta_samp,
                                sim_mat,
                                order_samp,bounds=alpha_bounds)
        
        
        delta_samp = metropolis_step_delta(delta_samp, rw_sd, a_delta, b_delta, w,
                            partition_samp,
                                alpha_samp,
                                sim_mat,
                                order_samp,bounds=delta_bounds)
        
        #alpha_samp = 0.5
        #delta_samp = 0
        
        
        # Calc log prob of result
        log_prob_samp = calc_log_joint(partition=partition_samp,
                                    phi=phi_samp,
                                    y=Ys,
                                    x=Xs,
                                    sim_mat=sim_mat,
                                    order=order_samp,
                                    alpha=alpha_samp,
                                    delta=delta_samp,
                                    sigma_reg = sigma_reg)
        

        if g>n_burn_in:

            # Save sampled values
            log_prob_save.append(log_prob_samp)
            partition_save.append(partition_samp)
            alpha_save.append(alpha_samp)
            delta_save.append(delta_samp)
            phi_save.append(phi_samp)


    return log_prob_save, partition_save, alpha_save, delta_save, phi_save
