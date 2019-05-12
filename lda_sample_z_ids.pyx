from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
# from libc.stdlib cimport rand, RAND_MAX # It is slower
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.utility cimport pair
ctypedef np.float64_t dtype_t
ctypedef np.uint32_t uitype_t
ctypedef vector[uitype_t] vecu_t
ctypedef map[Py_ssize_t, vecu_t] map_ivecu_t
ctypedef map[Py_ssize_t, Py_ssize_t] map_ii_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_ids(dtype_t[:,:,::1] Adk_avg,
                 dtype_t[:,:,::1] Bkw_avg,
                 dtype_t[:,:,::1] phi_in,
                 dtype_t[:,::1] uni_rvs,
                 vector[map_ii_t] doc_dicts, # a copy from python list to c++ vector is made here
                 double alpha,
                 int num_sim,
                 int burn_in):

    cdef:
        Py_ssize_t M = Adk_avg.shape[0]
        Py_ssize_t D = Adk_avg.shape[1]
        Py_ssize_t K = Adk_avg.shape[2]
        Py_ssize_t W = Bkw_avg.shape[2]
        Py_ssize_t m
        # dtype_t[:,::1] uni_rvs = np.random.uniform(size = (M, num_words*(num_sim+1)))
        # vector[dtype_t[::1, :]] phi = [phi_in[m].copy_fortran() for m in range(M)] # WHY CANT I DO THIS???
        # dtype_t[::view.indirect, ::1, :] phi = np.array([phi_in[m].copy_fortran() for m in range(M)], order='F')
    #     vector[old_arr_t] phi = vector[old_arr_t](M, old_arr_t())
    # for m in range(M): phi[m] = phi_in[m].copy_fortran()

    for m in prange(M, nogil=True):
        sample_for_each_phi(Adk_avg[m], Bkw_avg[m], phi_in[m], uni_rvs[m], doc_dicts, alpha, num_sim, burn_in, D, K, W)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sample_for_each_phi(dtype_t[:,::1] Adk_avg,
                              dtype_t[:,::1] Bkw_avg,
                              dtype_t[:,::1] phi,
                              # dtype_t[::1,:] phi,
                              dtype_t[::1] uni_rvs,
                              vector[map_ii_t]& doc_dicts,
                              double alpha, int num_sim, int burn_in,
                              Py_ssize_t D, Py_ssize_t K, Py_ssize_t W) nogil:
    cdef:
        vector[vecu_t] Adk = vector[vecu_t](D, vecu_t(K, 0))
        vector[vecu_t] Bwk = vector[vecu_t](W, vecu_t(K, 0))
        vector[map_ivecu_t] z_map = vector[map_ivecu_t](D, map_ivecu_t())
        pair[Py_ssize_t, Py_ssize_t] w_cnt
        vector[dtype_t] cumprobs = vector[dtype_t](K, 0.)
        Py_ssize_t rc_start = 0, rc_mid, rc_stop = K
        Py_ssize_t uni_idx = 0
        Py_ssize_t d, w, i, k, sim, word_cnt, zInit, zOld, zNew
        double num_eff_smp = float(num_sim - burn_in)
        double prob_sum, uni_rv

    uni_rv = 1./float(K)
    for k in range(1, K):
        cumprobs[k] = cumprobs[k-1] + uni_rv

    # Make sure the counts are initialised to zero
    for i in range(D):
        for k in range(K):
            Adk_avg[i,k] = 0
    for k in range(K):
        for w in range(W):
            Bkw_avg[k,w] = 0
    # Initialise the z_id for each document in the batch
    for d in range(D):
        for w_cnt in doc_dicts[d]:
            w = w_cnt.first
            word_cnt = w_cnt.second
            z_map[d][w] = vecu_t(word_cnt, 0)
            for i in range(word_cnt): #z[d][w]:
                uni_rv = uni_rvs[uni_idx] #np.random.rand() * prob_sum
                uni_idx += 1
                # uni_rv = rand()/<double>RAND_MAX

                # inline randcat function call
                rc_start = 0
                rc_stop  = K
                while rc_start < rc_stop - 1:
                    rc_mid = (rc_start + rc_stop) // 2
                    if cumprobs[rc_mid] <= uni_rv:
                        rc_start = rc_mid
                    else:
                        rc_stop = rc_mid
                zInit    = rc_start
                Adk[d][zInit] += 1
                Bwk[w][zInit] += 1
                z_map[d][w][i] = zInit

    # Draw samples from the posterior on z_ids using Gibbs sampling
    for sim in range(num_sim):
        for d in range(0, D):
            for w_cnt in doc_dicts[d]:
                w = w_cnt.first
                word_cnt = w_cnt.second
                for i in range(word_cnt):
                    zOld = z_map[d][w][i]
                    prob_sum = 0
                    # Faster than using numpy elt product
                    for k in range(K):
                        cumprobs[k] = prob_sum
                        prob_sum +=  (alpha + Adk[d][k] - (k == zOld)) * phi[k,w]
                    uni_rv = prob_sum * uni_rvs[uni_idx]
                    uni_idx += 1
                    # uni_rv = rand()/<double>RAND_MAX * prob_sum

                    # inline randcat function call
                    rc_start = 0
                    rc_stop  = K
                    while rc_start < rc_stop - 1:
                        rc_mid = (rc_start + rc_stop) // 2
                        if cumprobs[rc_mid] <= uni_rv:
                            rc_start = rc_mid
                        else:
                            rc_stop = rc_mid

                    zNew = rc_start
                    z_map[d][w][i] = zNew
                    Adk[d][zOld] -= 1; Adk[d][zNew] += 1
                    Bwk[w][zOld] -= 1; Bwk[w][zNew] += 1

        if sim >= burn_in:
            for d in range(D):
                for k in range(K):
                    Adk_avg[d,k] = Adk_avg[d,k] + Adk[d][k]
            for k in range(K):
                for w in range(W):
                    Bkw_avg[k,w] = Bkw_avg[k,w] + Bwk[w][k]

    for d in range(D):
        for k in range(K):
            Adk_avg[d,k] = Adk_avg[d,k] / num_eff_smp
    for k in range(K):
        for w in range(W):
            Bkw_avg[k,w] = Bkw_avg[k,w] / num_eff_smp

