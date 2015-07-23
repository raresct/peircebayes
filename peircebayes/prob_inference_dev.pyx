"""
:synopsis: Cython module for inner loop methods for :mod:`prob_inference` in a :class:`formula_gen.PBModel`.
"""

from __future__ import division

import copy

cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np

cdef extern from "gsl/gsl_rng.h":
    gsl_rng_type *gsl_rng_mt19937

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

DINT = np.int
DFLOAT = np.float

ctypedef np.int_t DINT_t
ctypedef np.float_t DFLOAT_t

def seed_gsl_cy(unsigned int seed):
    """
    Seed the GNU scientific library (gsl) random number generator (rng).
    
    :param seed: seed for the rng
    :type seed: unsigned int
    :rtype: None
    """
    gsl_rng_set(r, seed)

def backward_plate_cy(
    np.ndarray[DINT_t, ndim=2]      bdd,
    np.ndarray[DINT_t, ndim=3]      plate,
    np.ndarray[DFLOAT_t, ndim=2]    prob):

    """
    Compute backward probability for a set of identical BDDs (with different parameters).
    
    :param bdd:
    :type bdd: see :class:`knowledge_compilation.BDD`
    :param plate:
    :type plate: see :class:`formula_gen.PBPlate`
    :rtype: ndarray of shape [N, NNodes, 2]
    """

    cdef unsigned int nobs   = plate.shape[0]
    cdef unsigned int nnodes = bdd.shape[0]
    cdef np.ndarray[DFLOAT_t, ndim=3] beta = np.zeros([nobs, nnodes, 2],
        dtype=DFLOAT)
    # 0 if root.IsComplement() else 1
    cdef unsigned int nnodes1 = nnodes-1
    cdef unsigned int one_idx = 0 if bdd[nnodes1, 3] else 1
    cdef unsigned int i,j

    # so you can read the updates - don't uncomment
    #cdef DINT_t label = bdd[i,0]
    #cdef DINT_t high_idx = bdd[i,1]
    #cdef DINT_t low_idx = bdd[i,2]
    #cdef DINT_t is_compl = bdd[i,3]
    #cdef DFLOAT_t node_prob = bdd_param_idx[j, label]
    #cdef DINT_t low_comp    = bdd[low_idx, 3] # low.IsComplement()

    for i in range(nnodes):
        if bdd[i,0] == -1: # root
            for j in range(nobs):
                beta[j,i,one_idx] = 1
        else: # non-root
            for j in range(nobs):
                beta[j, i, 1] = (prob[j,bdd[i,0]]*beta[j, bdd[i,1], 1]+
                    (1-prob[j,bdd[i,0]])*beta[j, bdd[i,2], 1-bdd[bdd[i,2],3]])
                beta[j, i, 0] = (prob[j,bdd[i,0]]*beta[j, bdd[i,1], 0]+
                    (1-prob[j,bdd[i,0]])*beta[j, bdd[i,2], bdd[bdd[i,2],3]])
    return beta


def sample_bdd_plate_cy(
    np.ndarray[DINT_t, ndim=3]      plate,
    np.ndarray[DINT_t, ndim=2]      bdd,
    np.ndarray[DINT_t, ndim=1]      reps,
    np.ndarray[DFLOAT_t, ndim=3]    beta,
    np.ndarray[DFLOAT_t, ndim=2]    prob,
                                    x # TODO optimize to array
    ):
    """
    Sample a set of BDDs and update x. Sampling x means:
    
    1. :func:`backward_plate_cy`
    2. :func:`sample_bdd_plate_cy`
    
    :param plate:
    :type plate: see :class:`formula_gen.PBPlate`
    :param bdd:
    :type bdd: see :class:`knowledge_compilation.BDD`
    :param reps:
    :type reps: see :class:`formula_gen.PBPlate`
    :param beta:
    :type beta: returned by :func:`backward_plate_cy`
    :param prob: probabilities of the propositional variables in the BDDs
    :type prob: returned by :func:`reparam_cy` (spoiler NXM ndarray)
    :param x: datastructure for x
    :type x: list of NDist ndarrays, each of shape n_vars X n_cat (see :class:`formula_gen.PBModel`)
    :rtype: None
    """

    cdef unsigned int i,j,ii,jj,kk,N
    cdef unsigned int nobs   = reps.shape[0]
    cdef unsigned int nnodes = bdd.shape[0]
    cdef unsigned int nnodes1 = nnodes-1
    cdef np.ndarray[DINT_t, ndim=2] bdd_reps = np.zeros([nnodes, nobs],
        dtype=DINT)
    cdef float p1
    cdef:
       np.ndarray[np.double_t, ndim=1] p = np.zeros([2], dtype=np.double)
       size_t K = p.shape[0]
       np.ndarray[np.uint32_t, ndim=1] n = np.empty_like(p, dtype='uint32')
    for j in range(nobs):
        bdd_reps[nnodes1,j] = reps[j]
    for i in range(nnodes1, -1, -1):
        if bdd[i,0] == -1:
            continue
        for j in range(nobs):
            p1 = prob[j,bdd[i,0]]*beta[j,bdd[i,1],1]/beta[j,i,1]
            p[0] = p1
            p[1] = 1-p1
            N = bdd_reps[i,j]
            # void gsl_ran_multinomial (const gsl_rng * r, size_t K,
            # unsigned int N, const double p[], unsigned int n[])
            gsl_ran_multinomial(r, K, N, <double*> p.data,
                <unsigned int *> n.data)
            # samples are in n, update counts
            ii = plate[j,bdd[i,0]][0]
            jj = plate[j,bdd[i,0]][1]
            kk = plate[j,bdd[i,0]][2]
            x[ii][jj,kk] += n[0]
            if n[1]>0 and x[ii].shape[1]-2 == kk:
                x[ii][jj,kk+1] += n[1]
            # update reps
            bdd_reps[bdd[i,1],j] += n[0]
            bdd_reps[bdd[i,2],j] += n[1]

# TODO this one needs heavy optimization
def reparam_cy(
    cat_list,       # list of motherfucking dictionaries
    theta,          # list of arrays
    np.ndarray[DINT_t, ndim=3] plate,
    np.ndarray[DINT_t, ndim=2] kid):

    """
    Compute the parameters of the propositional variables in the BDDs based on theta.
    
    :param cat_list: 
    :type cat_list: see :class:`formula_gen.PBPlate`
    :param theta:
    :type theta: list of NDist ndarrays, each of shape n_vars X n_cat (see :class:`formula_gen.PBModel`)
    :param plate:
    :type plate:  see :class:`formula_gen.PBPlate`
    :param kid:
    :type kid: see :class:`formula_gen.PBPlate`
    :rtype: ndarray of shape N X M    
    """

    cdef:
        unsigned int n,m,i,j
        unsigned int N = plate.shape[0]
        unsigned int M = plate.shape[1]
        np.ndarray[DFLOAT_t, ndim=2] bdd_param_plate = np.zeros([N,M],
            dtype=DFLOAT)

    for n in range(N):
        P = copy.copy(cat_list[n])
        for (i,j),L in cat_list[n].iteritems():
            if len(L) == 1:
                P[(i,j)] = [theta[i][j,L[0]]]
            else:
                P[(i,j)] = reparam_row_cy(theta[i][j,L].reshape(-1))
        for m in range(M):
                bdd_param_plate[n,m] = P[(plate[n,m,0],plate[n,m,1])][kid[n,m]]
    return bdd_param_plate

cdef reparam_row_cy(
    np.ndarray[DFLOAT_t, ndim=1] params):

    cdef:
        unsigned int l1 = params.shape[0]-1
        unsigned int i,j
        float prod
        np.ndarray[DFLOAT_t, ndim=1] reparams = np.zeros([l1])

    for j in range(l1):
        prod = 1
        for i in range(j):
            prod *= 1-reparams[i]
        reparams[j] = params[j]/prod
    return reparams

