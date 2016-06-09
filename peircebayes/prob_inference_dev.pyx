"""
:synopsis: Cython module for inner loop methods for :mod:`prob_inference` in a :class:`formula_gen.PBModel`.
"""

from __future__ import division

import copy

cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np

from cpython cimport bool

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
    np.ndarray[DFLOAT_t, ndim=2]    prob):

    """
    Compute backward probability for a set of identical BDDs (with different parameters).

    :param bdd:
    :type bdd: see :class:`knowledge_compilation.BDD`
    :param prob:
    :type prob: a float ndarray of shape [N, M]
    :rtype: ndarray of shape [N, NNodes, 2]
    """

    cdef unsigned int nobs   = prob.shape[0]
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
        if bdd[i,0] == -1: # leaf
            for j in range(nobs):
                beta[j,i,one_idx] = 1
        else: # non-leaf
            for j in range(nobs):
                beta[j, i, 1] = (prob[j,bdd[i,0]]*beta[j, bdd[i,1], 1]+
                    (1-prob[j,bdd[i,0]])*beta[j, bdd[i,2], 1-bdd[bdd[i,2],3]])
                beta[j, i, 0] = (prob[j,bdd[i,0]]*beta[j, bdd[i,1], 0]+
                    (1-prob[j,bdd[i,0]])*beta[j, bdd[i,2], bdd[bdd[i,2],3]])
    return beta

def backward_ob_cy(
    np.ndarray[DINT_t, ndim=2]      bdd,
    np.ndarray[DFLOAT_t, ndim=1]    prob):

    """
    Compute backward probability for a single BDD.

    :param bdd:
    :type bdd: see :class:`knowledge_compilation.BDD`
    :param prob:
    :type prob: a float ndarray of shape [M,1]
    :rtype: ndarray of shape [NNodes, 2]
    """
    cdef:
        unsigned int nnodes = bdd.shape[0]
        np.ndarray[DFLOAT_t, ndim=2] beta = np.zeros([nnodes, 2],
            dtype=DFLOAT)
        # 0 if root.IsComplement() else 1
        unsigned int nnodes1 = nnodes-1
        unsigned int one_idx = 0 if bdd[nnodes1, 3] else 1
        unsigned int i

    # so you can read the updates - don't uncomment
    #cdef DINT_t label = bdd[i,0]
    #cdef DINT_t high_idx = bdd[i,1]
    #cdef DINT_t low_idx = bdd[i,2]
    #cdef DINT_t is_compl = bdd[i,3]
    #cdef DFLOAT_t node_prob = bdd_param_idx[j, label]
    #cdef DINT_t low_comp    = bdd[low_idx, 3] # low.IsComplement()

    for i in range(nnodes):
        if bdd[i,0] == -1: # leaf
            beta[i,one_idx] = 1
        else: # non-leaf
            beta[i, 1] = (prob[bdd[i,0]]*beta[bdd[i,1], 1]+
                (1-prob[bdd[i,0]])*beta[bdd[i,2], 1-bdd[bdd[i,2],3]])
            beta[i, 0] = (prob[bdd[i,0]]*beta[bdd[i,1], 0]+
                (1-prob[bdd[i,0]])*beta[bdd[i,2], bdd[bdd[i,2],3]])
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
                # TODO The condition is bugged
                # need kk == cat_list[j][(ii,jj)][-2]
                # and kk+1 = cat_list[j][(ii,jj)][-1]
                x[ii][jj,kk+1] += n[1]
            # update reps
            bdd_reps[bdd[i,1],j] += n[0]
            bdd_reps[bdd[i,2],j] += n[1]

def sample_bdd_ob_cy(
    np.ndarray[DINT_t, ndim=2]      plate,
    np.ndarray[DINT_t, ndim=2]      bdd,
    np.ndarray[DFLOAT_t, ndim=2]    beta,
    np.ndarray[DFLOAT_t, ndim=1]    prob,
                                    x # TODO optimize to array
    ):
    """
    :param plate:
    :type plate: see :class:`formula_gen.PBPlate`, for one ob
    :param bdd:
    :type bdd: see :class:`knowledge_compilation.BDD`
    :param beta:
    :type beta: returned by :func:`backward_ob_cy`
    :param prob: probabilities of the propositional variables in the BDDs
    :type prob: returned by :func:`reparam_cy` for one ob (spoiler M ndarray)
    :param x: datastructure for x
    :type x: list of NDist ndarrays, each of shape n_vars X n_cat (see :class:`formula_gen.PBModel`)
    :rtype: None
    """

    l = []
    cdef:
        unsigned int i,j,ii,jj,kk,N
        unsigned int nnodes = bdd.shape[0]
        unsigned int nnodes1 = nnodes-1
        float p1
        np.ndarray[np.double_t, ndim=1] p = np.zeros([2], dtype=np.double)
        size_t K = p.shape[0]
        np.ndarray[np.uint32_t, ndim=1] n = np.empty_like(p, dtype='uint32')
    i = nnodes1
    while True:
        if bdd[i,0] == -1:
            break
        p1 = prob[bdd[i,0]]*beta[bdd[i,1],1]/beta[i,1]
        p[0] = p1
        p[1] = 1-p1
        N = 1
        # void gsl_ran_multinomial (const gsl_rng * r, size_t K,
        # unsigned int N, const double p[], unsigned int n[])
        gsl_ran_multinomial(r, K, N, <double*> p.data,
            <unsigned int *> n.data)
        # samples are in n, update counts
        ii = plate[bdd[i,0]][0]
        jj = plate[bdd[i,0]][1]
        kk = plate[bdd[i,0]][2]
        x[ii][jj,kk] += n[0]
        if n[1]>0 and x[ii].shape[1]-2 == kk:
            # TODO The condition is bugged
            # need kk == cat_list[j][(ii,jj)][-2]
            # and kk+1 = cat_list[j][(ii,jj)][-1]
            x[ii][jj,kk+1] += n[1]
        if n[0]>0:
            l.append((ii,jj,kk))
            i = bdd[i,1]
        else:
            if x[ii].shape[1]-2 == kk:
                l.append((ii,jj,kk+1))
            i = bdd[i,2]
    return l

# TODO this one needs heavy optimization
def reparam_cy(
    cat_list,       # list of motherfucking dictionaries
    theta,          # list of arrays
    np.ndarray[DINT_t, ndim=3] plate,
    np.ndarray[DINT_t, ndim=2] kid
):

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


def reparam_update_cy(
    cat_list,       # list of motherfucking dictionaries
    theta,          # list of arrays
    np.ndarray[DINT_t, ndim=3] plate,
    np.ndarray[DINT_t, ndim=2] kid,
    l, #list of (i,j,k)
    np.ndarray[DFLOAT_t, ndim=2] bdd_param_plate, # NXM array
    unsigned int n # integer
):

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
    """

    cdef:
        unsigned int m,i,j,k
        unsigned int N = plate.shape[0]
        unsigned int M = plate.shape[1]
        #np.ndarray[DFLOAT_t, ndim=2] bdd_param_plate = np.zeros([N,M],
        #    dtype=DFLOAT)


    #P = copy.copy(cat_list[n])
    #for (i,j,k) in l:
    #    if (i,j) in cat_list[n]:
    #        L = cat_list[n][(i,j)]
    #        if len(L) == 1:
    #            P[(i,j)] = [theta[i][j,L[0]]]
    #        else:
    #            P[(i,j)] = reparam_row_cy(theta[i][j,L].reshape(-1))
    #l_ai = [(a,i) for a,i,k in l]
    #for m in range(M):
    #    if (plate[n,m,0], plate[n,m,1]) in l_ai:
    #        bdd_param_plate[n,m] = P[(plate[n,m,0],plate[n,m,1])][kid[n,m]]
    P = copy.copy(cat_list[n])
    for (i,j),L in cat_list[n].iteritems():
        if len(L) == 1:
            P[(i,j)] = [theta[i][j,L[0]]]
        else:
            P[(i,j)] = reparam_row_cy(theta[i][j,L].reshape(-1))
    for m in range(M):
        bdd_param_plate[n,m] = P[(plate[n,m,0],plate[n,m,1])][kid[n,m]]



def reparam_row_wrap_cy(
    np.ndarray[DFLOAT_t, ndim=1] params):
    return reparam_row_cy(params)

cdef reparam_row_cy(
    np.ndarray[DFLOAT_t, ndim=1] params):

    cdef:
        unsigned int l1 = params.shape[0]-1
        unsigned int i,j
        float prod
        np.ndarray[DFLOAT_t, ndim=1] reparams = np.zeros([l1])
    prod = 1
    for j in range(l1):
        reparams[j] = params[j]/prod
        prod *= 1-reparams[j]

    return reparams

def cgs_iter_cy(self, obs_l):
    for (plate_idx, i, ob) in obs_l:
        pb_plate = self.pb_model.plates[plate_idx]
        curr_l = self.obs_d[(plate_idx, i, ob)]
        for aa,ii,ll in curr_l:
            self.x[aa][ii,ll] -=1
        if self.l is not None:
            self.update_theta(curr_l+self.l+self.old_l, plate_idx, i)
        betas = self.backward_ob(pb_plate.bdd, self.bdd_param[plate_idx][i,:])
        self.l = self.sample_bdd_ob(pb_plate.plate[i,:,:], pb_plate.bdd, betas, self.bdd_param[plate_idx][i,:])
        self.old_l = copy.copy(self.obs_d[(plate_idx, i, ob)])
        self.obs_d[(plate_idx, i, ob)] = copy.copy(self.l)

cdef class CyPBModel:
    ## attributes
    # former attributes of inference and pb_model
    cdef list alpha_post, alpha_tiled, bdd_param, distrib_cat_vars, theta, x
    # former pb_plate
    cdef list bdds, cat_lists, kids, plates
    # aux
    cdef iplates

    def __cinit__(self, list distrib_cat_vars, list distrib_args,
        list bdds, list cat_lists, list kids, list plates):
        self.distrib_cat_vars = distrib_cat_vars
        # set x
        self.reset_x()
        self.alpha_tiled = [np.tile(distrib_arg, (n_vars,1))
            for distrib_arg, (_,n_vars) in zip(distrib_args, distrib_cat_vars)]

        self.bdds = bdds
        self.cat_lists = cat_lists
        self.kids = kids
        self.plates = plates

        # define iplates
        self.iplates = []
        A = len(distrib_cat_vars)
        cdef:
            list iplate = []
        for plate, cat_list in zip(plates, cat_lists):
            N,M = plate.shape[0:2]
            iplate = range(N)
            for n in xrange(N):
                d_ai = {}
                cat_list[n]
                for m in xrange(M):
                    a,i,_ = plate[n,m]
                    if (a,i) in d_ai:
                        d_ai[(a,i)][1].append(m)
                    else:
                        d_ai[(a,i)] = [cat_list[n][(a,i)], [m]]
                iplate[n] = [(a,i,L,Lm) for (a,i), (L,Lm) in d_ai.iteritems()]
            self.iplates.append(iplate)
        #print(self.iplates)

    cdef reset_x(self):
        cdef:
            unsigned int n_cat, n_vars
        self.x = []
        for n_cat,n_vars in self.distrib_cat_vars:
            self.x.append(np.zeros((n_vars, n_cat)))

    cdef set_theta_mean(self):
        self.alpha_post = [alpha_d+x_distrib
            for x_distrib, alpha_d in zip(self.x,self.alpha_tiled)]
        self.theta=[]
        for alpha_d in self.alpha_post:
            self.theta.append(alpha_d/np.sum(alpha_d, axis=1)[:,None])
        # no frozen
        # no clip
        self.reparam()

    cdef reparam(self):
        #import time
        cdef:
            unsigned int i
        self.bdd_param = range(len(self.bdds))
        for i in range(len(self.bdds)):
            if self.bdds[i] is None:
                self.bdd_param[i] = []
            else:
                #start_test = time.clock()
                #self.bdd_param[i] = reparam_cy(self.cat_lists[i], self.theta,
                #    self.plates[i], self.kids[i])
                self.bdd_param[i] = self.reparam_cy2(i, self.kids[i])
                #end_test = time.clock()
                #print('reparam time: {}'.format(end_test-start_test))

    def reparam_cy2(self,
        unsigned int it,
        np.ndarray[DINT_t, ndim=2] kid):
        cdef:
            unsigned int n,m,a,i,j
            unsigned int N = self.plates[it].shape[0]
            unsigned int M = self.plates[it].shape[1]
            np.ndarray[DFLOAT_t, ndim=2] bdd_param_plate = np.zeros([N,M],
                dtype=DFLOAT)
            list v,L,Lm
            float prod,val,k

        for n in range(N):
            for (a,i,L,Lm) in self.iplates[it][n]:
                if len(L) == 1:
                    v = [self.theta[a][i,L[0]]]
                else:
                    v = []
                    prod = 1
                    for k in self.theta[a][i,L]:
                        val = k/prod
                        prod *= 1-val
                        v.append(val)
                for m in Lm:
                    bdd_param_plate[n,m] = v[kid[n,m]]
        return bdd_param_plate


def test(distrib_cat_vars, distrib_args,
    bdds, cat_lists, kids, plates):
    import time
    start_init = time.clock()
    pb = CyPBModel(distrib_cat_vars, distrib_args,
        bdds, cat_lists, kids, plates)
    end_init = time.clock()
    print('init time: {}'.format(end_init-start_init))
    start_test = time.clock()
    pb.set_theta_mean()
    end_test = time.clock()
    print('test time: {}'.format(end_test-start_test))



