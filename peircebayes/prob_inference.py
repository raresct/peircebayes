#!/usr/bin/env python

"""
:synopsis: Module for probabilistic inference on a :class:`formula_gen.PBModel`.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import bernoulli
from scipy.special import gammaln
import matplotlib.pyplot as plt
import itertools
import copy
#import gc
import psutil

import knowledge_compilation

from prob_inference_dev import (backward_plate_cy, sample_bdd_plate_cy,
    reparam_cy)

# logging stuff
import logging
from logging import debug as logd

# timing stuff
import time

# inference class
class PBInference:
    """
    Container of inference methods for a :class:`formula_gen.PBModel`.
    """
    def __init__(self, pb_model):
        # input
        self.pb_model       = pb_model
        # important attributes
        self.x              = None
        self.theta          = None
        # other
        self.bdd_param      = None
        self.theta_avg      = None
        self.beta           = None
        self.alpha_tiled    = [np.tile(distrib[0], (distrib[2],1))
                                for distrib in pb_model.distribs]
        self.alpha          = [distrib[0] for distrib in pb_model.distribs]
        self.symmetric      = [np.all(alpha_d==alpha_d[0])
                                for alpha_d in self.alpha]

    def backward_plates(self):
        """
        Compute backward probabilities for all plates.
        
        :rtype: list of els. returned by :func:`prob_inference_dev.backward_plate_cy`
        """
        return [self.backward_plate4(pb_plate, plate_idx)
            for (plate_idx, pb_plate) in enumerate(self.pb_model.plates)]

    def backward_plate4(self, pb_plate, plate_idx):
        return backward_plate_cy(pb_plate.bdd, pb_plate.plate,
            self.bdd_param[plate_idx])

    def backward_plate3(self, pb_plate, plate_idx):
        bdd = pb_plate.bdd
        plate = pb_plate.plate
        #logd(plate.shape[0])
        #logd(bdd.shape[0])
        beta = np.zeros((plate.shape[0], bdd.shape[0], 2))
        one_idx = 0 if bdd[-1][3] else 1 # 0 if root.IsComplement() else 1
        for i,bdd_node in enumerate(bdd):
            if bdd_node[0] == -1:
                beta[:,i,one_idx] = np.ones(plate.shape[0])
            else:
                for plate_row in range(beta.shape[0]):
                    label, high_idx, low_idx, _  = bdd_node
                    node_prob   = self.bdd_param[plate_idx][plate_row, label]
                    low_comp    = bdd[low_idx, 3] # low.IsComplement()
                    beta[plate_row, i, 1] = (
                        node_prob*beta[plate_row, high_idx, 1]+
                        (1-node_prob)*beta[plate_row, low_idx, int(not low_comp)] )
                    beta[plate_row, i, 0] = (
                        node_prob*beta[plate_row, high_idx, 0]+
                        (1-node_prob)*beta[plate_row, low_idx, int(low_comp)] )
        #print beta
        return beta

    def backward_plate2(self, pb_plate, plate_idx):
        bdd = pb_plate.bdd
        plate = pb_plate.plate
        #logd(plate.shape[0])
        #logd(len(bdd.h_nodes))
        beta = np.zeros((plate.shape[0], len(bdd.h_nodes), 2))
        one_idx = 0 if bdd.root.IsComplement() else 1
        beta[:,0,one_idx] = np.ones(plate.shape[0])
        for i, (curr_node, h_curr, idx_curr) in enumerate(zip(
            bdd.nodes[1:], bdd.h_nodes[1:], bdd.idx_nodes[1:])):
            i += 1
            high, low = curr_node.T(), curr_node.E()
            h_high, h_low = map(myhash, [high, low])
            i_high = bdd.h_nodes.index(h_high)
            i_low = bdd.h_nodes.index(h_low)
            for plate_row in range(plate.shape[0]):
                curr_theta = self.bdd_param[plate_idx][plate_row, idx_curr]
                if low.IsComplement():
                    beta[plate_row, i, 1] = (
                        curr_theta*beta[plate_row, i_high, 1]+
                        (1-curr_theta)*beta[plate_row, i_low, 0])
                    beta[plate_row, i, 0] = (
                        curr_theta*beta[plate_row, i_high, 0]+
                        (1-curr_theta)*beta[plate_row, i_low, 1])
                else:
                    beta[plate_row, i ,1] = (
                        curr_theta*beta[plate_row, i_high, 1]+
                        (1-curr_theta)*beta[plate_row, i_low, 1])
                    beta[plate_row, i ,0] = (
                        curr_theta*beta[plate_row, i_high, 0]+
                        (1-curr_theta)*beta[plate_row, i_low, 0])
        #print beta
        return beta

    def backward_plate(self, pb_plate, plate_idx):
        bdd = pb_plate.bdd
        plate = pb_plate.plate
        h_one, h_root = [bdd.h_nodes[i] for i in [0,-1]]
        beta_init = {(h_one,0):0, (h_one,1):1}
        if bdd.root.IsComplement():
            beta_init = {(h_one,0):1, (h_one,1):0}
        beta = [beta_init.copy() for i in range(len(plate))]
        for curr_node, h_curr, idx_curr in zip(
            bdd.nodes[1:], bdd.h_nodes[1:], bdd.idx_nodes[1:]):
            high, low = curr_node.T(), curr_node.E()
            h_high, h_low = map(self.myhash, [high, low])
            for plate_row in range(plate.shape[0]):
                curr_theta = self.bdd_param[plate_idx][plate_row, idx_curr]
                if low.IsComplement():
                    beta[plate_row][(h_curr,1)] = (
                        curr_theta*beta[plate_row][(h_high,1)]+
                        (1-curr_theta)*beta[plate_row][(h_low, 0)])
                    beta[plate_row][(h_curr,0)] = (
                        curr_theta*beta[plate_row][(h_high,0)]+
                        (1-curr_theta)*beta[plate_row][(h_low, 1)])
                else:
                    beta[plate_row][(h_curr,1)] = (
                        curr_theta*beta[plate_row][(h_high,1)]+
                        (1-curr_theta)*beta[plate_row][(h_low, 1)])
                    beta[plate_row][(h_curr,0)] = (
                        curr_theta*beta[plate_row][(h_high,0)]+
                        (1-curr_theta)*beta[plate_row][(h_low, 0)])
        #print beta
        return beta

    # TODO check these
    # bdd params, bdd reps
    def sample_bdd_plates(self):
        """
        Sample x method. Calls :meth:`backward_plates` and for each plate :func:`prob_inference_dev.sample_bdd_plate_cy`. Updates x.        
        
        :rtype: None
        """

        self.reset_x()
        #mem1 = psutil.virtual_memory().percent
        #logd('before beta: {}'.format(mem1))
        start = time.clock()
        betas = self.backward_plates()
        end = time.clock()
        logd('beta time: {} s'.format(end-start))

        #mem2 = psutil.virtual_memory().percent
        #logd('after beta: {}'.format(mem2))


        #start = time.clock()
        for (plate_idx, (pb_plate, beta)) in enumerate(zip(
            self.pb_model.plates, betas)):
            self.sample_bdd_plate3(pb_plate, beta, plate_idx)
        #logd(self.x)
        #end = time.clock()
        #logd('sample bdd time: {} s'.format(end-start))
        
    def sample_bdd_plate3(self, pb_plate, beta, plate_idx):
        plate = pb_plate.plate
        bdd = pb_plate.bdd
        reps = pb_plate.reps
        prob = self.bdd_param[plate_idx]
        sample_bdd_plate_cy(plate, bdd, reps, beta, prob, self.x)

    def sample_bdd_plate2(self, pb_plate, beta, plate_idx):
        bdd = pb_plate.bdd
        reps = pb_plate.reps
        NNodes = bdd.shape[0]
        NParamSets = reps.shape[0]
        self.bdd_reps = np.zeros((NNodes, NParamSets))
        self.bdd_reps[-1,:] = reps
        for i in range(bdd.shape[0]-1, -1, -1):
            bdd_node = bdd[i]
            if bdd_node[0] == -1:
                continue
            # must compute
            # TODO precompute as much as possible
            reps = self.bdd_reps[i,:].reshape(-1,1)
            try:
                ber_args = np.hstack(
                # this is p1
                ((self.bdd_param[plate_idx][:, bdd_node[0]]*
                    beta[:, bdd_node[1], 1]/ beta[:, i, 1])
                    .reshape(-1,1),
                # stacked with reps
                reps))
            except Exception as e:
                logd(e)
            logd(ber_args)
            samples = np.apply_along_axis(
                lambda x: np.random.multinomial(x[1], [x[0], None]),
                1, ber_args)
            # TODO figure out how to update self.x with vectorization
            # idea: group by i in (i,j,k) and work with the matrices separately
            #print self.x
            np.apply_along_axis(self.update_x_high_count,1,
                np.hstack((pb_plate.plate[:, bdd_node[0]], samples[:,[0]]))
                .astype(int))
            np.apply_along_axis(self.update_x_low_count,1,
                np.hstack((pb_plate.plate[:, bdd_node[0]], samples[:,[1]]))
                .astype(int))
            start_upd_reps = time.clock()
            self.bdd_reps[bdd_node[1],:] += samples[:,[0]].reshape(-1)
            self.bdd_reps[bdd_node[2],:] += samples[:,[1]].reshape(-1)
            end_upd_reps = time.clock()
            #logd('per update time: {} s'.format(end_upd_reps-start_upd_reps))

    def sample_bdd_plate(self, pb_plate, beta, plate_idx):
        # bdd_reps is a dict of key=hash(node), val = (index, repetitions)
        # bdd_reps is an array of shape (NNodes, NParamSets)
        # bdd_reps[node_number_bft, plate_row] = n_reps
        bdd = pb_plate.bdd
        reps = pb_plate.reps
        NNodes = len(bdd.nodes[1:])
        NParamSets = reps.shape[0]
        self.bdd_reps = np.zeros((NNodes, NParamSets))
        # 0 for root
        self.bdd_reps[0,:] = reps
        nodes = list(reversed(bdd.nodes[1:]))
        h_nodes = list(reversed(bdd.h_nodes[1:]))
        i = 0
        for (node_no_bft, (curr_node,h_curr)) in enumerate(zip(
            nodes,h_nodes)):
            #start = time.clock()
            self.sample_node_4(node_no_bft, curr_node, h_curr,
                pb_plate, beta, plate_idx, h_nodes)
            #end = time.clock()
            #logd('per node time: {} s'.format(end-start))
            #for j in range(self.bdd_reps.shape[0]):
            #    print self.bdd_reps[j,:]
            i+=1
            #if i>4:
            #    return
        #print self.x

    def update_x_high_count(self, samples):
        i,j,k,samples_high = samples
        self.x[i][j,k] += samples_high
        return 0

    def update_x_low_count(self, samples):
        i,j,k,samples_low = samples
        if samples_low>0 and self.theta[i].shape[1]-2 == k:
            self.x[i][j,k+1] += samples_low
        return 0

    def sample_node_4(self, node_no_bft, curr_node, h_curr,
        pb_plate, beta, plate_idx, h_nodes):
        high, low = curr_node.T(), curr_node.E()
        h_high, h_low = map(myhash, [high, low])
        i_high = pb_plate.bdd.h_nodes.index(h_high)
        i_low = pb_plate.bdd.h_nodes.index(h_low)
        i_curr = pb_plate.bdd.h_nodes.index(h_curr)
        # must compute
        # TODO precompute as much as possible
        reps = self.bdd_reps[node_no_bft,:].reshape(-1,1)
        try:
            ber_args = np.hstack(
            # this is p1

            ((self.bdd_param[plate_idx][:, node_no_bft]*
                beta[:, i_high, 1]/ beta[:, i_curr, 1])
                .reshape(-1,1),
            # stacked with reps
            reps))
        except Exception as e:
            #print len(h_nodes)
            #print self.bdd_param[0].shape
            #print plate_idx
            #print node_no_bft
            #print i_high
            #print beta[:,i_curr,1]
            logd(e)
        samples = np.apply_along_axis(
            lambda x: np.random.multinomial(x[1], [x[0], None]),
            1, ber_args)
        # TODO figure out how to update self.x with vectorization
        # idea: group by i in (i,j,k) and work with the matrices separately
        #print self.x
        np.apply_along_axis(self.update_x_high_count,1,
            np.hstack((pb_plate.plate[:, node_no_bft], samples[:,[0]]))
            .astype(int))
        np.apply_along_axis(self.update_x_low_count,1,
            np.hstack((pb_plate.plate[:, node_no_bft], samples[:,[1]]))
            .astype(int))
        start_upd_reps = time.clock()
        try:
            node_no_high = h_nodes.index(h_high)
            self.bdd_reps[node_no_high,:] += samples[:,[0]].reshape(-1)
        except:
            pass # leaf node
        try:
            node_no_low = h_nodes.index(h_low)
            self.bdd_reps[node_no_low,:] += samples[:,[1]].reshape(-1)
        except:
            pass # leaf node
        end_upd_reps = time.clock()
        #logd('per update time: {} s'.format(end_upd_reps-start_upd_reps))


    def sample_node_3(self, node_no_bft, curr_node, h_curr,
        pb_plate, beta, plate_idx, h_nodes):
        # TODO this doesn't work currently, find out why and fix it
        # no more curr_reps we need bdd_reps[row] instead
        high, low = curr_node.T(), curr_node.E()
        h_high, h_low = map(self.myhash, [high, low])
        #i_high = pb_plate.bdd.h_nodes.index(h_high)
        #i_low = pb_plate.bdd.h_nodes.index(h_low)
        # must compute
        # TODO precompute as much as possible
        reps = self.bdd_reps[node_no_bft,:].reshape(-1,1)
        ber_args = np.hstack(
        # this is p1
        ((self.bdd_param[plate_idx][:, node_no_bft]*
            np.array([beta_row[(h_high,1)] for beta_row in beta])
            /np.array([beta_row[(h_curr,1)] for beta_row in beta]))
            .reshape(-1,1),
        # stacked with reps
        reps))
        samples = np.apply_along_axis(
            lambda x: np.random.multinomial(x[1], [x[0], None]),
            1, ber_args)
        # TODO figure out how to update self.x with vectorization
        # idea: group by i in (i,j,k) and work with the matrices separately
        #print self.x
        np.apply_along_axis(self.update_x_high_count,1,
            np.hstack((pb_plate.plate[:, node_no_bft], samples[:,[0]]))
            .astype(int))
        np.apply_along_axis(self.update_x_low_count,1,
            np.hstack((pb_plate.plate[:, node_no_bft], samples[:,[1]]))
            .astype(int))
        start_upd_reps = time.clock()
        try:
            node_no_high = h_nodes.index(h_high)
            self.bdd_reps[node_no_high,:] += samples[:,[0]].reshape(-1)
        except:
            pass # leaf node
        try:
            node_no_low = h_nodes.index(h_low)
            self.bdd_reps[node_no_low,:] += samples[:,[1]].reshape(-1)
        except:
            pass # leaf node
        end_upd_reps = time.clock()
        #logd('per update time: {} s'.format(end_upd_reps-start_upd_reps))


    def sample_node_2(self, node_no_bft, curr_node, h_curr,
        pb_plate, beta, plate_idx, h_nodes):
        # debug
        import warnings
        warnings.filterwarnings("error")

        # no more curr_reps we need bdd_reps[row] instead
        high, low = curr_node.T(), curr_node.E()
        h_high, h_low = map(self.myhash, [high, low])

        # must compute
        # TODO precompute as much as possible
        curr_theta = self.bdd_param[plate_idx][:, node_no_bft]
        #print curr_theta.shape
        prob_curr_node = np.array([beta_row[(h_curr,1)] for beta_row in beta])
        #print prob_curr_node.shape
        prob_child_1 = np.array([beta_row[(h_high,1)] for beta_row in beta])
        #print prob_child_1.shape
        # P(N sampled 1) = P(N)*beta(Child^1[N])/beta(N)
        #eps = 10**-300
        #curr_theta = curr_theta.clip(min=eps)
        #prob_child_1 = prob_child_1.clip(min=eps)
        #prob_curr_node = prob_curr_node.clip(min=eps)
        try:
            p1 = curr_theta*prob_child_1/prob_curr_node
        except:
            print prob_curr_node, curr_theta*prob_child_1, node_no_bft, plate_idx
            p1 = np.log(curr_theta)+np.log(prob_child_1)-np.log(prob_curr_node)
            print 'exp p1'
            print np.exp(p1)
            return
        p1 = p1.reshape(p1.shape[0], 1)
        reps = self.bdd_reps[node_no_bft,:]
        reps = reps.reshape(reps.shape[0],1)
        ber_args = np.hstack((p1, reps))
        #print ber_args.shape
        #print p1[:10]
        #print reps[:10]
        samples = np.apply_along_axis(
            lambda x: np.random.multinomial(x[1], [x[0], None]),
            1, ber_args)
        samples_high = samples[:,[0]]
        samples_low = samples[:,[1]]
        #print samples_low[:10]
        #print
        #print samples_high[:10]
        #print samples_high_col.shape
        #print formula.plate[:, node_no_bft].shape
        #print np.hstack((samples_high_col, formula.plate[:, node_no_bft])).shape

        # TODO figure out how to update self.x with vectorization
        # idea: group by i in (i,j,k) and work with the matrices separately
        #print self.x
        np.apply_along_axis(self.update_x_high_count,1,
            np.hstack((pb_plate.plate[:, node_no_bft], samples_high))
            .astype(int))
        #print self.x[0]
        np.apply_along_axis(self.update_x_low_count,1,
            np.hstack((pb_plate.plate[:, node_no_bft], samples_low))
            .astype(int))

        #node_no_high = h_nodes.index(h_high)
        #reps_high = self.bdd_reps[node_no_high,:]
        #reps_high = reps_high.reshape(reps_high.shape[0],1)
        #reps_high += samples_high
        # update reps
        start_upd_reps = time.clock()
        try:
            node_no_high = h_nodes.index(h_high)
            reps_high = self.bdd_reps[node_no_high,:]
            reps_high = reps_high.reshape(reps_high.shape[0],1)
            reps_high += samples_high
        except:
            pass # leaf node
        try:
            node_no_low = h_nodes.index(h_low)
            reps_low = self.bdd_reps[node_no_low,:]
            reps_low = reps_low.reshape(reps_low.shape[0],1)
            reps_low += samples_low
        except:
            pass # leaf node
        end_upd_reps = time.clock()
        #logd('per update time: {} s'.format(end_upd_reps-start_upd_reps))


    def sample_node_1(self, node_no_bft, curr_node, h_curr,
        formula, beta, plate_idx, h_nodes):
        # no more curr_reps we need bdd_reps[row] instead
        high, low = curr_node.T(), curr_node.E()
        h_high, h_low = map(self.myhash, [high, low])
        creps = self.bdd_reps[node_no_bft].tocoo()
        for _,plate_row, rep in itertools.izip(
            creps.row, creps.col, creps.data):
            i,j,k = formula.plate[plate_row, node_no_bft]
            #if plate_row==2 and idx_curr == 1:
            #    print i,j,k,idx_curr
            #    print self.theta[i].shape[1]
            #    return
            curr_theta = self.bdd_param[plate_idx][plate_row, node_no_bft]
            prob_curr_node = beta[plate_row][(h_curr,1)]
            prob_child_1 = beta[plate_row][(h_high,1)]
            # P(N sampled 1) = P(N)*beta(Child^1[N])/beta(N)
            p1 = curr_theta*prob_child_1/prob_curr_node
            ber_samples = bernoulli.rvs(p1, size=rep)
            samples_high = np.sum(ber_samples)
            self.x[i][j,k] += samples_high
            # deal with the last variable of the pds
            samples_low = rep-samples_high
            # bugged = shape-2
            if samples_low>0 and self.theta[i].shape[1]-2 == k:
                self.x[i][j,k+1] += samples_low
            # update reps
            start_upd_reps = time.clock()
            if samples_high > 0:
                try:
                    node_no_high = h_nodes.index(h_high)
                    self.bdd_reps[node_no_high,plate_row] += samples_high
                except:
                    pass # leaf node
            if samples_low > 0:
                try:
                    node_no_low = h_nodes.index(h_low)
                    self.bdd_reps[node_no_low,plate_row] += samples_low
                except:
                    pass # leaf node
            end_upd_reps = time.clock()
            #logd('per update time: {} s'.format(end_upd_reps-start_upd_reps))


    def reset_x(self):
        """
        Reset counts in x. Updates x.
        
        :rtype: None
        """
        distribs = self.pb_model.distribs
        self.x = [np.zeros((distrib[2], distrib[1])) for distrib in distribs]

    def gibbs_sampler_plates(self, n, burn_in, lag, track_ll, in_file, metric, algo):
        """
        Gibbs sampling. Updates x and theta, creates a theta_avg attribute and optionally saves metric to an output file.
        
        :param n: 
        :type n: see '-n' in :ref:`cl`
        :param burn_in: 
        :type burn_in: see '-b' in :ref:`cl`
        :param lag:
        :type lag: see '-l' in :ref:`cl`
        :param track_ll:
        :type track_ll: see '-t' in :ref:`cl`
        :param in_file:
        :type in_file: see 'in_file' in :ref:`cl`
        :param metric:
        :type metric: see '-m' in :ref:`cl`
        :param algo:
        :type algo: see '-a' in :ref:`cl`
        
        :rtype: None
        """
        start_gibbs = time.time()
        metrics = {
            'joint'     : lambda x: x.joint_collapsed(),
            'lda_ll'    : lambda x: x.likelihood_lda()
        }
        metric_f = metrics[metric]

        #ngc = gc.collect()
        #logd('Unreachable objects:{}'.format(ngc))

        n = n+burn_in # n is total number of iterations

        thetas = []
        if track_ll:
            lls = []

        self.first_sample = None
        # main loop
        for i in range(n):
            logd('Sampling iteration: {}'.format(i))
            if algo == 'ungibbs':
                start = time.clock()
                # SAMPLE THETA
                #logd('SAMPLE THETA')
                self.sample_theta()
                if not self.first_sample:
                    self.first_sample = self.theta
                end = time.clock()
                logd('sample theta time: {} s'.format(end-start))
                #logd('theta')
                #logd(self.theta)
                thetas.append(self.theta)
                # SAMPLE X
                #logd('SAMPLE X')
                start = time.clock()
                self.sample_bdd_plates()
                end = time.clock()
                logd('sample x time: {} s'.format(end-start))
            elif algo == 'amcmc':
                start = time.clock()
                # SAMPLE THETA
                #logd('SAMPLE THETA')
                self.set_theta_mean()
                if not self.first_sample:
                    self.first_sample = self.theta
                end = time.clock()
                logd('sample theta time: {} s'.format(end-start))
                #logd('theta')
                #logd(self.theta)
                thetas.append(self.theta)
                # SAMPLE X
                #logd('SAMPLE X')
                start = time.clock()
                self.sample_bdd_plates()
                end = time.clock()
                logd('sample x time: {} s'.format(end-start))
            if track_ll:
                lls.append(metric_f(self))
            #logd('x')
            #logd(self.x)
        #self.plot_phi(theta)
        end_gibbs = time.time()
        logd('Sampling time: {} s '.format(end_gibbs-start_gibbs))

        # POST PROCESS
        if burn_in is not None and int(burn_in)>=0:
            thetas = thetas[burn_in:]
        if lag is not None and int(lag)>0:
            thetas = [theta for i,theta in enumerate(thetas) if i%lag==0]
        logd('No. of samples: {}'.format(len(thetas)))

        # avg_theta
        sum_thetas = [np.zeros(theta_d.shape) for theta_d in self.theta]
        for theta_it in thetas:
            for i, theta_d in enumerate(theta_it):
                sum_thetas[i] += theta_d
        self.theta_avg = [theta_d/float(len(thetas)) for theta_d in sum_thetas]
        logd('theta avg')
        for theta_it in self.theta_avg:
            logd(theta_it)

        # ll tracking post-process
        if track_ll:
            self.plot_ll(lls, burn_in, in_file)
            np.savez('/tmp/peircebayes/lls', **{'lls':np.array(lls)})
            #with open('ll_aprob', 'w') as fout:
            #    for ll in lls:
            #        fout.write('{} '.format(ll))

    def plot_ll(self, lls, burn_in, in_file):
        #lls = lls[1:]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(list(range(len(lls))), lls)
        plt.xlabel('Iterations')
        plt.ylabel('Log Likelihood')
        plt.title('PB in_file={}'.format(in_file))
        if burn_in > 0:
            plt.axvline(x=burn_in, linewidth=1, color='red', label='Burn in')
            plt.legend(loc='lower right')
        # TODO plot lag as well
        plt.savefig('/tmp/peircebayes/ll.pdf', format='pdf')

    def set_theta_mean(self):
        distribs = self.pb_model.distribs
        if self.x is None:
            self.reset_x()
        alpha_post = [alpha_d+x_distrib
                for x_distrib, alpha_d in zip(self.x,self.alpha_tiled)]
        self.theta = [np.array(
            [alpha_row/np.sum(alpha_row) for alpha_row in alpha_d]
            ).reshape(alpha_d.shape)
            for alpha_d in alpha_post]
        for i, distrib in enumerate(distribs):
            self.theta[i] = np.tile(distrib[0], (distrib[2],1)) if distrib[3] else self.theta[i]
        # clip pls TODO find a better workaround
        eps = 10**-300
        self.theta = [theta_row.clip(min=eps) for theta_row in self.theta]
        self.reparam2()
        
    def sample_theta(self):
        """
        Main method for sampling theta. Updates theta. Also updates parameters of propositional variables in BDDs.
        
        :rtype: None 
        """
        #start = time.clock()
        distribs = self.pb_model.distribs
        if self.x is None:
            self.reset_x()
        
        # SAMPLE THETA
        alpha_post = [alpha_d+x_distrib
                for x_distrib, alpha_d in zip(self.x,self.alpha_tiled)]
        self.theta = [np.array(
            [np.random.dirichlet(alpha_row,1)[0] for alpha_row in alpha_d]
            ).reshape(alpha_d.shape)
            for alpha_d in alpha_post]
        # freeze distribs TODO do this as a logical operator on arrays
        for i, distrib in enumerate(distribs):
            self.theta[i] = np.tile(distrib[0], (distrib[2],1)) if distrib[3] else self.theta[i]
        # clip pls TODO find a better workaround
        eps = 10**-300
        self.theta = [theta_row.clip(min=eps) for theta_row in self.theta]

        #end = time.clock()
        #logd('alpha to theta: {} s '.format(end-start))
        #print 'theta d2'
        #print self.theta[0][1,:]
        #print 'phi w2'
        #print self.theta[1][:,1]
        #print 'phi w4'
        #print self.theta[1][:,3]

        # REPARAM
        # update bdd_param for all plates
        #start = time.clock()
        self.reparam2()
        #end = time.clock()
        #logd('reparam: {} s '.format(end-start))

    def reparam2(self):
        self.bdd_param = []
        for pb_plate in self.pb_model.plates:
            self.bdd_param.append(reparam_cy(
                pb_plate.cat_list, self.theta, pb_plate.plate, pb_plate.kid))

    def reparam(self):
        self.bdd_param = []
        for pb_plate in self.pb_model.plates:
            bdd_param_plate = np.zeros(pb_plate.plate.shape[:-1])
            for n,cat_d in enumerate(pb_plate.cat_list):
                # build reparams
                P = copy.copy(cat_d)
                for (i,j),L in cat_d.iteritems():
                    if len(L) == 1:
                        P[(i,j)] = [self.theta[i][j,L[0]]]
                    else:
                        P[(i,j)] = self.reparam_row(self.theta[i][j,L])
                # apply them
                for m in range(pb_plate.plate.shape[1]):
                    i,j,k = pb_plate.plate[n,m]
                    bdd_param_plate[n,m] = P[(i,j)][pb_plate.kid[n,m]]
            self.bdd_param.append(bdd_param_plate)

    def reparam_row(self, params):
        params = params.reshape(-1, 1)
        all_params = np.hstack((params, np.ones(params.shape)))
        for j in range(params.shape[0]-1):
            all_params[j,1] = all_params[j,0]/np.prod(1-all_params[:j,1])
        return all_params[:-1,1]

    def likelihood(self):
        return None

    def joint_uncollapsed(self):
        # uncollapsed likelihood
        log_ll = 0
        for x_distrib, theta_distrib in zip(self.x, self.theta):
            for x_val, theta_val in zip(x_distrib.flatten(),
                theta_distrib.flatten()):
                log_ll += x_val*np.log(theta_val)
        return log_ll

    def joint_collapsed(self):
        # collapsed likelihood
        log_ll = 0
        for x_distrib, alpha_d, symm in zip(self.x, self.alpha_tiled, self.symmetric):
            if symm:
                alpha_row = alpha_d[0,:]
                k_a = alpha_row.shape[0]
                alpha_v = alpha_row[0]
                alpha_v2 = k_a*alpha_v
                log_ll += x_distrib.shape[0]*(gammaln(alpha_v2)-
                    np.sum(gammaln(alpha_row)))
                log_ll += np.sum(gammaln(x_distrib+alpha_d))
                log_ll -= np.sum(gammaln(np.sum(x_distrib+alpha_d, axis=0)))
            else:
                log_ll += x_distrib.shape[0]*(gammaln(np.sum(alpha_d)) -
                    np.sum(gammaln(alpha_d)))
                log_ll += np.sum(gammaln(x_distrib+alpha_d))
                log_ll -= np.sum(gammaln(np.sum(x_distrib+alpha_d, axis=0)))
        return log_ll

    def likelihood_lda(self):
        log_ll = 0
        x_distrib = self.x[1]
        alpha_d = self.alpha_tiled[1]
        symm = self.symmetric[1]
        if symm:
            alpha_row = alpha_d[0,:]
            k_a = alpha_row.shape[0]
            alpha_v = alpha_row[0]
            alpha_v2 = k_a*alpha_v
            log_ll += x_distrib.shape[0]*(gammaln(alpha_v2)-
                k_a*gammaln(alpha_v))
            log_ll += np.sum(gammaln(x_distrib+alpha_d))
            log_ll -= np.sum(gammaln(np.sum(x_distrib+alpha_d, axis=1)))
        else:
            log_ll += x_distrib.shape[0]*(gammaln(np.sum(alpha_d)) -
                np.sum(gammaln(alpha_d)))
            log_ll += np.sum(gammaln(x_distrib+alpha_d))
            log_ll -= np.sum(gammaln(np.sum(x_distrib+alpha_d, axis=1)))
        return log_ll

def main():
    pass

if __name__=='__main__':
    main()

