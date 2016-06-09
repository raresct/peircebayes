#!/usr/bin/env python2

"""
:synopsis: Module for probabilistic inference on a :class:`formula_gen.PBModel`.
"""

from __future__ import division

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
    reparam_cy, backward_ob_cy, sample_bdd_ob_cy, cgs_iter_cy, reparam_update_cy)

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
        return [self.backward_plate(pb_plate, plate_idx) if not(pb_plate.bdd is None)
            else 0
            for (plate_idx, pb_plate) in enumerate(self.pb_model.plates)]

    def backward_plate(self, pb_plate, plate_idx):
        #logd(plate_idx)
        #if plate_idx == 18:
        #    logd(pb_plate.bdd)
        #    logd(self.bdd_param[plate_idx])
        r = backward_plate_cy(pb_plate.bdd, self.bdd_param[plate_idx])
        #logd(plate_idx)
        return r


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
        #logd('beta time: {} s'.format(end-start))
        #mem2 = psutil.virtual_memory().percent
        #logd('after beta: {}'.format(mem2))

        #start = time.clock()
        for (plate_idx, (pb_plate, beta)) in enumerate(zip(
            self.pb_model.plates, betas)):
            if not(pb_plate.bdd is None):
                self.sample_bdd_plate(pb_plate, beta, plate_idx)
        #logd(self.x)
        #end = time.clock()
        #logd('sample bdd time: {} s'.format(end-start))
        return betas

    def sample_bdd_plate(self, pb_plate, beta, plate_idx):
        plate = pb_plate.plate
        bdd = pb_plate.bdd
        reps = pb_plate.reps
        prob = self.bdd_param[plate_idx]
        sample_bdd_plate_cy(plate, bdd, reps, beta, prob, self.x)

    def reset_x(self):
        """
        Reset counts in x. Updates x.

        :rtype: None
        """
        distribs = self.pb_model.distribs
        self.x = [np.zeros((distrib[2], distrib[1])) for distrib in distribs]

    def generic_sampler(self, n, burn_in, lag, track_ll, in_file, dbg_iter,
        init_f, iteration_f, post_process_samples_f, post_process_lls_f):
        '''
        samples = list
        track_ll = boolean

        Signature for functions:
        - init_f() -> None
        - iteration_f(track_ll) -> sample, ll
        - post_process_samples_f(samples) -> None
        - post_process_lls_f(samples) -> None

        '''
        max_iter = n+burn_in
        samples = []
        if track_ll:
            lls = []
        init_f()
        logd('Done init!')
        for i in range(max_iter):
            start_it = time.time()
            sample, ll = iteration_f(track_ll)
            end_it = time.time()
            #logd('time: {}'.format(end_it-start_it))
            #logd(np.sum(self.x[0], axis=1))
            #logd(np.sum(self.x[1], axis=0))
            #logd(np.sum(self.x[1]))
            #logd(ll)
            if np.isclose(i%dbg_iter, 0):
                logd('iteration {} time: {} s'.format(i, end_it-start_it))
                #logd('nbdds')
                #logd(len(self.pb_model.plates))
                #logd(ll)
            if i>=burn_in and (i-burn_in)%lag==0:
                samples.append(sample)
                if track_ll:
                    lls.append(ll)
        logd('Number of samples: {}'.format(len(samples)))
        post_process_samples_f(samples)
        if track_ll and post_process_lls_f:
            post_process_lls_f(lls, in_file)

    def infer(self, options):
        self.sample(**options)

    def sample(self, n, burn_in, lag, track_ll, metric, in_file, algo, dbg_percent=5.0):
        dbg_iter = np.round((n+burn_in)*dbg_percent/100)
        metrics = {
            'joint' :   self.joint_collapsed,
            'lda'   :   self.likelihood_lda,
            'total' :   self.total_ll,
            'joint_un' :   self.joint_uncollapsed
        }
        self.metric = metrics[metric]
        algos = {
            'ugs'       : lambda x:
                (lambda: None, x.ugs_iteration, x.ugs_samples, x.ugs_lls),
            'amcmc'     : lambda x:
                (lambda: None, x.amcmc_iteration, x.ugs_samples, x.ugs_lls),
            'cgs'       : lambda x:
                (x.cgs_init, x.cgs_iteration, x.cgs_samples, x.ugs_lls),
            'predict'   : lambda x:
                (lambda : None, x.predict_iteration, x.predict_samples, None)
        }
        start_sampling = time.time()
        self.generic_sampler(n, burn_in, lag, track_ll, in_file, dbg_iter,
            *algos[algo](self))
        end_sampling = time.time()
        logd('Sampling time: {} s '.format(end_sampling-start_sampling))

    def ugs_iteration(self, track_ll):
        self.sample_theta()
        self.betas = self.sample_bdd_plates()
        ll = self.metric() if track_ll else None
        test = np.array(self.betas[0])[:,-1,1]
        sample = [np.sum(np.log(beta[:,-1,1])) if hasattr(beta, 'shape') else np.array(beta)
            for beta in self.betas]
        return (self.theta, self.alpha_post, sample), ll

    def amcmc_iteration(self, track_ll):
        self.set_theta_mean()
        self.sample_bdd_plates()
        ll = self.metric() if track_ll else None
        return (self.theta, self.alpha_post), ll

    def predict_iteration(self, track_ll):
        self.sample_theta()
        betas = self.backward_plates()
        sample = [beta[:,-1,1] if hasattr(beta, 'shape') else np.array(beta)
            for beta in betas]
        return sample, None

    def cgs_init(self):
        #self.l = None
        #self.old_l = None
        self.set_theta_mean()
        #self.sample_theta()
        # this will be included in obs_l
        self.obs_d = {}
        for plate_idx, pb_plate in enumerate(self.pb_model.plates):
            for i,rep in enumerate(pb_plate.reps):
                for ob in range(rep):
                    betas = self.backward_ob(pb_plate.bdd, self.bdd_param[plate_idx][i,:])
                    l = self.sample_bdd_ob(pb_plate.plate[i,:,:], pb_plate.bdd, betas, self.bdd_param[plate_idx][i,:])
                    self.obs_d[(plate_idx, i, ob)] = copy.copy(l)
        self.set_theta_mean()
        #logd(self.x)

    def cgs_iteration2(self, track_ll):
        cgs_iter_cy(self)
        ll = self.metric() if track_ll else None
        return (self.theta, self.alpha_post), ll

    def cgs_iteration3(self, track_ll):
        for plate_idx, pb_plate in enumerate(self.pb_model.plates):
            for i,rep in enumerate(pb_plate.reps):
                for ob in range(rep):
                    # if theta not seta compute it from priors
                    for aa,ii,ll in self.obs_d[(plate_idx, i, ob)]:
                        self.x[aa][ii,ll] -=1
                    #self.set_theta_mean()
                    # sample ob
                    #print pb_plate.bdd.shape
                    #print self.bdd_param[plate_idx][i,:].shape
                    betas = self.backward_ob(pb_plate.bdd, self.bdd_param[plate_idx][i,:])
                    #print betas.shape
                    l = self.sample_bdd_ob(pb_plate.plate[i,:,:], pb_plate.bdd, betas, self.bdd_param[plate_idx][i,:])
                    self.obs_d[(plate_idx, i, ob)] = l
                    if ob < rep-1:
                        plate_idx2 = plate_idx
                        i2 = i
                    elif i<len(pb_plate.reps)-1:
                        plate_idx2 = plate_idx
                        i2 = i+1
                    elif plate_idx<len(self.pb_model.plates)-1:
                        plateidx2 = plate_idx+1
                        i2 = 0
                    else:
                        plate_idx2 = 0
                        i2 = 0
                    self.update_theta(l, plate_idx2, i2)
                    #print self.x
                    #print l
                    #return
        ll = self.metric() if track_ll else None
       # logd(self.x)
        return (self.theta, self.alpha_post), ll

    def cgs_iteration4(self, track_ll):
        for plate_idx, pb_plate in enumerate(self.pb_model.plates):
            bag_of_obs = copy.copy(pb_plate.reps)
            reps_sum = np.sum(pb_plate.reps)
            bag_size = pb_plate.reps.shape[0]
            bag_sum = np.sum(bag_of_obs)
            while np.sum(bag_of_obs)>0:
                i = int(np.random.choice(bag_size, 1, True, bag_of_obs/bag_sum))
                ob = bag_of_obs[i]-1
                bag_of_obs[i] -= 1
                if bag_sum < reps_sum:
                    self.update_theta(l, plate_idx, i)
                bag_sum -=1
                for aa,ii,ll in self.obs_d[(plate_idx, i, ob)]:
                    self.x[aa][ii,ll] -=1
                betas = self.backward_ob(pb_plate.bdd, self.bdd_param[plate_idx][i,:])
                l = self.sample_bdd_ob(pb_plate.plate[i,:,:], pb_plate.bdd, betas, self.bdd_param[plate_idx][i,:], True)
                self.obs_d[(plate_idx, i, ob)] = copy.copy(l)
                #return
        ll = self.metric() if track_ll else None
        return (self.theta, self.alpha_post), ll

    def cgs_iteration(self, track_ll):
        obs_l = copy.copy(self.obs_d.keys())
        np.random.shuffle(obs_l)
        #self.reset_x()
        #self.set_theta_mean()
        #first = 0
        for (plate_idx, i, ob) in obs_l:
            pb_plate = self.pb_model.plates[plate_idx]
            curr_l = self.obs_d[(plate_idx, i, ob)]
            for aa,ii,ll in curr_l:
            #    if self.x[aa][ii,ll] <= 0:
            #        print aa,ii,ll
                self.x[aa][ii,ll] -=1
                #logd((aa, ii, ll))
            self.update_theta(curr_l, plate_idx, i)
            #if self.l is not None:
                #print self.x[0][56,:]
                #self.update_theta(curr_l+self.l+self.old_l, plate_idx, i)
                #self.update_theta(curr_l, plate_idx, i)
                #if first > 2 and first < 10:
                    #theta1 = copy.copy(self.theta)
                    #print self.bdd_param[plate_idx][i,:]
                    #print self.x[0][56,:]
                    #p1 = self.bdd_param[plate_idx][i,:]
                    #self.set_theta_mean()
                    #p2 = self.bdd_param[plate_idx][i,:]
                    #logd(np.prod(np.isclose(p1, p2)))
                    #theta2 = copy.copy(self.theta)
                    #logd(np.prod(np.isclose(theta1[0], theta2[0])))
                    #logd(np.prod(np.isclose(theta1[1], theta2[1])))
            #first +=1
                #print theta1[0][56,:]
                #print np.sum(theta1[0][56,:])
                #print theta2[0][56,:]
                #print np.sum(theta2[0][56,:])
            betas = self.backward_ob(pb_plate.bdd, self.bdd_param[plate_idx][i,:])
            l = self.sample_bdd_ob(pb_plate.plate[i,:,:], pb_plate.bdd, betas, self.bdd_param[plate_idx][i,:])
            self.update_theta(l)
            self.obs_d[(plate_idx, i, ob)] = copy.copy(l)
            #return
        #logd(self.x)
        #self.update_theta(curr_l+self.l+self.old_l, plate_idx, i)
        #theta1 = copy.copy(self.theta)
        #print self.x[0][56,:]
        #self.set_theta_mean()
        #theta2 = copy.copy(self.theta)
        #logd(np.prod(np.isclose(theta1[0], theta2[0])))
        #logd(np.prod(np.isclose(theta1[1], theta2[1])))
        ll = self.metric() if track_ll else None
        # third element of the tuple is the 'total likelihood' of the model (betas)
        # currently not computed here, nor in cgs_samples
        return (self.theta, self.alpha_post), ll

    def cgs_iteration6(self, track_ll):
        obs_l = copy.copy(self.obs_d.keys())
        np.random.shuffle(obs_l)
        cgs_iter_cy(self, obs_l)
        ll = self.metric() if track_ll else None
        return (self.theta, self.alpha_post), ll


    def backward_ob(self, bdd, bdd_param):
        return backward_ob_cy(bdd, bdd_param)
    def sample_bdd_ob(self, plate, bdd, betas, prob):
        return sample_bdd_ob_cy(plate, bdd, betas, prob, self.x)

    @staticmethod
    def pb_avg(thetas):
        sum_thetas = [np.zeros(theta_d.shape) for theta_d in thetas[0]]
        for theta_it in thetas:
            for i, theta_d in enumerate(theta_it):
                sum_thetas[i] += theta_d
        return [theta_d/float(len(thetas)) for theta_d in sum_thetas]

    def ugs_samples(self, samples):
        thetas, alphas, logbetas = zip(*samples)
        self.last_theta = thetas[-1]
        self.theta_avg = PBInference.pb_avg(thetas)
        #logd('theta avg')
        #for theta_it in self.theta_avg:
        #    logd(theta_it)
        self.alpha_avg = PBInference.pb_avg(alphas)
        logd(np.array(logbetas).shape)
        #logd(betas)
        #logd(len(betas)) # 100
        #logd(len(betas[0])) # 361
        #logd(float(betas[0][0]))
        #logd(np.sum(np.average(logbetas, axis=0)))
        self.total_ll = 0
        duds = 0
        for pb_plate in self.pb_model.plates:
            if pb_plate.bdd is None:
                duds += 1
                self.total_ll += np.log(10**-6)
        self.total_ll += np.average(np.sum(logbetas, axis=1))
        logd('total_ll')
        logd(self.total_ll)
        logd('duds')
        logd(duds)
        #logd('alpha avg')
        #for alpha_it in self.alpha_avg:
        #    logd(alpha_it)

    def cgs_samples(self, samples):
        # no complete likelihood (based on beta) atm
        thetas, alphas = zip(*samples)
        self.last_theta = thetas[-1]
        self.theta_avg = PBInference.pb_avg(thetas)
        self.alpha_avg = PBInference.pb_avg(alphas)

    def predict_samples(self, samples):
        avg_beta = [np.zeros(beta.shape) for beta in samples[0]]
        #logd(len(avg_beta))
        #logd(avg_beta[0].shape)
        for beta_it in samples:
            avg_beta = [beta_arr+beta_avg
                for beta_arr, beta_avg in zip(beta_it, avg_beta)]
        for beta_avg in avg_beta:
            beta_avg /= len(samples)
        self.avg_beta = avg_beta

    def ugs_lls(self, lls, in_file):
        self.plot_ll(lls, in_file=in_file)

    def plot_ll(self, lls, burn_in=None, in_file=None):
        #lls = lls[1:]
        import matplotlib.pyplot as plt
        plt.figure('lls')
        plt.plot(list(range(len(lls))), lls)
        plt.xlabel('Iterations')
        plt.ylabel('Log Likelihood')
        plt.title('PB in_file={}'.format(in_file))
        if burn_in > 0:
            #plt.axvline(x=burn_in, linewidth=1, color='red', label='Burn in')
            plt.legend(loc='lower right')
        # defer savefig to peircebayes script
        #plt.savefig('/tmp/peircebayes/ll.pdf', format='pdf')
        # instead
        self.lls = lls

    def set_theta_mean(self):
        distribs = self.pb_model.distribs
        if self.x is None:
            self.reset_x()
        self.alpha_post = [alpha_d+x_distrib
                for x_distrib, alpha_d in zip(self.x,self.alpha_tiled)]
        self.theta = [np.array(
            [alpha_row/np.sum(alpha_row) for alpha_row in alpha_d]
            ).reshape(alpha_d.shape)
            for alpha_d in self.alpha_post]
        for i, distrib in enumerate(distribs):
            self.theta[i] = np.tile(distrib[0], (distrib[2],1)) if distrib[3] else self.theta[i]
        # clip pls TODO find a better workaround
        eps = 10**-300
        self.theta = [theta_row.clip(min=eps) for theta_row in self.theta]
        self.reparam2()
        #print 'post theta mean'
        #print self.theta[0][56,:]

    def update_theta(self, l, plate_idx=None, i=None):
        #print l
        for (aa,ii,ll) in l:
            posterior = self.alpha_tiled[aa][ii,:]+self.x[aa][ii,:]
            self.theta[aa][ii,:] = posterior/np.sum(posterior)
            #self.theta[aa][ii,:] = np.random.dirichlet(posterior,1)[0]
        if plate_idx is not None and i is not None:
            self.reparam_update(l, plate_idx, i)
        #print 'post update'
        #print self.theta[0][56,:]

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
        self.alpha_post = [alpha_d+x_distrib
                for x_distrib, alpha_d in zip(self.x,self.alpha_tiled)]
        self.theta = [np.array(
            [np.random.dirichlet(alpha_row,1)[0] for alpha_row in alpha_d]
            ).reshape(alpha_d.shape)
            for alpha_d in self.alpha_post]
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

    def reparam_update(self, l, plate_idx, i):
        #self.bdd_param = []
        pb_plate = self.pb_model.plates[plate_idx]
        bdd_param_plate = self.bdd_param[plate_idx]
        if not(pb_plate.bdd is None):
            reparam_update_cy(
                pb_plate.cat_list, self.theta, pb_plate.plate, pb_plate.kid,
                l, bdd_param_plate, i)

    def reparam2(self):
        self.bdd_param = []
        for pb_plate in self.pb_model.plates:
            if not(pb_plate.bdd is None):
                self.bdd_param.append(reparam_cy(
                    pb_plate.cat_list, self.theta, pb_plate.plate, pb_plate.kid))
            else:
                self.bdd_param.append([])

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
                log_ll -= np.sum(gammaln(np.sum(x_distrib+alpha_d, axis=1)))
            else:
                log_ll += x_distrib.shape[0]*(gammaln(np.sum(alpha_d)) -
                    np.sum(gammaln(alpha_d)))
                log_ll += np.sum(gammaln(x_distrib+alpha_d))
                log_ll -= np.sum(gammaln(np.sum(x_distrib+alpha_d, axis=1)))
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

    def total_ll(self):
        logbetas = [np.log(beta[:,-1,1]) if hasattr(beta, 'shape') else np.array(beta)
            for beta in self.betas]
        self.total_ll = 0
        duds = 0
        for pb_plate in self.pb_model.plates:
            if pb_plate.bdd is None:
                duds += 1
                self.total_ll += np.log(10**-6)
        self.total_ll += np.sum(logbetas)
        logd('total_ll')
        logd(self.total_ll)
        logd('duds')
        logd(duds)
        return self.total_ll

def main():
    pass

if __name__=='__main__':
    main()

