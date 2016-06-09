#!/usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt

import shlex
import subprocess
import unittest
import sys

# logging stuff
import logging
from logging import debug as logd

class PBCoreTests(unittest.TestCase):
    @staticmethod
    def call_pb(cmd_str):
        devnull = open('/dev/null', 'w')
        p1 = subprocess.Popen(shlex.split(cmd_str),
            stdout=devnull, stderr=devnull)
        p1.wait()

    def test_coins2(self):
        logd('### COINS2 ###')
        cmd_str = 'peircebayes ./coins/coins2_example.pb -d'
        PBCoreTests.call_pb(cmd_str)
        with open('/tmp/peircebayes/aprob/plates/out1.plate', 'r') as fin:
            out1 = fin.read()
        with open('coins/test2.plate', 'r') as fin:
            test_out = fin.read()
        self.assertEqual(out1, test_out)
        # ground truth
        from coins.coins2_gen import p1,p2,p3
        thetas = np.array([p1, p2, p3])
        # learned params
        thetash = np.load('/tmp/peircebayes/avg_samples.npz')
        thetash = thetash['arr_0']
        thetash = thetash[:, 1]
        logd('Original params:\n{}'.format(thetas))
        logd('Learned params:\n{}'.format(thetash))
        norm2 = np.linalg.norm(thetas-thetash)
        logd('Norm2:\n{}'.format(norm2))
        self.assertTrue(norm2<0.1)

    def test_coins2_cgs(self):
        logd('### COINS2 CGS ###')
        cmd_str = 'peircebayes ./coins/coins2_example.pb -d -a cgs'
        PBCoreTests.call_pb(cmd_str)
        with open('/tmp/peircebayes/aprob/plates/out1.plate', 'r') as fin:
            out1 = fin.read()
        with open('coins/test2.plate', 'r') as fin:
            test_out = fin.read()
        self.assertEqual(out1, test_out)
        # ground truth
        from coins.coins2_gen import p1,p2,p3
        thetas = np.array([p1, p2, p3])
        # learned params
        thetash = np.load('/tmp/peircebayes/avg_samples.npz')
        thetash = thetash['arr_0']
        thetash = thetash[:, 1]
        logd('Original params:\n{}'.format(thetas))
        logd('Learned params:\n{}'.format(thetash))
        norm2 = np.linalg.norm(thetas-thetash)
        logd('Norm2:\n{}'.format(norm2))
        self.assertTrue(norm2<0.1)


    def test_coin_beta_args(self):
        logd('### COIN BETA ARGS ###')
        cmd_str = 'peircebayes ./coins/coin_beta_args.pb  -d'
        PBCoreTests.call_pb(cmd_str)
        thetash = np.load('/tmp/peircebayes/avg_samples.npz')
        thetash = thetash['arr_0']
        thetash = thetash[:, 1]
        logd('Param should be Beta(15, 85) distributed.')
        logd('Learned params:\n{}'.format(thetash))

    def test_coin_categorical(self):
        logd('### COIN CATEGORICAL ###')
        cmd_str = 'peircebayes ./coins/coin_categorical.pb  -d'
        PBCoreTests.call_pb(cmd_str)
        thetash = np.load('/tmp/peircebayes/avg_samples.npz')
        thetash = thetash['arr_0']
        thetash = thetash[:, 1]
        thetas = [0.5]
        logd('Param should be (0.5, 0.5).')
        logd('Learned params:\n{}'.format(thetash))
        norm2 = np.linalg.norm(thetas-thetash)
        logd('Norm2:\n{}'.format(norm2))
        self.assertTrue(norm2<0.001)

    def test_lda1(self):
        logd('### LDA1 ###')
        cmd_str = 'peircebayes ./lda/lda_example.pb -d'
        PBCoreTests.call_pb(cmd_str)
        with open('/tmp/peircebayes/aprob/plates/out1.plate', 'r') as fin:
            out1 = fin.read()
        with open('lda/lda_example_test.plate', 'r') as fin:
            test_out = fin.read()
        self.assertEqual(out1, test_out)
        thetash = np.load('/tmp/peircebayes/avg_samples.npz')
        muh = thetash['arr_0']
        phih = thetash['arr_1']
        logd('Learned mu:\n{}'.format(muh))
        logd('Learned phi:\n{}'.format(phih))

    def test_rim1(self):
        logd('### RIM1 ###')
        cmd_str = 'peircebayes ./rim/rim_example1.pb -d'
        PBCoreTests.call_pb(cmd_str)
        with open('/tmp/peircebayes/aprob/plates/out1.plate', 'r') as fin:
            out1 = fin.read()
        with open('rim/rim1_test.plate', 'r') as fin:
            test_out = fin.read()
        self.assertEqual(out1, test_out)

    def test_rim2(self):
        logd('### RIM2 ###')
        cmd_str = 'peircebayes ./rim/rim_example2.pb -d'
        PBCoreTests.call_pb(cmd_str)
        from rim.sample_mallows import p2,p3,p4
        p = [p2,p3,p4]
        thetas = np.load('/tmp/peircebayes/avg_samples.npz')
        ph2 = thetas['arr_0'][-1]
        ph3 = thetas['arr_1'][-1]
        ph4 = thetas['arr_2'][-1]
        ph = [ph2, ph3, ph4]
        logd('Original params:\n{}'.format('\n'.join([str(np.array(pi))
            for pi in p])))
        logd('Learned params:\n{}'.format('\n'.join([str(phi)
            for phi in ph])))
        norm2 = np.average([ np.linalg.norm(pi-pih) for pi,pih in zip(p,ph) ])
        logd('Norm2:\n{}'.format(norm2))
        self.assertTrue(norm2<0.1)


class PBOptTests(unittest.TestCase):
    @staticmethod
    def plot_topics(T,phi,fname):
        f, axs = plt.subplots(1,T+1,figsize=(15,1))
        ax = axs[0]
        ax.text(0,0.4, "Topics: ", fontsize = 16)
        ax.axis("off")
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        for (ax, (i,phi_t)) in zip(axs[1:], enumerate(phi)):
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.imshow(phi_t, cmap='Greys_r', interpolation='none')
        f.savefig('lda/{}.pdf'.format(fname), format='pdf')

    def test_lda2(self):
        logd('### LDA2 ###')
        cmd_str = ('peircebayes ./lda/lda_example2.pb'+
            ' -n 150 -b 100 -d -t')
        PBCoreTests.call_pb(cmd_str)
        # original params
        # phi is given as the horizontal and vertical topics
        # on 5X5 images
        #  word vocabulary
        W = 25
        # image size
        L = int(np.sqrt(W))
        # topics
        T = 2*L
        phi = [np.zeros((L, L)) for i in range(T)]
        line = 0
        for phi_t in phi:
            if line >= L:
                trueLine = int(line - L)
                phi_t[:,trueLine] = 1./L*np.ones(L)
            else:
                phi_t[line] = 1./L*np.ones(L)
            line += 1
        # plot original params
        PBOptTests.plot_topics(T, phi, 'lda2_phi')
        phi_flat = np.array(phi).reshape((T,W))
        thetash = np.load('/tmp/peircebayes/avg_samples.npz')
        muh = thetash['arr_0']
        phih_flat = thetash['arr_1']
        logd('Learned mu shape:\n{}'.format(muh.shape))
        logd('Learned phi shape:\n{}'.format(phih_flat.shape))
        phih = [phi_t.reshape(L,L) for phi_t in list(phih_flat)]
        # plot learned params
        PBOptTests.plot_topics(T, phih, 'lda2_phih')
        norm2 = np.average(np.apply_along_axis(np.linalg.norm, 1,
            phi_flat-phih_flat))
        logd('Average Norm2 on phi:\n{}'.format(norm2))
        self.assertTrue(norm2 < 1)
        PBCoreTests.call_pb('cp /tmp/peircebayes/lls.npz lda/lda2_lls.npz')

    def test_lda_amcmc(self):
        logd('### LDA AMCMC (depends on LDA2) ###')
        cmd_str = ("peircebayes ./lda/lda_example2.pb"+
            " -n 150 -b 100 -d -t -a amcmc")
        PBCoreTests.call_pb(cmd_str)
        PBCoreTests.call_pb('cp /tmp/peircebayes/lls.npz lda/lda_amcmc_lls.npz')
        with open('lda/lda2_lls.npz', 'r') as f:
            lda2_lls = np.load(f)['lls']
        with open('lda/lda_amcmc_lls.npz', 'r') as f:
            lda_amcmc_lls = np.load(f)['lls']
        x = np.arange(lda2_lls.shape[0])
        logd('lda2')
        logd(lda2_lls)
        logd('lda amcmc')
        logd(lda_amcmc_lls)
        plt.figure()
        plt.plot(x, lda2_lls, linestyle='-.', color='b', label='PB_ugs')
        plt.plot(x, lda_amcmc_lls, linestyle='-', color='r', label='PB_amcmc')
        plt.legend(loc='lower right')
        plt.savefig('lda/lls_ugs_amcmc.pdf', format='pdf')

    def test_lda_cgs(self):
        logd('### LDA CGS (depends on LDA2) ###')
        cmd_str = ("peircebayes ./lda/lda_example2.pb"+
            " -n 150 -b 100 -d -t -a cgs")
        PBCoreTests.call_pb(cmd_str)
        PBCoreTests.call_pb('cp /tmp/peircebayes/lls.npz lda/lda_amcmc_lls.npz')
        with open('lda/lda2_lls.npz', 'r') as f:
            lda2_lls = np.load(f)['lls']
        with open('lda/lda_amcmc_lls.npz', 'r') as f:
            lda_amcmc_lls = np.load(f)['lls']
        x = np.arange(lda2_lls.shape[0])
        logd('lda2')
        logd(lda2_lls)
        logd('lda cgs')
        logd(lda_amcmc_lls)
        plt.figure()
        plt.plot(x, lda2_lls, linestyle='-.', color='b', label='PB_ugs')
        plt.plot(x, lda_amcmc_lls, linestyle='-', color='r', label='PB_cgs')
        plt.legend(loc='lower right')
        plt.savefig('lda/lls_ugs_cgs.pdf', format='pdf')


def core_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PBCoreTests)

def opt_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PBOptTests)

def main():
    logging.basicConfig(filename='tests.log', filemode='w',
        level=logging.DEBUG)
    s1 = core_suite()
    s2 = opt_suite()
    l = []
    if len(sys.argv) == 1 or sys.argv[1] == 'core':
        l = [s1]
    elif sys.argv[1] == 'opt':
        l = [s2]
    else:
        l = [s1,s2]
    alltests = unittest.TestSuite(l)
    unittest.TextTestRunner(verbosity=2).run(alltests)

if __name__=='__main__':
    main()
