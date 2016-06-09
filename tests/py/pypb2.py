import functools
import copy
import numpy as np
import operator
import pickle

from scipy.sparse import csr_matrix

class DrawError(Exception):
    def __init__(self, a, i, k1, k2):
        self.a = a
        self.i = i
        self.k1 = k1
        self.k2 = k2
    def __str__(self):
        return ('Different draws in distribution ({}, {}):\t{} and {}'.
            format(self.a,self.i,self.k1,self.k2))

class DistribError(Exception):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'Invalid distribution name:\t{}'.format(self.name)

class PBDef:
    def __init__(self, distribs, out='data/out1.plate'):
        self.defs = {}
        self.explain = None
        self.curr_explanation = {}
        self.curr_solution = []
        self.a = 0
        self.count=1
        for distrib in distribs:
            self.add_dirichlet(*distrib)
        self.suff_stats()
        self.out = out
        self.prob_graph = {}
        self.update_d = {}
        self.data = []
        with open(self.out, 'w') as f:
            pass

    def suff_stats(self):
        self.x, self.xsum, self.alphas = {}, {}, {}
        for _,(prior, a, k_a, i_a) in self.defs.items():
            self.x[a] = np.zeros((i_a, k_a))
            self.xsum[a] = np.sum(self.x[a], axis=1)
            self.alphas[a] = np.array([prior, prior*k_a])

    def add_dirichlet(self, prior, name, k_a, i_a):
        self.defs[name] = (prior, self.a, k_a, i_a)
        self.a += 1

    def draw(self, name, k, i, count=1):
        if name not in self.defs:
            raise DistribError(name)
        (_, a, k_a, i_a) = self.defs[name]
        assert i<i_a
        assert k<k_a

        #sum_x_alpha_str = ('s1',a,i,k) #'sum_x_alpha_{}_{}_{}'.format(a,i,k)
        #sum_xsum_alpha_str = ('s2',a,i) #'sum_xsum_alpha_{}_{}'.format(a,i)
        #ratio_str = ('r',a,i,k)#'ratio_{}_{}_{}'.format(a,i,k)
        #mdraw_str = ('d',a,i,k)#'mdraw_{}_{}_{}'.format(a,i,k)
        #print('{} {} {}'.format(a,i,k))
        #mdraw_graph = {
        #    sum_x_alpha_str     : (operator.add, self.x[a][i,k], self.alphas[a][0]),
        #    sum_xsum_alpha_str  : (operator.add, self.xsum[a][i], self.alphas[a][1]),
        #    ratio_str           : (operator.truediv, sum_x_alpha_str, sum_xsum_alpha_str),
        #    mdraw_str           : (operator.pow, ratio_str, count)
        #}
        #self.prob_graph = {**self.prob_graph, **mdraw_graph}
        #self.prob_graph.update(mdraw_graph)
        #self.prob_graph['c'][1].append(mdraw_str)

        #curr_counts = self.update_d.get((a,i,k),0)
        #self.update_d[a,i,k] = curr_counts+count

        #if k>=k_a:
        #    print('{},{}'.format(k,k_a))
        #    return

        if (a, i, k) in self.curr_explanation:
            self.curr_explanation[(a,i,k)] += count
        else:
            self.curr_explanation[(a,i,k)] = count

    def draw_mem(self, name, k, i):
        if name not in self.defs:
            raise DistribError(name)
        (_, a, k_a, i_a) = self.defs[name]
        assert i<i_a
        assert k<k_a

        #if (a, i) in self.curr_explanation:
        #    for k2 in self.curr_explanation[(a,i)]:
        #        if k != k2:
        #            raise DrawError(a, i, k, k2)
        #else:
        #    self.curr_explanation[(a,i)] = {k:1}

    def __call__(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            self.curr_explanation = {}
            self.curr_solution = []
            #self.prob_graph = {}
            #self.prob_graph['c'] = (np.prod, [])
            #i = 0
            #self.update_d = {}
            for _ in fn(self, *args, **kwargs):
                #print(self.prob_graph['curr_out'])
                #self.prob_graph[('o',i)] = copy.copy(self.prob_graph['c'])
                #self.prob_graph['c'] = (np.prod, [])
                #self.prob_graph[('u',i)] = self.update_d
                #self.update_d = {}
                self.curr_solution.append(copy.copy(self.curr_explanation))
                self.curr_explanation = {}
                #i+=1
            #self.prob_graph['n_sum'] = (np.sum, [('o',j) for j in range(i)])
            #self.prob_graph['n_div'] = (np.divide, [('o',j) for j in range(i)], 'n_sum')
            #self.prob_graph['sample_pre'] = (np.random.multinomial, 1, 'n_div')
            #self.prob_graph['sample'] = (np.argmax, 'sample_pre')

            # debug
            #print(self.prob_graph)

            #import dask

            #import time

            #start = time.time()
            #for i in range(10**4):
            #    res = dask.threaded.get(self.prob_graph, 'sample')
            #end = time.time()
            #print('time {}'.format(end-start))

            #res = dask.threaded.get(self.prob_graph, 'mdraw_0_0_0')

            #res = dask.threaded.get(self.prob_graph, 'sample')
            #res2 = dask.threaded.get(self.prob_graph, 'out_update_{}'.format(res))
            #print(res2)

            #self.prob_graph['count'] = kwargs['count']
            #assert int(self.prob_graph['count']) > 0
            self.count = int(kwargs['count'])
            assert self.count>0
            #self.data.append(self.prob_graph)
            self.write_solution()

            #for i in range(self.count):
            #    res = dask.threaded.get(self.prob_graph, 'sample')
        return decorated

    def infer(self, burnin=100):
        print(burnin)
        print(self.data)

    def write_solution(self):
        with open(self.out, 'a') as f:
            #pickle.dump((self.curr_solution,self.count), f)
            f.write(';'.join(
                ['.'.join(
                    ['{},{},{},{}'.format(a,i,k,c)
                        for (a,i,k),c in expl.items()]
                ) for expl in self.curr_solution]
                 +[str(self.count)])+'\n')

