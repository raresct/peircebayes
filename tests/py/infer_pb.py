import pickle
import dask
import numpy as np

class InferPB:
    def __init__(self, defs_file):
        with open(defs_file, 'rb') as f:
            self.defs = pickle.load(f)
        self.suff_stats()
        self.prepare_infer()

    def suff_stats(self):
        self.x, self.xsum, self.alphas = {}, {}, {}
        for _,(prior, a, k_a, i_a) in self.defs.items():
            self.x[a] = np.zeros((i_a, k_a))
            self.xsum[a] = np.sum(self.x[a], axis=1)
            self.alphas[a] = np.array([prior, prior*k_a])

    def prepare_infer(self):
        with open('out1.plate', 'r') as fin:
            # draws graph is a graph of computation graphs
            draws_graph = {}
            # obs_graph is a list of computation graphs
            obs_graphs = []
            # number of times each observation is repeated
            reps = []
            for line in fin:
                obs_graph = {}
                line_split = line.split(';')
                line_sols, line_reps = line_split[:-1], line_split[-1]
                reps.append(int(line_reps))
                for (j,line_idxs) in enumerate(line_sols):
                    # add out node TODO
                    curr_parent = None
                    for idxs in line_idxs.split('.'):
                        a, i, k, c = [int(idx) for idx in idxs.split(',')]
                        if (a,i,k,c) not in draws_graph:
                             draws_graph[(a,i,k,c)] = {} # TODO
                        # add to out list TODO
                        if ('o',a,i,k,c) in obs_graph and curr_parent:
                            _, l = obs_graph[('o',a,i,k,c)]
                            l.append[curr_parent]
                        else:
                            obs_graph[('o',a,i,k,c)] = (np.prod, [])
                        curr_parent = a,i,k,c



