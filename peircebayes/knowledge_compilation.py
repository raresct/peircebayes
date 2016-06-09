#!/usr/bin/env python

"""
:synopsis: Module for compiling the formulae in a :class:`formula_gen.PBModel` into (RO)BDDs.
"""

import pycudd
import sys
import re
import shutil
import os
import time
import copy

import numpy as np

# logging stuff
import logging
from logging import debug as logd

# parallel
from joblib import Parallel, delayed
import multiprocessing


#class PickalableSwigPyObject(SwigPyObject, PickalableSWIG):
#    def __init__(self, *args):
#        self.args = args
#        SwigPyObject.__init__(self)

#class PickalableSWIG:
#    def __setstate__(self, state):
#        self.__init__(*state['args'])

#    def __getstate__(self):
#        return {'args': self.args}

def compile_k(pb_model, debug, tmp_dir):
    """
    Adds a bdd attribute of type :class:`BDD` to each plate in a :class:`formula_gen.PBModel`. If f_str is empty, the BDD will be None.

    :param pb_model:
    :type pb_model: :class:`formula_gen.PBModel`
    :rtype: None

    """
    mgr = pycudd.DdManager()
    mgr.SetDefault()
    for i, pb_plate in enumerate(pb_model.plates):
        #logd(pb_plate.plate.shape)
        n_vars = pb_plate.plate.shape[1]
        if pb_plate.f_str:
            bdd = BDD(mgr, pb_plate.f_str, n_vars, debug, i, tmp_dir)
            #pb_plate.bdd = bdd
            pb_plate.bdd = bdd.bdd
        else:
            #logd('==='+str(i)+'===')
            pb_plate.bdd = None

# doesn't work, don't know why, probably due to pycudd
def compile_k_parallel(pb_model, debug, tmp_dir):
    """
    Adds a bdd attribute of type :class:`BDD` to each plate in a :class:`formula_gen.PBModel`. If f_str is empty, the BDD will be None.

    :param pb_model:
    :type pb_model: :class:`formula_gen.PBModel`
    :rtype: None

    """
    logd('parallel_kc')
    start = time.time()
    mgr = pycudd.DdManager()
    mgr.SetDefault()
    mgrPickable = PickalableSwigPyObject(mgr)
    n_cores = multiprocessing.cpu_count()
    batch_size = int(float(len(pb_model.plates))/float(n_cores))
    l = [copy.copy(pb_plate) for pb_plate in pb_model.plates]
    bdds = Parallel(n_jobs=n_cores, batch_size=batch_size)(delayed(create_bdd)(mgrPickable, pb_plate, debug, tmp_dir) for pb_plate in l)
    logd('parallel_kc0 took: {}'.format(time.time()-start))
    for bdd,pb_plate in zip(bdds, pb_model.plates):
        pb_plate.bdd = bdd
    logd('parallel_kc took: {}'.format(time.time()-start))

def create_bdd(mgr, pb_plate, debug, tmp_dir):
    if pb_plate.f_str:
        n_vars = pb_plate.plate.shape[1]
        return BDD(mgr, pb_plate.f_str, n_vars, debug, 0, tmp_dir).bdd
    else:
        return None


class BDD:
    '''
    Takes:

    * f_str : a string of a boolean formula
    * n_vars : the number of variables in f_str

    and creates an atrtibute:

    * bdd : an array of NNodes X 4, where bdd[i] = label, high, low, iscomp, where

        * label   = level of the node, -1 for constant nodes
        * high    = index in bdd of high child, -1 for constant nodes
        * low     = index in bdd of low child, -1 for constant nodes
        * iscomp  = boolean integer, 0 if node is complement, 1 otherwise

    See code for the methods & logic. PyCUDD is used to compile the (RO)BDD.

    '''
    def __init__(self, mgr, f_str, n_vars, debug, i, tmp_dir):
        # attributes set by compilation
        # TODO try to remove as many as possible
        #self.mgr        = None
        #self.root       = None
        #self.parents    = None
        #self.nodes      = None
        #self.h_nodes    = None
        #self.idx_nodes  = None
        self.bdd = None
        self.tmp_dir = tmp_dir
        self.compile2(mgr, f_str, n_vars, debug, i)

    def compile2(self, mgr, f_str, n_vars, debug, i):
        self.bdd = self.compile_pycudd2(mgr, f_str, n_vars, debug, i)
        #(self.mgr, self.root, self.parents, self.nodes, self.h_nodes,
        #    self.idx_nodes) = self.compile_pycudd(f_str, n_vars)

    def compile(self, f_str, n_vars):
        #self.bdd = self.compile_pycudd(f_str, n_vars)
        (self.mgr, self.root, self.parents, self.nodes, self.h_nodes,
            self.idx_nodes) = self.compile_pycudd(f_str, n_vars)

    def compile_pycudd2(self, mgr, f_str, n_vars, debug, idx):
        # initialize PyCUDD
        # add variables
        #pattern = '\d+'
        #to_var = lambda match : 'v'+str(int(match.group(0))-1)
        #to_exec = re.sub(pattern, to_var, f_str)
        #logd(to_exec)
        for i in range(n_vars):
            exec 'v{} = mgr.IthVar({})'.format(i,i) in locals()
        #logd('ok1')
        exec 'f = {}'.format(f_str) in locals()
        #logd('ok2')
        # Debug BDD
        #if n_vars>10:
        if debug:
            f.DumpDot()
            shutil.move('out.dot',
                os.path.join(self.tmp_dir, 'bdd_plate_{}.dot'.format(idx)))
        #logd(f.T().T().NodeReadIndex())
        #logd(f.E().E().NodeReadIndex())
        #root = f
        # OLD compute node parents
        #parents, nodes, h_nodes, idx_nodes = self._compute_node_parents(root)
        #return mgr, root, parents, nodes, h_nodes, idx_nodes
        return self._parse_bdd(f)

    def compile_pycudd(self, f_str, n_vars):
        # initialize PyCUDD
        mgr = pycudd.DdManager()
        mgr.SetDefault()
        # add variables
        #pattern = '\d+'
        #to_var = lambda match : 'v'+str(int(match.group(0))-1)
        #to_exec = re.sub(pattern, to_var, f_str)
        #logd(to_exec)
        for i in range(n_vars):
            exec 'v{} = mgr.IthVar({})'.format(i,i) in locals()
        exec 'f = {}'.format(f_str) in locals()

        # Debug BDD
        if n_vars>10:
            f.DumpDot()

        #logd(f.T().T().NodeReadIndex())
        #logd(f.E().E().NodeReadIndex())
        root = f
        # OLD compute node parents
        parents, nodes, h_nodes, idx_nodes = self._compute_node_parents(root)
        return mgr, root, parents, nodes, h_nodes, idx_nodes


    def _parse_bdd(self, root):
        if root.IsConstant():
            return np.array([-1,-1,-1, root.IsComplement()]).reshape((1,4))
        to_visit        = [root]
        visited         = set([hash(root)])
        visited_nodes   = [root]
        while to_visit:
            curr_node = to_visit.pop()
            high,low = curr_node.T(), curr_node.E()
            if hash(low) not in visited:
                if not low.IsConstant():
                    to_visit.append(low)
                visited.add(hash(low))
                visited_nodes.append(low)
            if hash(high) not in visited:
                if not high.IsConstant():
                    to_visit.append(high)
                visited.add(hash(high))
                visited_nodes.append(high)
        sorted_nodes    = sorted(visited_nodes, key= lambda x: x.NodeReadIndex(),
            reverse=True)
        logd('No. nodes: {}'.format(len(sorted_nodes)))
        h_nodes         = map(hash, sorted_nodes)
        bdd             = np.zeros((len(sorted_nodes), 4), dtype=np.int)
        for i,node in enumerate(sorted_nodes):
            if node.IsConstant():
                bdd[i] = [-1, -1, -1, node.IsComplement()]
            else:
                high, low = node.T(), node.E()
                high_idx, low_idx = map(lambda x: h_nodes.index(hash(x)),
                    [high, low])
                bdd[i] = [node.NodeReadIndex(), high_idx,
                    low_idx, node.IsComplement()]
        #logd(bdd)
        return bdd

    def _compute_node_parents(self, root):
        myhash = PyCUDDUtil.cudd_hash
        parents = {}
        # init all_nodes
        h_root = myhash(root)
        all_nodes = {h_root:root}
        # init parents
        labels = {}
        visited = set([h_root])
        to_visit = [h_root]
        i = j = 0
        while to_visit:
            # get next node
            i+=1
            curr_node = all_nodes[to_visit.pop()]
            if curr_node.IsConstant():
                j+=1
                continue
            # safe to get children
            high,low = curr_node.T(), curr_node.E()
            h_high, h_low, h_curr = map(myhash, [high, low, curr_node])
            all_nodes[h_low] = low
            all_nodes[h_high] = high
            if h_low not in visited:
                to_visit.append(h_low)
                visited.add(h_low)
            if h_high not in visited:
                to_visit.append(h_high)
                visited.add(h_high)
            parents = PyCUDDUtil._add_to_parents(
                parents, curr_node, [h_low, h_high])
            if h_curr==h_root:
                node_desc = 'root'
            else:
                # TODO there is a bug when adding the parrents
                node_desc = [
                    (p.NodeReadIndex(), myhash(p), myhash(p.T())==h_curr)
                    for p in parents[h_curr]]
            labels[h_curr] = (curr_node.NodeReadIndex(), node_desc)
        sorted_h = sorted(visited,
            key= lambda x: all_nodes[x].NodeReadIndex(), reverse=True)
        sorted_nodes = [all_nodes[h] for h in sorted_h]
        h_nodes = map(myhash, sorted_nodes)
        idx_nodes = map(lambda x: x.NodeReadIndex(), sorted_nodes)
        return parents, sorted_nodes, h_nodes,idx_nodes

def main():
    pass

if __name__=='__main__':
    main()
