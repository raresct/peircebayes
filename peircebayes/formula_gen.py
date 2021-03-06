#!/usr/bin/env python2

"""
:synopsis: Module for parsing abductive solutions and formula generation. BDDs are compiled in :mod:`knowledge_compilation`.
"""

import os
import numpy as np
import copy
import re

# logging stuff
import logging
from logging import debug as logd

class PBPlate:
    """
    Container for:

    * plate: NXMX3 array, where

        * N = number of data points
        * M = number of propositional variables
        * plate[N,M] = (i,j,k), where

            * i,j = index of the random variable
            * k = category of the random variable

    * reps: NX1 array, where

        * reps[N] = how many times the data point is repeated

    * cat_list: NX1 list, where

        * cat_list[N] is a dictionary {(i,j) : L}, where

            * L = list of categories that the variable indexed by i and j takes in some data point N

    * kid: NXM array, where:

        * assume plate[N,M] = i,j,k, then kid[N,M] = cat_list[N][(i,j)].index(k), i.e. the index of the category of variable i,j in the cat_list for data point N

    * f_str: the string of the formula, with the following operators:

        * '&' - AND
        * '|' - OR
        * '~' - NOT

    * bdd: A BDD object created after knowledge compilation (default None)
    """
    def __init__(self, plate, reps, kid, cat_list, f_str, bdd=None):
        self.plate      = plate
        self.reps       = reps
        self.cat_list   = cat_list
        self.kid        = kid
        self.f_str      = f_str
        self.bdd        = bdd

class PBModel:
    """
    Instances of this class are passed to the probabilistic inference. Takes a dictionary option_args with keys:

    * 'probs_file'    : string of path to the pb file containing definitions of probability distributions
    * 'dir_file'      : string of path to the pb directory containing pb plate files

    and a constant defined in constructor:

    * frozen_types    : a list of types of frozen distributions

    and parses the files to produce:

    * plates : a list of P :class:`PBPlate` objects, where

        * P = number of plates

    * distribs : a list of NDist tuples (args, n_cat, n_vars, frozen), where

        * NDist   = number of probability distributions
        * args    = vector of (hyper)parameters
        * n_cat   = number of categories
        * n_vars  = number of distributions sampled from distrib_args
        * frozen  = boolean, if True, distribution is frozen
    """
    def __init__(self, option_args):
        self.probs_file     = option_args['probs_file']
        self.dir_file       = option_args['dir_file']
        self.frozen_types   = ['categorical']
        self.parse()

    def __str__(self):
        return '\n'.join(['''
Plate {}:
    Plate indexes:
    {}
    Plate reps:
    {}
    Plate cat_list:
    {}
    Plate kid:
    {}
    Plate f_str:
    {}
'''.format(i, self.plates[i].plate, self.plates[i].reps,
    self.plates[i].cat_list, self.plates[i].kid, self.plates[i].f_str)
    for i in range(len(self.plates))])

    def parse(self):
        """
        Main function. Called by __init__.

        :rtype: None
        """
        try:
            self.parse_probs()
            #n_plate_file = os.path.join(self.dir_file, 'n_plates')
            #with open(n_plate_file, 'r') as fin:
            #    n_plates = int(fin.read().strip())
            self.plates = [PBPlate(*self.parse_plate(f_name))
                for f_name in os.listdir(self.dir_file) if f_name.endswith('.plate')]
        except IOError as e:
            logd('Parsing failed!')
            logging.exception('Parsing failed: '+str(e))

    def parse_plate(self, f_name):
        """
        Parses a plate file. Returns a tuple of plate, reps, kid, cat_list, f_str to create a :class:`PBPlate` object. Called by :func:`parse`.

        :param i: index of the plate
        :type i: int
        :rtype: tuple
        """
        #plate_file = os.path.join(self.dir_file,'out{}.plate'.format(ii))
        plate_file = os.path.join(self.dir_file, f_name)
        logd(plate_file)
        with open(plate_file, 'r') as fin:
            plate = []
            kid = []
            reps = []
            cat_list = []
            for line in fin:
                line_split = line.split(';')
                line_sols, line_reps = line_split[:-1], line_split[-1]
                reps.append(int(line_reps))
                cat_d = {}
                plate_row = set()
                for line_idxs in line_sols:
                    for idxs in line_idxs.split('.'):
                        i, j, k = [int(idx) for idx in idxs.split(',')]
                        if cat_d.has_key((i,j)):
                            if k not in cat_d[(i,j)]:
                                cat_d[(i,j)].append(k)
                        else:
                            cat_d[(i,j)] = [k]
                        plate_row.add((i,j,k))
                # sort plate_row
                plate_row = sorted(list(plate_row),
                    key = lambda x: (x[0], x[1], x[2]))
                # sort cat_d and remove last choices from plate_row
                # if necessary (>1 choice)
                # however, keep a copy of plate_row to generate
                # the formula from
                plate_row_f = copy.copy(plate_row)
                for (i,j) in cat_d:
                    cat_d[(i,j)] = sorted(cat_d[(i,j)])
                    if len(cat_d[(i,j)])>1:
                        last_k = cat_d[(i,j)][-1]
                        plate_row.remove((i, j, last_k))
                plate.append(plate_row)
                cat_list.append(cat_d)
                kid_row = []
                for i,j,k in plate_row:
                    kid_row.append(cat_d[(i,j)].index(k))
                kid.append(kid_row)
            # convert to numpy
            plate = np.array(plate)
            reps = np.array(reps)
            kid = np.array(kid)
            #logd('=== i: {} ==='.format(ii))
            f_str = self.bool_gen(line_sols, plate_row_f, cat_d)
            #logd(plate.shape)
            #logd(reps.shape)
            #logd(kid.shape)
            #logd(len(cat_list))
            #logd(plate)
            #logd(reps)
            #logd(kid)
            #logd(cat_list)
            #logd('f_str')
            #logd(f_str)
            return plate, reps, kid, cat_list, f_str

    def bool_gen(self, line_sols, plate_row, cat_d):
        """
        Creates a boolean function string from a single plate row (more specifically, the last one). Called by :func:`parse_plate`.

        :param line_sols: encoding of a DNF. A list (disjunction) of strings (conjunction) in which '.' stands for AND, and elements of the conjunction are i,j,k .
        :type line_sols: string
        :param plate_row: distinct choices (i,j,k) in line_sols
        :type plate_row: set
        :param cat_d: see cat_list[N] in the doc of :class:`PBPlate`.
        :type cat_d: dict
        :rtype: string
        """
        #sols_l = [[[int( for] el for el in l.split('.')] for l in line_sols]
        #logd(sols_l)
        #return
        sols_s = '|'.join(line_sols)
        #logd(plate_row)
        sols_s = sols_s.replace('.', '&')
        #logd(sols_s)
        #logd(cat_d)
        dbg_vars = set()
        for (i,j,k) in plate_row:
            k_list = cat_d[(i,j)]
            len_k_list = len(k_list)
            # find idx
            idx = 0
            for (cat_i,cat_j) in cat_d:
                if cat_i<i or (cat_i == i and cat_j<j):
                    idx += np.max((len(cat_d[(cat_i,cat_j)])-1,1))
            start_idx = idx
            idx += k_list.index(k)
            # build ad_str
            ad_str = ''
            if len_k_list==1:
                ad_str = 'v{}'.format(idx)
            elif k == k_list[-1]:
                ad_str = '&'.join(['~v{}'.format(vid)
                    for vid in range(start_idx, start_idx+len_k_list-1)])
            else:
                ad_str = '&'.join(['~v{}'.format(vid)
                    for vid in range(start_idx, idx)]+
                    ['v{}'.format(idx)])
            '''
            logd((i,j,k))
            logd(start_idx)
            logd(idx)
            logd(ad_str)
            logd(k_list)
            '''
            dbg_vars.add((i,j,k,start_idx))
            sols_s = '|'.join([ '&'.join([ ad_str
                if ',' in s and [int(c) for c in s.split(',')] == [i,j,k]
                else s
                for s in conj.split('&')
              ])
                for conj in sols_s.split('|')
            ])
            #logd(sols_s)
        #with open('/tmp/peircebayes/dbg_vars.txt', 'w') as f:
        #    for (i,j,k,idx) in sorted(list(dbg_vars), key=lambda x: x[3]):
        #        f.write(','.join([str(el) for el in [i,j,k,idx]])+'\n')
        return sols_s

    def parse_probs(self):
        """
        Parses the distribution file. Creates distribs (see doc for :class:`PBModel`). Called by :func:`parse`.

        :rtype: None
        """

        distribs = []
        with open(self.probs_file, 'r') as fin:
            distribs = [self.parse_distrib(d) for d in fin]
        self.distribs = distribs

    def parse_distrib(self, d):
        """
        Parses a single distribution and returns a tuple of args, n_cat, n_vars, frozen (see doc for :class:`PBModel`). Called by :func:`parse_probs`.

        :param d: String describing a distribution.
        :type d: string
        :rtype: tuple
        """

        distrib_split = d.strip().split(' ')
        if len(distrib_split) < 3:
           raise Exception('Probs file incorrectly written!')
        distrib_type = distrib_split[-1]
        distrib_type = True if distrib_type in self.frozen_types else False
        distrib_n_vars = int(distrib_split[-2])
        distrib_n_cat = int(distrib_split[-3])
        distrib_args = np.array([float(p) for p in distrib_split[:-3]])
        if distrib_args.shape[0] == 1: # symmetric prior
            distrib_args = distrib_args[0]*np.ones(distrib_n_cat)
        elif distrib_args.shape[0] != distrib_n_cat:
            raise Exception('Probs file incorrectly written!')
        return (distrib_args, distrib_n_cat, distrib_n_vars, distrib_type)

def main():
    pass

if __name__=='__main__':
    main()
