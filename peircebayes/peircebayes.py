#!/usr/bin/env python2

"""
:synopsis: Module for peircebayes entry script.
"""

from pkg_resources import resource_filename

from formula_gen import PBModel
from knowledge_compilation import compile_k
from prob_inference import PBInference

import numpy as np

import subprocess
import sys
import os
import shutil
import argparse
import json
import time

# logging stuff
import logging
from logging import debug as logd

def rewrite_dir(dir_path):
    """
    Ovewrite the argument directory path.

    :param dir_path: directory path to overwrite
    :type dir_path: path
    :rtype: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def logical_inference(in_file, tmp_dir):
    """
    The logical inference part of the peircebayes pipeline.

    :param in_file: input file for peircebayes
    :type in_file: path
    :param tmp_dir: directory for results
    :type tmp_dir: path
    :rtype: None
    """
    in_file_name = os.path.basename(in_file)
    query_file_name = 'query.pl'
    aprob_file_name='aprob.pl'
    query_file = os.path.join(tmp_dir, query_file_name)
    aprob_out_name = 'aprob'
    aprob_out = os.path.join(tmp_dir, aprob_out_name)
    plate_out_name = 'plates'
    plate_out = os.path.join(aprob_out, plate_out_name)
    # installed
    aprob_file = resource_filename('peircebayes',aprob_file_name)
    # not installed
    #aprob_file = os.path.join('/home/rares/p/peircebayes/peircebayes/',
    #    aprob_file_name)
    copy_in_file_name = 'in_file.pl'
    copy_in_file = os.path.join(tmp_dir, copy_in_file_name)
    pl_log_file_name = 'aprob.log'
    pl_log_file = os.path.join(tmp_dir, pl_log_file_name)

    # create temporary dir and copy input to in_file.pl
    #rewrite_dir(tmp_dir) # this was moved to main function
    os.mkdir(aprob_out)
    os.mkdir(plate_out)
    shutil.copy(in_file, tmp_dir)
    shutil.move(os.path.join(tmp_dir, in_file_name), copy_in_file)

    # write query file
    write_query_file(query_file, aprob_file, copy_in_file)

    # call prolog
    with open(pl_log_file, 'w') as fout:
        #p1 = subprocess.Popen(['sicstus', '-l', query_file, '--noinfo'],
        #    stdout=fout, stderr=fout)
        # yap
        p1 = subprocess.Popen(['yap', '-l', query_file],
            stdout=fout, stderr=fout)
        p1.wait()


def write_query_file(query_file, aprob_file, copy_in_file):
    """
    Create query file for prolog. Used by :func:`logical_inference`.

    :param query_file: query file
    :type query_file: path
    :param aprob_file: path to aprob source
    :type aprob_file: path
    :param copy_in_file: path to a copy of in_file
    :type copy_in_file: path
    :rtype: None
    """
    query_str = 'load_pb'
    with open(query_file, 'w') as fout:
        fout.write('''
% load the Aprob prolog file
:- ['{}'].
% load the input file and query
:- {}('{}').
% stop prolog
:- halt.
'''.format(aprob_file, query_str, copy_in_file))


def parse_and_compile():
    """
    Parse files produced by :func:`logical_inference` and compile them to a :class:`formula_gen.PBModel`.

    :rtype: :class:`formula_gen.PBModel`
    """
    option_args = {
        'probs_file'    : '/tmp/peircebayes/aprob/out.probs',
        'dir_file'      : '/tmp/peircebayes/aprob/plates'
    }
    model = PBModel(option_args)
    compile_k(model)
    # prune empty bdds
    model.plates = [plate for plate in model.plates if not(plate.bdd is None)]
    #logd(model)
    return model

def probabilistic_inference(model, options):
    """
    The probabilistic inference part of the peircebayes pipeline. Should be called after :func:`parse_and_compile`.
    
    :param model: model built by knowledge compilation
    :type model: :class:`formula_gen.PBModel`
    :param options: options for inference
    :type options: dict
    :rtype: None
    """
    inf = PBInference(model)
    inf.gibbs_sampler_plates(**options)
    np.savez('/tmp/peircebayes/avg_samples', *inf.theta_avg)
    np.savez('/tmp/peircebayes/last_sample', *inf.theta)
    np.savez('/tmp/peircebayes/first_sample', *inf.first_sample)

def cl_parse():
    """
    Creates the command line argument parser.
    
    :rtype: :class:`argparse.ArgumentParser`    
    """
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_file",
        help="input pb file")
    parser.add_argument("-n", type=int,
        default = 100,
        help="number of samples")
    parser.add_argument("-b", "--burn", type=int,
        default = 0,
        help="number of burn-in iterations")
    parser.add_argument("-l", "--lag", type=int,
        default = 1,
        help="keep samples every l iterations (l=1 means no lag)")
    parser.add_argument("-s", "--seed", type=int,
        default = 1234,
        help="seed for numpy rng")
    parser.add_argument("-t", "--track", action="store_true",
        help="include metric information (see -m)")
    parser.add_argument("-d","--debug", action="store_true",
        help="include some debug information")
    parser.add_argument("-c","--config", action="store_true",
        help="load arguments from json as in_file")
    parser.add_argument("-m","--metric", choices=['joint', 'lda_ll'],
        default = 'joint',
        help="choose metric to track")
    parser.add_argument("-a","--algo", choices=['ungibbs', 'amcmc'],
        default = 'ungibbs',
        help='''choose sampling algorithm. 
    'ungibbs' is uncollapsed gibbs sampling on x and theta
    'amcmc' is ancestral mcmc, i.e. sample x and theta is E[Dir(alpha+x)]
        ''')    
    return parser

def peircebayes():
    """
    Main function. Parses args, then runs the peircebayes pipeline: 
    
    1. :func:`logical_inference`
    2. :func:`parse_and_compile`
    3. :func:`probabilistic_inference`

    :rtype: None
    """
    
    cl_args = cl_parse().parse_args()
    args = json.load(cl_args.in_file) if cl_args.config else cl_args
    tmp_dir = '/tmp/peircebayes' # TODO make this an option... maybe?
    rewrite_dir(tmp_dir)
    if args.debug:
        logging.basicConfig(filename='/tmp/peircebayes/pb.log',
            level=logging.DEBUG)
    np.random.seed(args.seed)
    gibbs_options = {
        'n'         : args.n,
        'burn_in'   : args.burn,
        'lag'       : args.lag,
        'track_ll'  : args.track,
        'in_file'   : args.in_file,
        'metric'    : args.metric,
        'algo'      : args.algo
    }
    logd('### Started Logical Inference ###')
    t_li_start = time.time()
    logical_inference(args.in_file, tmp_dir)
    t_li_end = time.time()
    logd('Finished Logical Inference in:\n{} seconds'.format(t_li_end-t_li_start))
    logd('### Started Knowledge Compilation ###')
    t_kc_start = time.time()
    model = parse_and_compile()
    t_kc_end = time.time()
    logd('Finished Knowledge Compilation in:\n{} seconds'.format(t_kc_end-t_kc_start))
    logd('### Started Probabilistic Inference ###')
    t_pi_start = time.time()
    probabilistic_inference(model, gibbs_options)
    t_pi_end = time.time()
    logd('Finished Probabilistic Inference in:\n{} seconds'.format(t_pi_end-t_pi_start))

def main():
    peircebayes()

if __name__=='__main__':
    main()
