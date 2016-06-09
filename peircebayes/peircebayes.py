#!/usr/bin/env python2

"""
:synopsis: Module for peircebayes entry script.
"""

from pkg_resources import resource_filename

from formula_gen import PBModel
from knowledge_compilation import compile_k, compile_k_parallel
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
    :type dir_path: str
    :rtype: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def l_abduction(in_file, tmp_dir, out_folder):
    """
    The logical inference part of the peircebayes pipeline.

    :param in_file: (path to) input file for peircebayes
    :type in_file: str
    :param tmp_dir: (path to) directory for results
    :type tmp_dir: str
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
    write_query_file(query_file, aprob_file, copy_in_file, tmp_dir, out_folder)

    # call prolog
    with open(pl_log_file, 'w') as fout:
        #p1 = subprocess.Popen(['sicstus', '-l', query_file, '--noinfo'],
        #    stdout=fout, stderr=fout)
        # yap
        p1 = subprocess.Popen(['yap', '-l', query_file],
            stdout=fout, stderr=fout)
        p1.wait()

def l_py(in_file, tmp_dir):
    pass

def l_asp(in_file, tmp_dir):
    pass


def write_query_file(query_file, aprob_file, copy_in_file, tmp_dir, out_folder):
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
    plate_files_name = ('out' if out_folder=='/tmp/peircebayes' else
        os.path.basename(os.path.normpath(out_folder))+'_')
    with open(query_file, 'w') as fout:
        fout.write('''
% load the Aprob prolog file
:- ['{}'].
% set probs_file flag
:-abduction:my_set_value(probs_file, '{}').
:-abduction:my_set_value(n_file, '{}').
:-abduction:my_set_value(plate_file, '{}').
% load the input file and query
:- {}('{}').
% stop prolog
:- halt.
'''.format(aprob_file, os.path.join(tmp_dir, 'aprob', 'out.probs'),
    os.path.join(tmp_dir, 'aprob', 'plates', 'n_plates'),
    os.path.join(tmp_dir, 'aprob', 'plates', plate_files_name),
    query_str, copy_in_file))


def parse_and_compile(debug, tmp_dir, parallel_kc):
    """
    Parse files produced by :func:`logical_inference` and compile them to a :class:`formula_gen.PBModel`.

    :rtype: :class:`formula_gen.PBModel`
    """
    option_args = {
        'probs_file'    : os.path.join(tmp_dir, 'aprob', 'out.probs'),
        'dir_file'      : os.path.join(tmp_dir, 'aprob', 'plates')
    }
    start = time.time()
    model = PBModel(option_args)
    logd('bla')
    #logd(model.plates)
    logd('formula_gen takes:  {}'.format(time.time()-start))
    logd('parallel_kc: {}'.format(parallel_kc))
    start = time.time()
    if parallel_kc:
        compile_k_parallel(model, debug, tmp_dir)
    else:
        compile_k(model, debug, tmp_dir)
    logd('kc takes:  {}'.format(time.time()-start))
    ## prune empty bdds
    ##model.plates = [plate for plate in model.plates if not(plate.bdd is None)]
    #logd(model)
    return model

def probabilistic_inference(model, options, tmp_dir):
    """
    The probabilistic inference part of the peircebayes pipeline.
    Should be called after :func:`parse_and_compile`.

    :param model: model built by knowledge compilation
    :type model: :class:`formula_gen.PBModel`
    :param options: options for inference
    :type options: dict
    :rtype: None
    """
    inf = PBInference(model)
    inf.infer(options)
    if options['algo']=='predict':
        np.savez(os.path.join(tmp_dir, 'avg_beta'), *inf.avg_beta)
    else:
        np.savez(os.path.join(tmp_dir, 'avg_samples'), *inf.theta_avg)
        np.savez(os.path.join(tmp_dir, 'last_sample'), *inf.last_theta)
        #np.savez('/tmp/peircebayes/first_sample', *inf.first_sample)
        np.savez(os.path.join(tmp_dir,'alpha_avg'), *inf.alpha_avg)
        if hasattr(inf, 'total_ll'):
            np.savez(os.path.join(tmp_dir,'total_ll'), inf.total_ll)
        #np.savez('/tmp/peircebayes/avg_beta', *inf.avg_beta)
    if options['track_ll']:
        import matplotlib.pyplot as plt
        np.savez(os.path.join(tmp_dir,'lls'), **{'lls':np.array(inf.lls)})
        plt.figure('lls')
        plt.savefig(os.path.join(tmp_dir,'lls.pdf'), format='pdf')

def probabilistic_inference2(model, options, tmp_dir):
    from prob_inference_dev import test
    distrib_cat_vars = [(distrib[1], distrib[2]) for distrib in model.distribs]
    distrib_args = [distrib[0] for distrib in model.distribs]
    bdds = [plate.bdd for plate in model.plates]
    cat_lists = [plate.cat_list for plate in model.plates]
    kids = [plate.kid for plate in model.plates]
    plates = [plate.plate for plate in model.plates]
    test(distrib_cat_vars, distrib_args, bdds, cat_lists, kids, plates)

def cl_parse():
    """
    Creates the command line argument parser.

    :rtype: :class:`argparse.ArgumentParser`
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_file",
        nargs='?',
        default=None,
        help="input pb file")
    parser.add_argument("-o", "--output", type=str,
        default = '/tmp/peircebayes',
        help="output directory")
    parser.add_argument("-n", type=int,
        default = 100,
        help="number of samples")
    parser.add_argument("--abduction", action="store_true",
        help="do only logical inference, i.e. abduction")
    parser.add_argument("--learning", action="store_true",
        help="do only parameter learning (you need to specify with -o the result of abduction)")
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
        help="track metric per iteration")
    parser.add_argument("-m", "--metric", choices = ['joint', 'lda', 'total', 'joint_un'],
        default ='joint',
        help="choose metric to track")
    parser.add_argument("-d","--debug", action="store_true",
        help="include some debug information")
    parser.add_argument("-c","--config", action="store_true",
        help="load arguments from json as in_file")
    parser.add_argument("-a","--algo",
        choices=['ugs', 'amcmc', 'cgs', 'predict'],
        default = 'ugs',
        help='''choose sampling algorithm.
    'ugs' is uncollapsed gibbs sampling on x and theta
    'amcmc' is ancestral mcmc, i.e. sample x and theta is E[Dir(alpha+x)]
    'cgs' is collapsed gibbs sampling
    'predict' is sample theta then record backward prob on bdd
        ''')
    parser.add_argument("-w", "--wrapper",
        choices = ['extension', 'abduction', 'py', 'asp'],
        default ='extension',
        help='''choose wrapper for logical inference:
    abduction: the original PB wrapper (Prolog-like), extensions=[.pb, .pl]
    asp: answer set programming (ASP) wrapper, extensions=[.asp]
    py: python wrapper, extensions=[.py]
    extension: choose wrapper based on input file extension
    ''')
    parser.add_argument('--pkc', action="store_true",
        help = '''parallel knowledge compilation, useful when there are lots of plates''')
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

    if not args.in_file and not args.learning:
        print('You need to specify an input file for abduction!')
        return

    tmp_dir = args.output
    if not args.learning:
        rewrite_dir(tmp_dir)
    if args.debug:
        logging.basicConfig(filename=os.path.join(tmp_dir,'pb.log'),
            level=logging.DEBUG)
    np.random.seed(args.seed)
    infer_options = {
        'n'         : args.n,
        'burn_in'   : args.burn,
        'lag'       : args.lag,
        'track_ll'  : args.track,
        'metric'    : args.metric,
        'in_file'   : args.in_file,
        'algo'      : args.algo,
    }

    if not args.learning:
        if args.wrapper=='extension':
            ext = os.path.splitext(args.in_file)[1][1:]
        else:
            ext = args.wrapper
        ext_d = {
            'abduction': lambda x: l_abduction(*x),
            'pb': lambda x: l_abduction(*x),
            'pl': lambda x: l_abduction(*x),
            'py': lambda x: l_py(*x),
            'asp': lambda x: l_asp(*x)
        }
        if ext in ext_d:
            logical_inference = ext_d[ext]
        else:
            print('Wrong file extension!'+
                'See help (-h) on the -w option for supported extensions.')
            return

        logd('### Started Logical Inference ###')
        t_li_start = time.time()
        logical_inference((args.in_file, tmp_dir, args.output))
        t_li_end = time.time()
        logd('Finished Logical Inference in:\n{} seconds'.format(t_li_end-t_li_start))
    if not args.abduction:
        logd('### Started Knowledge Compilation ###')
        t_kc_start = time.time()
        model = parse_and_compile(args.debug, tmp_dir, args.pkc)
        t_kc_end = time.time()
        logd('Finished Knowledge Compilation in:\n{} seconds'.format(t_kc_end-t_kc_start))
        logd('### Started Probabilistic Inference ###')
        t_pi_start = time.time()
        probabilistic_inference(model, infer_options, tmp_dir)
        t_pi_end = time.time()
        logd('Finished Probabilistic Inference in:\n{} seconds'.format(t_pi_end-t_pi_start))

def main():
    peircebayes()

if __name__=='__main__':
    main()
