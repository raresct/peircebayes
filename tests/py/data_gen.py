import numpy as np
import re
import os
import sys
import pickle
import itertools

from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer

def get_counts_cat(pref, cat):
    pattern = re.compile('.+\.txt')
    data_dir = join(pref,cat)
    files = [ join(data_dir,f)
        for f in listdir(data_dir)
        if isfile(join(data_dir,f)) and re.match(pattern, f)]
    docs = []
    sub_cats = []
    for fname in files:
        with open(fname, 'r') as f:
            for line in f:
                fields = line.split('\t')
                tokens = [t for t in ' '.join(fields[2:]).strip().split(' ')
                    if len(t)>1]
                docs.append(' '.join(tokens))
                sub_cats.append(fields[1].split(' '))
    return docs, sub_cats, len(docs)

def get_counts(pref):
    ns = []
    all_docs = []
    all_sub_cats = []
    for cat in ['q-fin', 'stat', 'q-bio', 'cs', 'physics']:
        docs, sub_cats, n_docs = get_counts_cat(pref, cat)
        ns.append(n_docs)
        all_docs += docs
        all_sub_cats += sub_cats
    vectorizer = CountVectorizer(stop_words='english')
    counts = vectorizer.fit_transform(all_docs)
    with open(join(pref, 'arxiv_info.txt'), 'w') as fout:
        #fout.write('Size of dataset before pre-processing:' +
        #    ' {} documents.\n'.format(len(data0)))
        fout.write('Size of dataset after pre-processing: {} documents\n'.format(
            counts.shape[0]))
        fout.write('Vocabulary: {} tokens\n'.format(counts.shape[1]))
        fout.write('Average document length: {} tokens\n'.format(
            np.average(np.sum(counts.toarray(), axis=1))))
        fout.write('Number of tokens: {} tokens\n'.format(
            np.sum(counts.toarray())))
    pickle.dump(vectorizer.vocabulary_, open(join(pref, 'vocab.pkl'), 'w'))
    pickle.dump(all_sub_cats, open(join(pref, 'sub_cats.pkl'), 'w'))
    pickle.dump(ns, open(join(pref, 'ns.pkl'), 'w'))
    return counts,vectorizer

def data_gen():
    pref = 'data'
    pb_obs = join(pref, 'arxiv_obs.py')
    pb_lda = join(pref, 'pb_lda.pb')
    pb_hlda = join(pref, 'pb_hlda.pb')
    pb_lda2 = join(pref, 'pb_lda2.pb')
    T = 25 # number of topics
    T2 = 5
    alpha = 50./float(T)
    alpha2 = 50./float(T2)
    beta = 0.1
   
    counts,vect = get_counts(pref)
    with open(pb_obs, 'w') as fout:
        fout.write('corpus = [\n')
        for d in range(counts.shape[0]):
            out_str = '['
            list_str = []
            crow = counts[d,:].tocoo()
            for term_idx,count in itertools.izip(crow.col, crow.data):
                list_str.append('({},{})'.format(term_idx, count))
            out_str += ','.join(list_str)
            out_str += ']'+',' if d <counts.shape[0]-1 else ']'
            fout.write(out_str)
        fout.write(']\n')
    print 'Done writing pb.'

def main():
    data_gen()

if __name__=='__main__':
    main()
