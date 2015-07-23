
import numpy as np
import pickle
import os
import itertools
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

def can_be_noun(test_word):
    synsets = nltk.corpus.wordnet.synsets(test_word)
    if len(synsets) == 0:
        return True
    for s in synsets:
        if s.pos == 'n':
            return True
    return False

def get_counts(pref):
    cats = [ 'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
    news_all = fetch_20newsgroups(subset='all',
        remove=('headers', 'footers', 'quotes'), categories=cats)
    data0 = news_all.data
    data1 = [d for d in data0 if d and not d.isspace()]
    # lowercase, lemmatize and remove stop words
    tokens_l = [nltk.word_tokenize(d) for d in data1]
    wnl = nltk.WordNetLemmatizer()
    stop = nltk.corpus.stopwords.words('english')
    data2 = [ ' '.join([wnl.lemmatize(t.lower())
        for t in tokens if t.lower() not in stop
            and len(t)>2]) #and can_be_noun(t.lower())])
        for tokens in tokens_l]
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(data2)
    with open(os.path.join(pref, 'news20_info.txt'), 'w') as fout:
        fout.write('Size of dataset before pre-processing:' +
            ' {} documents.\n'.format(len(data0)))
        fout.write('Size of dataset after pre-processing: {} documents\n'.format(
            counts.shape[0]))
        fout.write('Vocabulary: {} words\n'.format(counts.shape[1]))
        fout.write('Average document length: {} words\n'.format(
            np.average(np.sum(counts.toarray(), axis=1))))
    pickle.dump(vectorizer.vocabulary_, open(os.path.join(pref, 'vocab.pkl'), 
        'w'))
    return counts,vectorizer

def write_plda(counts, vect, file_str):
    with open(file_str, 'w') as fout:
        for d in range(counts.shape[0]):
            fout.write(' '.join(
                ['{} {}'.format(term.decode('ascii', 'ignore'), counts[d,i])
                for term,i in vect.vocabulary_.iteritems() if counts[d,i]>0])
                +'\n')
    print 'Done writing plda.'

def write_pb(counts, vect, file_str, T, alpha, beta):
    D = counts.shape[0]
    V = counts.shape[1]
    with open(file_str, 'w') as fout:
        fout.write('''
% LDA newscomp

% needed to ground constraints
:- enforce_labeling(true).

''')
        fout.write('''
% prob distribs
pb_dirichlet({}, theta, {}, {}).
pb_dirichlet({}, phi, {}, {}).

% plate
pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [Topic in 1..{}, theta(Topic,Doc), phi(Token,Topic)]
).

'''.format(alpha, T, D, beta, V, T, T))
        for d in range(counts.shape[0]):
            out_str = 'observe(d({}), ['.format(d+1)
            list_str = []
            crow = counts[d,:].tocoo()
            for term_idx,count in itertools.izip(crow.col, crow.data):
                list_str.append('(w({}),{})'.format(term_idx+1, count))
            out_str += ','.join(list_str)
            out_str += ']).\n'
            fout.write(out_str)
    print 'Done writing pb.'

def write_pb_constraints(counts, vect, file_str, T, alpha, beta):
    D = counts.shape[0]
    V = counts.shape[1]
    with open(file_str, 'w') as fout:
        fout.write('''
% LDA newscomp + constraints

% needed to ground constraints
:- enforce_labeling(true).

''')
        # write seed constraints
        words_t1 = ['hardware', 'machine', 'memory', 'cpu']
        words_t2 = ['software', 'program', 'version', 'shareware']
        str_t1 = '\n'.join(['seed({}, [1]).'.format(vect.vocabulary_[word]+1)
            for word in words_t1])
        str_t2 = '\n'.join(['seed({}, [2]).'.format(vect.vocabulary_[word]+1)
            for word in words_t2])
        fout.write(''' 
seed_naf(Token) :- seed(Token, _).

{}

{}             
        '''.format(str_t1, str_t2))
        fout.write('''
% prob distribs
pb_dirichlet({}, theta, {}, {}).
pb_dirichlet({}, phi, {}, {}).

% plate
pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList), 
        \+ seed_naf(Token)],
    Count,
    [Topic in 1..{}, theta(Topic,Doc), phi(Token,Topic)]
).

pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList), 
        seed_naf(Token)],
    Count,
    [seed(Token, TopicList), member(Topic, TopicList),
        theta(Topic,Doc), phi(Token,Topic)]
).

'''.format(alpha, T, D, beta, V, T, T))
        for d in range(counts.shape[0]):
            out_str = 'observe(d({}), ['.format(d+1)
            list_str = []
            crow = counts[d,:].tocoo()
            for term_idx,count in itertools.izip(crow.col, crow.data):
                list_str.append('(w({}),{})'.format(term_idx+1, count))
            out_str += ','.join(list_str)
            out_str += ']).\n'
            fout.write(out_str)
    print 'Done writing pb constraints.'


def main():
    pref = 'data'
    plda_f = 'news20comp.plda'
    pb_f = 'news20comp.pb'
    pb_c_f = 'news20comp_c.pb'

    T = 10 # number of topics
    alpha = 1.
    beta = 1.

    plda_fstr = os.path.join(pref,plda_f)
    pb_fstr = os.path.join(pref,pb_f)
    pb_c_fstr = os.path.join(pref,pb_c_f)
    counts,vect = get_counts(pref)
    #write_plda(counts, vect, plda_fstr)
    write_pb(counts, vect, pb_fstr, T, alpha, beta)
    write_pb_constraints(counts, vect, pb_c_fstr, T, alpha, beta)

if __name__=='__main__':
    main()

