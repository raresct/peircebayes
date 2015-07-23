import pickle
import numpy as np
from pprint import pprint

#theta_pb = np.load('/tmp/peircebayes/avg_samples.npz')
theta_pb = np.load('/home/rares/Desktop/peircebayes_all_no_sampling/last_sample.npz')
phi = theta_pb['arr_1']
print phi.shape

vocab = pickle.load(open('data/vocab.pkl', 'r'))
inv = dict((v, k) for k, v in vocab.iteritems())

axis = 1
index = list(np.ix_(*[np.arange(i) for i in phi.shape]))
index[axis] = phi.argsort(axis)
a = phi[index][:,-20:]
counts = np.rint(a/np.sum(a, axis=1).reshape(-1,1)*1000).tolist()
idx_l = index[axis][:,-20:].tolist()
words = [[inv[i] for i in subl] for subl in idx_l]
pprint(words)

strs = [' '.join([(w+' ')*int(c) for c,w in zip(sub_c, sub_w)])
    for sub_c, sub_w in zip(counts, words)]

#pprint(strs)
