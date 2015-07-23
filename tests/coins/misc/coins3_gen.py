import numpy as np
from scipy.stats import bernoulli
import pandas as pd

N = 5000

p1 = 0.7
p2 = 0.2
p3 = 0.9

c1 = bernoulli.rvs(p1, size=N)
c2 = bernoulli.rvs(p2, size=(N,2))
c3 = bernoulli.rvs(p3, size=(N,2))

s2 = np.sum(c2, axis=1)
s3 = np.sum(c3, axis=1)

A = np.vstack((c1, s2, s3, np.zeros(N))).T
for x in np.nditer(A, flags=['external_loop'], order='C', op_flags=['readwrite']):
    x[3] = x[2] if x[0] else x[1]
res = A[:, 3]

vals, counts = np.unique(res, return_counts=True)

with open('coins3_artificial.pb', 'w') as fout:
    fout.write('''
% aprob debug flags
%:- set_value(dbg_read,2).
%:- set_value(dbg_query,2).
%:- set_value(dbg_write,2).

''')
    for i,j in zip(vals, counts):
        fout.write('observe({}, {}).\n'.format(int(i+1), j))
    fout.write('''
pb_dirichlet(1.0, coin1, 2, 1).    
pb_dirichlet(1.0, coins23, 3, 2).

generate(X) :-
    coin1(1, 1),
    coins23(X, 1).
generate(X) :-
    coin1(2, 1),
    coins23(X, 2).

pb_plate(
   [observe(X, Count)],
   Count,
   [generate(X)]).
''')
