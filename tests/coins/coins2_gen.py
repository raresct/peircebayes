#/usr/bin/env python2

import numpy as np
from scipy.stats import bernoulli
import pandas as pd
import logging
from logging import debug as logd

# it's an ugly hack, but I'm too lazy to make classes
p1 = 0.7
p2 = 0.2
p3 = 0.9

def main():
    N = 5000

    c1 = bernoulli.rvs(p1, size=N)
    c2 = bernoulli.rvs(p2, size=N)
    c3 = bernoulli.rvs(p3, size=N)

    res = (~c1 & c2)|(c1&c3)

    T = np.sum(res)
    F = np.size(res)-np.sum(res)

    logd('{}/{} \t {}'.format(T,N,T/float(N)))

    df = pd.DataFrame(np.c_[c1,res, range(N)])
    g = df.groupby([0,1])[2].apply(lambda x:len(x.unique()))

    with open('coins2_artificial.pb', 'w') as fout:
        fout.write('''
% aprob debug flags
%:- set_value(dbg_read,2).
%:- set_value(dbg_query,2).
%:- set_value(dbg_write,2).
''')
        for i,j in reversed(zip(g,g.keys())):
            j_str = [2 if el==1 else 1 for el in j]
            fout.write('observe({}, {}, {}).\n'.format(j_str[0], j_str[1], i))
        fout.write('''
pb_dirichlet(1.0, toss, 2, 3).

generate(1, Val) :-
    toss(1, 1),
    toss(Val, 2).
generate(2, Val) :-
    toss(2, 1),
    toss(Val, 3).

pb_plate(
   [observe(Val1, Val2, Count)],
   Count,
   [generate(Val1, Val2)]).
''')

if __name__=='__main__':
    main()
