import numpy as np
from scipy.stats import bernoulli
import pandas as pd

N = 2000000

p1 = 0.7
p2 = 0.2
p3 = 0.9
p4 = 0.6
p5 = 0.5

c1 = bernoulli.rvs(p1, size=N).reshape(-1,1)
c2 = bernoulli.rvs(p2, size=(N,3))
c3 = bernoulli.rvs(p3, size=(N,3))
c4 = bernoulli.rvs(p4, size=(N,3))
c5 = bernoulli.rvs(p5, size=(N,3))

s2 = np.sum(c2, axis=1).reshape(-1,1)
s3 = np.sum(c3, axis=1).reshape(-1,1)
s4 = np.sum(c4, axis=1).reshape(-1,1)
s5 = np.sum(c5, axis=1).reshape(-1,1)
A = np.hstack((c1, s2, s3, s4, s5, np.zeros((N,2))))

def coins4(x):
    if x[0]:
        x[5] = x[3]
        x[6] = x[4]
    else:
        x[5] = x[1]
        x[6] = x[2]
    return x[5:]

res = np.apply_along_axis(coins4, axis=1, arr=A)

df = pd.DataFrame(np.c_[res, range(N)])
g = df.groupby([0,1])[2].apply(lambda x:len(x.unique()))

with open('coins4_artificial.pb', 'w') as fout:
    fout.write('''
% Coin Toss Model 4 (artificial)    

''')
    for i,j in reversed(zip(g,g.keys())):
        j_str = [int(el+1) for el in j]
        fout.write('observe({}, {}, {}).\n'.format(j_str[0], j_str[1], i))
    fout.write('''
pb_dirichlet(1.0, coin1, 2, 1).    
pb_dirichlet(1.0, coins25, 4, 4).

generate(Val1, Val2) :-
    coin1(1, 1),
    coins25(Val1, 1),
    coins25(Val2, 2).
generate(Val1, Val2) :-
    coin1(2, 1),
    coins25(Val1, 3),
    coins25(Val2, 4).

pb_plate(
   [observe(Val1,Val2, Count)],
   Count,
   [generate(Val1, Val2)]).
''')
