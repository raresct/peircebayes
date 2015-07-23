import numpy as np
from scipy.stats import bernoulli
from numpy.random import choice
import pandas as pd

N = 2000

p1 = 0.7 # choose doc

p2 = 0.2 # choose bag for doc 1
p3 = 0.9 # choose bag for doc 2

p4 = 0.8 # choose topic from bag 1
p5 = 0.6 # choose topic from bag 2

p6 = [0.7, 0.1, 0.1, 0.1] # choose word from topic
p7 = [0.1, 0.1, 0.1, 0.7]

c1 = bernoulli.rvs(p1, size=N).reshape(-1,1) 
c2 = bernoulli.rvs(p2, size=N).reshape(-1,1) 
c3 = bernoulli.rvs(p3, size=N).reshape(-1,1)
c4 = bernoulli.rvs(p4, size=N).reshape(-1,1)
c5 = bernoulli.rvs(p5, size=N).reshape(-1,1)

c6 = choice(len(p6), N, p6).reshape(-1,1) # choose word from topic
c7 = choice(len(p7), N, p7).reshape(-1,1)


A = np.hstack((c1, c2, c3, c4, c5, c6, c7, np.zeros((N,2))))

def coins5(x):
    x[7] = x[0]
    if not x[0]: # first doc
        if not x[1]: # first bag
            if not x[3]: # first topic
                x[8] = x[5]
            else: # second topic
                x[8] = x[6]
        else: # second bag
            if not x[4]: # first topic
                x[8] = x[5]
            else: # second topic
                x[8] = x[6]
    else: # second doc
        if not x[2]: # first bag
            if not x[3]: # first topic
                x[8] = x[5]
            else: # second topic
                x[8] = x[6]
        else: # second bag
            if not x[4]: # first topic
                x[8] = x[5]
            else: # second topic
                x[8] = x[6]
    return x[7:]

res = np.apply_along_axis(coins5, axis=1, arr=A)

df = pd.DataFrame(np.c_[res, range(N)])
g = df.groupby([0,1])[2].apply(lambda x:len(x.unique()))

with open('coins5_artificial.pb', 'w') as fout:
    fout.write('''
% Coin Toss Model 5 (artificial)    

:- enforce_labeling(true).   

''')
    for i,j in reversed(zip(g,g.keys())):
        j_str = [int(el+1) for el in j]
        fout.write('observe({}, {}, {}).\n'.format(j_str[0], j_str[1], i))
    fout.write('''

pb_dirichlet(1.0, doc, 2, 1). % for debug, 2 documents
pb_dirichlet(1.0, bag, 2, 2). % 2 bags for 2 documents     
pb_dirichlet(1.0, topic, 2, 2). % 2 topics for 2 bags     
pb_dirichlet(1.0, token, 4, 2). % 4 words for 2 topics

generate(Doc, Token) :-
    Bag in 1..2,
    Topic in 1..2,
    doc(Doc, 1), % for debug
    bag(Bag, Doc),
    topic(Topic, Bag),
    token(Token, Topic).

pb_plate(
   [observe(Doc, Token, Count)],
   Count,
   [generate(Doc, Token)]).
''')
