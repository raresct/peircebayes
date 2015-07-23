#!/usr/bin/env python2

import numpy as np
import itertools

from pprint import pprint

# ugly hack bla bla
phi = 0.1
p2 = [phi/(1.+phi), 1./(1.+phi)]
p3 = [phi**2/(1.+phi+phi**2), phi/(1.+phi+phi**2), 1./(1.+phi+phi**2)]
p4 = [phi**3/(1.+phi+phi**2+phi**3), phi**2/(1.+phi+phi**2+phi**3),
    phi/(1.+phi+phi**2+phi**3), 1./(1.+phi+phi**2+phi**3)]

def main():
    N = 2000

    ins = ['a','b','c','d']
    # initialize
    counts = {}
    for i in itertools.permutations(ins):
        counts[i] = 0

    def sample_mallows(ins, p2, p3, p4):
        multcount2 = np.random.multinomial(1, p2, size=1)[0]
        idx2 = np.nonzero(multcount2)[0][0]
        multcount3 = np.random.multinomial(1, p3, size=1)[0]
        idx3 = np.nonzero(multcount3)[0][0]
        multcount4 = np.random.multinomial(1, p4, size=1)[0]
        idx4 = np.nonzero(multcount4)[0][0]
        sample = [ins[0]]
        sample.insert(idx2, ins[1])
        sample.insert(idx3, ins[2])
        sample.insert(idx4, ins[3])
        return tuple(sample)


    for i in range(N):
        counts[sample_mallows(ins, p2, p3, p4)] += 1

    pprint(counts)
    pprint(p2)
    pprint(p3)
    pprint(p4)

    obs_str = ''
    for sample,count in counts.iteritems():
        if count>0:
            list_str = '[{}]'.format(','.join([char for char in sample]))
            obs_str += 'observe({}, {}).\n'.format(list_str, count)

    with open('rim_artificial.pb', 'w') as fout:
        fout.write('''
% aprob debug flags
%:- set_value(dbg_read,2).
%:- set_value(dbg_query,2).
%:- set_value(dbg_write,2).

{}

pb_dirichlet(1.0, p2, 2, 1).
pb_dirichlet(1.0, p3, 3, 1).
pb_dirichlet(1.0, p4, 4, 1).

insert_rim([], ToIns, Ins,
    Pos, Ins1) :-
    append(Ins, [ToIns], Ins1),
    length(Ins1, Pos).
insert_rim([H|_T], ToIns, Ins,
    Pos, Ins1) :-
    nth1(Pos, Ins, H),
    nth1(Pos, Ins1, ToIns, Ins).
insert_rim([H|T] , ToIns, Ins,
    Pos, Ins1) :-
    \+member(H, Ins),
    insert_rim(T, ToIns, Ins,
        Pos, Ins1).

generate([H|T], Sample):-
    generate(T, Sample, [H], 2).

generate([], Sample, Sample, _Idx) :- print('ok'),nl.
generate([ToIns|T], Sample, Ins, Idx) :-
    % insert next element at Pos
    % yielding a new list Ins1
    append(_, [ToIns|Rest], Sample),
    insert_rim(Rest, ToIns, Ins,
        Pos, Ins1),
    print(Pos),nl,
    % build prob predicate in Pred
    number_chars(Idx, LIdx),
    append(['p'], LIdx, LF),
    atom_chars(F, LF),
    Pred =.. [F, Pos, 1],
    % call prob predicate
    pb_call(Pred),
    Idx1 is Idx+1,
    % recurse
    generate(T, Sample, Ins1, Idx1).

pb_plate(
   [observe(Sample, Count)],
   Count,
   [generate([{}], Sample)]).
'''.format(obs_str, ','.join(ins)))

if __name__=='__main__':
    main()


