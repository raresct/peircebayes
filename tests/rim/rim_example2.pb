
% aprob debug flags
%:- set_value(dbg_read,2).
%:- set_value(dbg_query,2).
%:- set_value(dbg_write,2).

observe([a,b,d,c], 144).
observe([b,c,d,a], 1).
observe([b,c,a,d], 12).
observe([b,d,a,c], 1).
observe([c,a,b,d], 20).
observe([b,a,d,c], 19).
observe([a,d,b,c], 22).
observe([a,c,d,b], 13).
observe([a,b,c,d], 1455).
observe([c,a,d,b], 2).
observe([b,a,c,d], 159).
observe([a,d,c,b], 2).
observe([d,a,c,b], 1).
observe([a,c,b,d], 148).
observe([c,b,a,d], 1).


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
   [generate([a,b,c,d], Sample)]).
