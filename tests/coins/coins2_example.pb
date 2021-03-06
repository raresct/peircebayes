
% aprob debug flags
:- set_value(dbg_read,2).
:- set_value(dbg_query,2).
:- set_value(dbg_write,2).

observe(2, 2, 3151).
observe(2, 1, 346).
observe(1, 2, 310).
observe(1, 1, 1193).

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
   
   
