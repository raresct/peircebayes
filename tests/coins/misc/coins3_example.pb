
% aprob debug flags
%:- set_value(dbg_read,2).
%:- set_value(dbg_query,2).
%:- set_value(dbg_write,2).

observe(1, 977).
observe(2, 1067).
observe(3, 2956).

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
