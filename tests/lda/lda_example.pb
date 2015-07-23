% needed to ground constraints
:- enforce_labeling(true).

% pb debug flags
%:- set_value(dbg_read,2).
%:- set_value(dbg_query,2).
%:- set_value(dbg_write,2).

observe(d(1), [ (w(1), 4), (w(4), 2) ]).
observe(d(2), [ (w(3), 1), (w(4), 5) ]).
observe(d(3), [ (w(1), 4), (w(2), 2) ]).

pb_dirichlet(1.0, mu, 2, 3).
pb_dirichlet(1.0, phi, 4, 2).

generate(Doc, Token) :-
    Topic in 1..2, 
    mu(Topic, Doc), 
    phi(Token, Topic).

pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [generate(Doc, Token)]
).
