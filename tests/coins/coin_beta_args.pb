observe(1, 10).

pb_dirichlet([5.0, 85.0], tail_coin, 2, 1).

pb_plate(
    [observe(Val, Count)],
    Count,
    [generate(Val)]
).

generate(Val) :-
    tail_coin(Val,1).


