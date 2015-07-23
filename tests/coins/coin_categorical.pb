observe(1, 10).

pb_categorical(0.5, fair_coin, 2, 1).

pb_plate(
    [observe(Val, Count)],
    Count,
    [generate(Val)]
).

generate(Val) :-
    fair_coin(Val,1).

