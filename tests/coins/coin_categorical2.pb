
observe(1, 1).
observe(2, 1).

pb_categorical(0.5, fair_coin, 2, 1).
pb_categorical([0.7, 0.3], trick1, 2, 1).
pb_categorical([0.9, 0.1], trick2, 2, 1).

pb_plate(
    [observe(Val, Count)],
    Count,
    [generate(Val)]
).

generate(Val) :-
    fair_coin(1,1),
    trick1(Val, 1).
generate(Val) :-
    fair_coin(2,1),
    trick2(Val, 1).    

