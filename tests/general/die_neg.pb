:- enforce_labeling(true).
pb_dirichlet(1.0, die, 6, 1).
pb_dirichlet(1.0, coins, 2, 4).
pb_plate([], 1, [Face in 1..4, die(Face,1), \+ die(3,1), coins(1,Face)]).
