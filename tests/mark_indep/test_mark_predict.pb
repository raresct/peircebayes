
pb_dirichlet([2.143,  3.853], theta1, 2, 1).
pb_dirichlet([1.144,  2.998], theta2, 2, 1).
pb_dirichlet([1.0,  2.998], theta3, 2, 1).
pb_dirichlet([1.0,  1.0], theta4, 2, 1).
pb_dirichlet([1.0,  1.0], theta5, 2, 1).


r :- s, q, theta3(2,1).
t :- s, \+ q, theta4(2,1).
p :- \+ s, q, theta5(2,1).

s :- theta1(2,1).
q :- theta2(2,1).

pb_plate([], 1, [r]).
pb_plate([], 1, [t]).
pb_plate([], 1, [p]).



