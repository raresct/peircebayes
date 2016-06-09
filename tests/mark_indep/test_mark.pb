
% see if this can be done better in BASIL

pb_dirichlet(1.0, theta1, 2, 1).
pb_dirichlet(1.0, theta2, 2, 1).
%pb_dirichlet(1.0, theta3, 2, 1).
%pb_dirichlet(1.0, theta4, 2, 1).
%pb_dirichlet(1.0, theta5, 2, 1).


r :- theta3(2,1), s, q.
t :- theta4(2,1), s, \+ q.
p :- theta5(2,1), \+ s, q.

s :- theta1(2,1).
q :- theta2(2,1).

pb_plate([], 2, [r]).
pb_plate([], 1, [\+ t]).
pb_plate([], 1, [\+ p]).
