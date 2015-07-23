
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Aprob - a probabilistic abduction system on top of  A system  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
 * by Calin Rares Turliuc
 * ct1810@imperial.ac.uk
 * 2015
 *
 * Credits for the A system implementation:
 */
/**
 * @author  Jiefei Ma
 * @date        Feb. 2011 (initial version)
 * Department of Computing, Imperial College London
 *
 */



% # ===================== Documentation ======================
%
% 1. Syntax for user input files:
%
% * abducible, defined (a.k.a. non-abducible) and builtin are predicates
%   e.g. holds(has(jeff, apple), 3), member(X, [2,3])
%
% * a negative literal is "\+ A" where "A" is an abducible, defined or builtin
%
% * a finite domain constraint is one used in "library(clpfd)"
%   e.g. X in 1..3, Y #< Z - 3, X #\= Y
%
% * a real domain constraint is one used in "library(clpr)"
%   e.g. {X + 3 = Y -2}
%
% * an equality is "X = Y", an inequality is "X =/= Y".
%
% 2. Each input file should specify an abductive framework:
%
% * an abducible is declared as:
%
%   abducible(happens(_,_)).
%
% * a builtin is declared as:
%
%   builtin(member(_,_)).
%
% * a rule is specified as:
%
%   holds(F,T) :-
%     happens(A, T1),
%     T1 #< T,
%     initiates(A, F, T1),
%     \+ clipped(F, T1, T).
%
% * an integrity is specified as:
%
%   ic :- happens(A1, T), happens(A2, T), A1=/=A2.
%
% 3. Loading and querying the System
%
% * Load a user defined abductive theory of <P, A, B, I>, where "P" is a
%   set of rules called the background knowledge, "A" is the set of abducible
%   predicates, "B" is the set of builtin predicates, and "I" is the set of
%   integrity constraints,  e.g.
%
%     ?- load_theory('blocksworld.pl').
%
%   or if the theory in different files:
%
%     ?- load_theories(['system.pl', 'policies.pl', 'domain.pl']).
%
% * Query the system:
%   e.g.

%     ?- query([T #> 0, T #< 10, holds(has(jiefei, apple), T)], Ans)
%
%  If succeeded, Ans has the form (As, Cs, Ns, Info) where "As" is the set of
%  collected abducibles, "Cs" is the set of collected inequalities, finite
%  domain constraints, and real domain constraints, "Ns" is the set of
%  collected dynamic denials and "Info" is the user collected data along the
%  computation.
%
% * To clean up the already loaded theory (so that a new one can be loaded):
%
%    ?- clear_theory.
%
% ==========================  End of Documentation ===========================

:- module(abduction, [
        % for the inequality store
        '=/='/2, inequalities/2,
        % #load_theory(+TheoryFile)
        load_theory/1,
        % #load_theories(+ListOfTheoryFiles)
        load_theories/1,
        % #clear_theory
        % for clearing the loaded theory
        clear_theory/0,
        % #query(+ListOfGoals, -SolvedState)
        % for querying the system
        query/2,
        % #eval(+ListOfGoals, -GloballyMinimalGroundHypotheses)
        eval/2,
        % #eval(+ListOfGoals, +TimeOut, -GloballyMinimalGroundHypotheses)
        eval/3,
        % #query_all(+ListOfGoals) and #eval_all(+ListOfGoals)
        % for pretty output of all the solutions
        query_all/1,
        eval_all/1,
        % eval_all(+ListOfGoals, -AllGroundQueryMinimalHypothesesPairs)
        % eval_all_with_ground_query(+ListOfGroundGoals, -AllGloballyMinimalGroundHypotheses)
        eval_all/2,
        eval_all_with_ground_query/2,
        % #enforce_labeling(+TrueOrFalse)
        % when it is set to 'true', all the finite domain variables in the state
        % will be grounded.  This can sometimes affect the performance (improved
        % or downgraded).
        enforce_labeling/1,
        % trace
        query_all_with_trace/1,
        query_all_with_trace/2,
        eval_all_with_trace/1,
        eval_all_with_trace/2,
        set_max_states/1,
        % depth bound
        set_depth_limit/1,
        clear_depth_limit/0,
        % turn on or off auto-builtin predicate recognition
        use_system_builtin/1,
    % Rares
    q_p/2,  % the former query_process
    query_prob/1, % does findall on q_p
    load_and_query_aprob/1,
    load_aprob/1, % for debug
    load_pb/1,
    q_pb/2
    % test
    %old
    %query_process/2, % query_process(Q, (Ds, NDs, Ns, Info)) - it functions pretty much the same as query/3, except that I do some post-processing on the Denial Store, i.e. I ground the abducibles and split them into denials with just one atom (which means that that atom must be false), which goes to the NDs list, and proper denials (with body length greater than 2). Also, duplicates are removed, and any ic which is a superset of another. This is mainly useful for Aprob, because the original Asystem is not really concerned with the denials obtained in Ns.
    %query_process_all/1, % same as query_all, except to query process. Again, no BDD.
    %query_exact_prob/3, % this does the BDD for the disjunction of the explanations, and I get the exact inference of the probability of the query
    %query_exact_prob_joint/3, % this does the BDD for the disjunction of the explanations, and I get the exact inference of the probability of the query
    %query_exact_prob_joint_max_nets/3,
    %query_exact_prob_nets/3, % this one also plots the networks
    %query_exact_prob_paths/3, % writes to stdout the bdd paths in the format
    %query_exact_prob_paths_file/3, % same thing, except the output is in the file aprob/out_files/bdd_paths.txt
    %query_nets/3, % this one plots the networks after doing findall on query_ss. To plot the nets, I only use the information in the Deltas (Ds).
    %query_no_ics/2, % do a query ignoring the integrity constraints
    %query_stats/3, % do a query and plot the evolution of the goals, deltas and denials in terms of size.
    %query_ss/4, % query with a custom selection strategy
    % test preds
    %write_denials2/4,
    %get_probs2/1
    % End Rares
    ]).

% ===================== Implementation ==============================

% ### Preamable ###

% need to declare the following operator for the subsequent conditional compilations
:- if(current_prolog_flag(dialect, yap)).
:- op(760, yfx, #<=>).
:- elif(current_prolog_flag(dialect, sicstus)).
:- op(760, yfx, #<==>).
:- endif.

% *** Constraint Solvers ***
:- use_module(library(clpfd)).
:- use_module(library(clpr)).

% For inequality store
:- use_module(library(atts)).
:- op(700, xfx, =/=).
:- attribute aliens/1.

% For meta-interpreter
:- use_module(library(ordsets), [
        ord_union/3,
        ord_intersection/3,
        list_to_ord_set/2
    ]).


% *** Utilities ***
:- if(current_prolog_flag(dialect, yap)).
%{ YAP
:- use_module(library(terms), [
        unifiable/3,
        variables_within_term/3
    ]).
:- use_module(library(apply_macros), [
        maplist/3,
        selectlist/3
    ]).
:- use_module(library(lists), [
        member/2,
        append/3,
        select/3,
         %Rares
        nth0/3,
        nth1/3,
        nth0/4,
        nth1/4,
        is_list/1,
        delete/3,
        exclude/3,
        append/2
    ]).

:- dynamic mysetvalue/2.
my_set_value(Key, Val) :- retractall(mysetvalue(Key, _Val)), assert(mysetvalue(Key, Val)).
my_get_value(Key, Val) :- mysetvalue(Key, Val).
inc_value(Key, NewVal, OldVal) :-
    get_value(Key, OldVal),
    NewVal is OldVal + 1,
    set_value(Key, NewVal).

%} YAP
:- elif(current_prolog_flag(dialect, sicstus)).
%{ SICStus
:- use_module(library(terms), [
        term_variables/2
    ]).
:- use_module(library(lists), [
        maplist/3,
        select/3,
    %Rares
    nth0/3,
    nth1/3,
    nth0/4,
    nth1/4,
    is_list/1,
    delete/3,
    exclude/3,
    append/2
    ]).
% Rares
:- use_module(library(sets), [
    list_to_set/2,
    union/3,
    subtract/3,
    subset/2
]).
:- use_module(library(process), [
  process_create/3,
  process_wait/2
]).
:- use_module(library(file_systems), [
  current_directory/1,
  current_directory/2
]).

% End Rares

variables_within_term(Vs, Term, OldVs) :-
    term_variables(Term, TVs),
    collect_old_variables(TVs, Vs, OldVs).
collect_old_variables([], _, []).
collect_old_variables([H|T], Vs, OldVs) :-
    collect_old_variables(T, Vs, OldVs1),
    (strictmember(Vs, H) ->
        OldVs = [H|OldVs1]
    ;
        OldVs = OldVs1
    ).

unifiable(X, Y, Eq) :-
    (var(X) ; var(Y)), !,
    (X == Y -> Eq = [] ; Eq = [X = Y]).
unifiable(X, Y, []) :-
    atomic(X), atomic(Y), !, X == Y.
unifiable(X, Y, Eqs) :-
    functor(X, F, A),
    functor(Y, F, A),
    X =.. [F|ArgsX],
    Y =.. [F|ArgsY],
    all_unifiable(ArgsX, ArgsY, Eqs).
all_unifiable([], [], []).
all_unifiable([X|TX], [Y|TY], AllEqs) :-
    unifiable(X, Y, Eqs),
    all_unifiable(TX, TY, RestEqs),
    append(Eqs, RestEqs, AllEqs).

selectlist(Pred, L1, L2) :-
    selectlist_aux(L1, Pred, L2).
selectlist_aux([], _, []).
selectlist_aux([H|T], P, L) :-
    selectlist_aux(T, P, L1),
    (\+ \+ call(P, H) ->
        L = [H|L1]
    ;
        L = L1
    ).

:- dynamic mysetvalue/2.
set_value(Key, Val) :- retractall(mysetvalue(Key, _Val)), assert(mysetvalue(Key, Val)).
get_value(Key, Val) :- mysetvalue(Key, Val).
inc_value(Key, NewVal, OldVal) :-
    get_value(Key, OldVal),
    NewVal is OldVal + 1,
    set_value(Key, NewVal).

forall(X, Y) :- \+ (X, \+ Y).
%} SICStus
:- endif.

unifiable(X, Y) :- \+ \+ (X = Y).

nonground(X) :- \+ ground(X).

strictdelete([], _, []).
strictdelete([H|T], X, R) :-
    (H == X ->
        strictdelete(T, X, R)
    ;
        strictdelete(T, X, R1),
        R = [H|R1]
    ).

%%% Rares Flags %%%
%% flag_name <tabs> domain <tabs> default <tabs> description

%%% Usual Files and File Paths
%%% (except input file - loaded with load_theory
%%%  and other asystem files)

% out_file      string          '../out/out.sat'       formula file
% probs_file    string          '../out/out.probs'     prob distribution file
% plate_file    string          '../out/out.plate'     plate file
% dict_file     string          '../out/abd.dict'      abducible dict file
% ic_file       string          '../out/ic.sat'        formula file for integrity constraints
% query_stats   string          '../out/query.stats'   query statistics
% query_profile string          '../out/query.profile' query time

%%% Logical Inference flags %%%

% prob_preds    {list}          [aprob_cat,
%                               aprob_cat_share,
%                               aprob_dirichlet,
%                               aprob_dirichlet_share,
%                               aprob_plate]
%                                               special probability predicates
% abd_plate     {off,on}        off             use plates to describe prob. models
% solve_method  {all, serial}   all             solve all queries or solve each and assemble formula
% log_inf       {exact, bound}  exact           type of inference
% ss            {o, u_old, u}   u               safe selection strategy
                                                %(o = original, u_old = unfold old, u = unfold)
% post_proc     {off,on}        on              do post processing of the ground formula
% remove_abds   {list}          []              list of abducibles to remove from solution
% no_ics        {off,on}        off             do the proof without integrity constraints
% fformat       {sat, bat, a}   sat             output the formulas in the DIMACS sat or bat format or infix aprob format
% all_abds      {off, on}       off             collect all abducibles (not only those appearing in the explanation formula)

%% Parameters for log_inf_bound

% bound_theta   R               TODO            TODO
% bound_beta    R               TODO            TODO
% bound_delta   R               TODO            TODO

%%% Probabilistic Inference Flags %%%

% prob_inf      {cond, joint}   joint           cond=P(Q|IC), joint=P(Q,IC)

%% BDD Compilation Flags
%% TODO

%%% Other Flags %%%

% Profiling/Statistics predicates

% proof_stats   {off,on}        off             gather statistics about the proof tree
% profile       {off,on}        off             profile execution time of the query

%% Debug Flags
%% Debug Flag Values - 0 off, 1 minimal, 2 verbose

% dbg_read      {0,1,2}         0               read input file related info
% dbg_query     {0,1,2}         0               query related info
% dbg_write     {0,1,2}         0               output file related info

%%% End Rares Flags %%%

%%% Rares - Set Default Flags %%%
% files
:-set_value(out_file, '../out/out.sat').

:-set_value(plate_file, '../out/out.plate').
:-set_value(dict_file, '../out/out.dbg').
:-set_value(ic_file, '../out/ic.sat').
:-set_value(query_stats, '../out/query.stats').
:-set_value(query_profile, '../out/query.profile').
% logical inference
%:-set_value(prob_preds, [aprob_cat, aprob_cat_share,
%                       aprob_dirichlet,aprob_dirichlet_share,
%                       aprob_plate]).

:-my_set_value(probs_file, '/tmp/peircebayes/aprob/out.probs').
:-my_set_value(pb_preds, [pb_dirichlet, pb_categorical, pb_plate]).

:-set_value(abd_plate, off).
:-set_value(solve_method, all).
:-set_value(log_inf, exact).
:-set_value(ss, u).
:-set_value(post_proc, on).
:-set_value(remove_abds, []).
:-set_value(no_ics, off).
:-set_value(fformat, sat).
:-set_value(all_abds, off).
% probabilistic inference
:-set_value(prob_inf, joint).
% other
:-set_value(proof_stats, off).
:-set_value(profile, off).
:-set_value(dbg_read, 0).
:-set_value(dbg_query, 0).
:-set_value(dbg_query, 0).

%%% Internal Flags
%:-set_value(pd_string(cat), 'Categorical').
%:-set_value(pd_string(dirichlet), 'Dirichlet').
:-set_value(abd_index, 0).
%%% End Rares - Set Default Flags %%%

% ----------------------------------------------------------------------------------------
% --- Preprocessing ---
use_system_builtin(true) :-
    set_value(sys_builtin, true).
use_system_builtin(false) :-
    set_value(sys_builtin, false).
:- use_system_builtin(true).

load_theory(DataFile) :-
    clear_theory,
    loadf(DataFile, Cls),
    assert_knowledge(Cls).

load_pb(DataFile) :-
    clear_theory,
    loadf(DataFile, Cls),
    preprocess_pb(Cls).

load_and_query_aprob(DataFile) :-
    clear_theory,
    loadf(DataFile, Cls),
    assert_and_query_knowledge(Cls).

load_aprob(DataFile) :-
    clear_theory,
    loadf(DataFile, Cls),
    assert_knowledge_dbg(Cls).

load_theories(DataFiles) :-
    clear_theory,
    loadfs(DataFiles, Cls),
    assert_knowledge(Cls).

clear_theory :-
    clean_up.

% Predicates used for specifying the in
:- dynamic abducible/1, builtin/1. % used by users and the system
:- dynamic control/1.   % used to interact with the runtime execution
:- dynamic enum/2, types/2. % EXP: used for specifying abducible argument types
:- dynamic abhead/2. % EXP: used for transforming rules with abducible in the head
:- dynamic rule/2, ic/1. % used by the meta-interpreter

clean_up :-
    retractall(abducible(_)),
    retractall(enum(_,_)), % EXP
    retractall(types(_, _)), % EXP
    retractall(abhead(_,_)),
    retractall(builtin(_)),
    retractall(rule(_,_)),
    retractall(ic(_)).

% [Literal Type]:
%
% In order to speed up the search, each input rule or integrity constraint
% is preprocessed when it is loaded, such that each literal "L" is wrapped
% as "(Type, L)", where "Type" is the literal type.
%
% There are four literal types:
%  a: abducible
%  b: builtin
%  c: control
%  d: defined
%  e: equality
%  i: inequality
%  n: negation
%  f: finite domain constraint
%  r: real domain constraint
%  -: failure goal (see below)
%
% There is a special literal, of the form "fail(Vs, Literals)", called the
% failure goal which can appear as a sub-goal during the abductive search.
% Logically, the failure goal is "\forall Vs . <- Literals", where "Vs" is
% the set of variables in "Literals" that are universally quantified with
% the scope the whole failure goal (denial).  All other varaibles in "Literals"
% are existentially quantified implicitly. "-" represents the type of a failure
% goal.

transform_and_wrap_literal(X, Xw) :-
    wrap_literal(X, (Type, G)),
    ((Type == a, functor(G, Ftr, Ary), abhead(Ftr, Ary)) ->
        % need to transform it
        atom_concat(Ftr, '_', NewFtr),
        G =.. [Ftr|Params],
        NewG =.. [NewFtr|Params],
        Xw = (d, NewG)
    ;
        Xw = (Type, G)
    ).

wrap_literal(X, Xw) :-
    (X = (\+ A) ->
        % for negative literal, we need to wrap the atom too
        wrap_atom(A, Aw),
        Xw = (n, \+ Aw)
    ;
        wrap_atom(X, Xw)
    ).

wrap_atom(X = Y, (e, X = Y)) :- !.
wrap_atom(X=/=Y, (i, X=/=Y)) :- !.

wrap_atom({X}, (r, {X})) :- !.

wrap_atom(X in Y, (f, X in Y)) :- !.
wrap_atom(X #= Y, (f, X #= Y)) :- !.
wrap_atom(X #< Y, (f, X #< Y)) :- !.
wrap_atom(X #=< Y, (f, X #=< Y)) :- !.
wrap_atom(X #> Y, (f, X #> Y)) :- !.
wrap_atom(X #>= Y, (f, X #>= Y)) :- !.
wrap_atom(X #\= Y, (f, X #\= Y)) :- !.
wrap_atom(#\ X, (f, #\ X)) :- !.
wrap_atom(X #/\ Y, (f, X #/\ Y)) :- !.
wrap_atom(X #\/ Y, (f, X #\/ Y)) :- !.

wrap_atom(call(X), (b, X)) :- !.
wrap_atom(X, (b, X)) :-
    get_value(sys_builtin, true),
    predicate_property(X, Prop),
    (Prop == built_in ; Prop = imported_from(_)), !.
wrap_atom(X, (b, X)) :-
    builtin(X), !.
wrap_atom(X, (c, X)) :-
    control(X), !.

wrap_atom(X, (a, X)) :-
    copy_term(X,Y),
    abducible(Y), !.

wrap_atom(X, (d, X)). % for anything else, it is assumed to be a defined.

unwrap_literal((n, \+ (_, A)), (\+ A)) :- !.
unwrap_literal((-, fail(Vs, BodyW)), fail(Vs, Body)) :-
    !, maplist(unwrap_literal, BodyW, Body).
unwrap_literal((_, A), A).

flatten_body(B, L) :-
    (B = (X, Y) ->
        flatten_body(Y, L1),
        L = [X|L1]
    ;
        L = [B]
    ).

list_to_tuple([X,Y], Tup) :-
    Tup =.. [',',X,Y].
list_to_tuple([H, _X, _Y |T], Tup) :-
    list_to_tuple([_X, _Y|T], Rest),
    Tup =.. [',', H, Rest].

transform_and_wrap_ic(Body, ic(NewBody)) :-
    flatten_body(Body, Lits),
    maplist(transform_and_wrap_literal, Lits, NewBody).
transform_and_wrap_fact(Head, rule(NewHead, [])) :-
    transform_and_wrap_literal(Head, NewHead).
transform_and_wrap_rule(Head, Body, rule(NewHead, NewBody)) :-
    flatten_body(Body, Lits),
    maplist(transform_and_wrap_literal, [Head|Lits], [NewHead|NewBody]).

unwrap_denial(fail(Vs, BodyW), fail(Vars, Body)) :-
    maplist(unwrap_literal, BodyW, Body),
    selectlist(nonground, Vs, Vars).

% /** API **/
% Read the user input file and perform preprocessing.
loadfs([], []).
loadfs([F|T], Cls):-
    loadfs(T, C2),
    loadf(F, C1),
    append(C1, C2, Cls).

loadf(DataFile, Cls) :-
    open(DataFile, read, IN),
    read_clauses(IN, Cls),
    close(IN).

assert_knowledge(Cls) :-
    assert_abducibles(Cls, Cls1),
    assert_enums(Cls1, Cls2), % EXP
    assert_types(Cls2, Cls3), % EXP
    assert_builtins(Cls3, Cls4), % change 4 to 3 to assert pr/2
    record_abducible_heads(Cls4),
    transform_and_assert_clauses(Cls4, Cls5),
    assert_prob_distributions(Cls5, []),
    assert_new_rules_for_transformed_abducibles.

assert_and_query_knowledge(Cls) :-
    assert_abducibles(Cls, Cls1),
    assert_enums(Cls1, Cls2), % EXP
    assert_types(Cls2, Cls3), % EXP
    assert_builtins(Cls3, Cls4), % change 4 to 3 to assert pr/2
    record_abducible_heads(Cls4),
    transform_and_assert_clauses(Cls4, Cls5),
    assert_prob_distributions(Cls5, Cls6),
    print(Cls6),nl,
    assert_new_rules_for_transformed_abducibles,
    assert_and_query_plates(Cls6, []).

preprocess_pb(Cls) :-
    transform_and_assert_clauses(Cls, Cls2),
    assert_pds(Cls2, Cls3),
    query_plates(Cls3, []).

assert_knowledge_dbg(Cls) :-
    assert_abducibles(Cls, Cls1),
    assert_enums(Cls1, Cls2), % EXP
    assert_types(Cls2, Cls3), % EXP
    assert_builtins(Cls3, Cls4), % change 4 to 3 to assert pr/2
    record_abducible_heads(Cls4),
    transform_and_assert_clauses(Cls4, Cls5),
    assert_prob_distributions(Cls5, _Cls6).

read_clauses(IN, Cls) :-
    catch(read(IN, Term), Err, (write(Err), fail)),
    (Term == end_of_file ->
        Cls = []
    ; Term = (:- D) ->
        call(D),
        read_clauses(IN, Cls)
    ;
        read_clauses(IN, ClsRest),
        Cls = [Term|ClsRest]
    ).

% EXP: control predicate support
:- assertz(control(current_abducibles(_))).
:- assertz(control(current_remaining_goals(_))).

assert_abducibles([], []).
assert_abducibles([abducible(X)|T], L) :-
    !,
    assertz(abducible(X)),
    assert_abducibles(T, L).
assert_abducibles([H|T], [H|L]) :-
    assert_abducibles(T, L).

assert_builtins([], []).
assert_builtins([builtin(X)|T], L) :-
    !, assertz(builtin(X)),
    assert_builtins(T, L).
assert_builtins([H|T], [H|L]) :-
    assert_builtins(T, L).

assert_enums([], []). % EXP
assert_enums([enum(X, D)|T], L) :-
    !,
    ground(X), ground(D),
    D = [_|_], % L must be a non-empty list
    assertz(enum(X, D)),
    assert_enums(T, L).
assert_enums([H|T], [H|L]) :-
    assert_enums(T, L).

% Rares
% added type, see force_all_type_conditions
assert_types([], []). % EXP
assert_types([types(X, Conds)|T], L) :-
    !,
    forall(member(C, Conds), (
    % Rares
    (
      functor(C, Ftr, 1),
      Ftr = type
    ;
          functor(C, Ftr, 2),
          (Ftr = '=' ; Ftr = type)
    )
    )),
    assertz(types(X, Conds)),
    assert_types(T, L).
assert_types([H|T], [H|L]) :-
    assert_types(T, L).
% End Rares

transform_and_assert_clauses([], []).
transform_and_assert_clauses([C|T], L2) :-
    (C = ( H :- B) ->
        (H == ic ->
            % preprocess and add integrity constraint
            transform_and_wrap_ic(B, IC),
            assert(IC),
            Arg = L2
        ; H =.. [Pred|_Args] ->
            % preprocess and add a rule
            transform_and_wrap_rule(H, B, Rule),
            assert(Rule),
            Arg = L2
        ;
            L2 = [C|Arg]
        )
    ;
        ( ( C =.. [Pred|_Args],
            my_get_value(pb_preds, PbPreds),
            member(Pred, PbPreds) )
        ->
            L2 = [C|Arg]
        ;
            transform_and_wrap_fact(C, Fact),
            assert(Fact),
            Arg = L2
        )
    ),
    transform_and_assert_clauses(T, Arg).

record_abducible_heads(Cls) :-
    findall(abhead(AbPred,Arity), (
            member(C, Cls),
            (C = (H :- _) ->
                A = H
            ;
                A = C
            ),
            abducible(A),
            functor(A, AbPred, Arity)
        ), AbHeads),
    list_to_ord_set(AbHeads, SortedAbHeads),
    forall(member(AbHead, SortedAbHeads), assertz(AbHead)).

assert_new_rules_for_transformed_abducibles :-
    forall(abhead(Ftr, Ary), (
        atom_concat(Ftr, '_', NewFtr),
        functor(H, NewFtr, Ary),
        H =.. [NewFtr|Args],
        B =.. [Ftr|Args],
        wrap_atom(H, Hw),
        wrap_atom(B, Bw),
        assert(rule(Hw, [Bw]))
    )).

% Rares

assert_pds(L1, L2) :-
    ((get_value(dbg_read, N), N>=1) ->
        print(L1),nl
    ; true
    ),
    assert_pds_r(L1, L2, ProbTemplates, ProbPredL, [], [], 1),
    my_set_value(prob_templates, ProbTemplates),
    my_set_value(prob_pred_l, ProbPredL),
    ((get_value(dbg_read, N), N>=1) ->
        my_get_value(prob_templates, ProbTemplates),
        my_get_value(prob_pred_l, ProbPredL),
        format('Prob Templates:\n~p\n',[ProbTemplates]),
        format('Prob Predicate List:\n~p\n',[ProbPredL])
    ; true
    ).

assert_pds_r([], [], ProbTemplates, ProbPredL, ProbTemplates, ProbPredL, _Idx).
assert_pds_r([ pb_dirichlet(Param, Pred, NCat, NInst) | T],
    L2, ProbTemplates, ProbPredL,
    ProbTemplatesAcc, ProbPredLAcc, Idx) :-
    (member((Pred, _), ProbPredLAcc) ->
        print('Error: Duplicate prob. predicate'),nl,
        set_value(pb_success, 0),!,fail
    ; true
    ),
    append(ProbTemplatesAcc, [(Param, NCat, NInst)], ProbTemplatesAcc1),
    append(ProbPredLAcc, [(Pred, Idx, dirichlet)], ProbPredLAcc1),
    Idx1 is Idx+NCat*NInst,
    assert_pds_r(T, L2, ProbTemplates, ProbPredL,
        ProbTemplatesAcc1,ProbPredLAcc1, Idx1).
assert_pds_r([ pb_categorical(Param, Pred, NCat, NInst) | T],
    L2, ProbTemplates, ProbPredL,
    ProbTemplatesAcc, ProbPredLAcc, Idx) :-
    (member((Pred, _), ProbPredLAcc) ->
        print('Error: Duplicate prob. predicate'),nl,
        set_value(pb_success, 0),!,fail
    ; true
    ),
    append(ProbTemplatesAcc, [(Param, NCat, NInst)], ProbTemplatesAcc1),
    append(ProbPredLAcc, [(Pred, Idx, categorical)], ProbPredLAcc1),
    Idx1 is Idx+NCat*NInst,
    assert_pds_r(T, L2, ProbTemplates, ProbPredL,
        ProbTemplatesAcc1,ProbPredLAcc1, Idx1).
assert_pds_r([H|T], [H|T2], ProbTemplates, ProbPredL,
    ProbTemplatesAcc, ProbPredLAcc, Idx) :-
    assert_pds_r(T, T2, ProbTemplates, ProbPredL,
        ProbTemplatesAcc, ProbPredLAcc, Idx).

query_plates(L1, L2) :-
    length(L1, LenL1),
    open('/tmp/peircebayes/aprob/plates/n_plates', write, Stream),
    write(Stream, LenL1),
    close(Stream),
    query_plates_r(L1, L2, 1),
    % write probs file
    write_probs_file.

write_probs_file :-
    my_get_value(prob_templates, ProbTemplates),
    my_get_value(prob_pred_l, PredL),
    my_get_value(probs_file, ProbsFile),
    open(ProbsFile, write, Stream),
    write_prob_templates_r(Stream, ProbTemplates, PredL),
    close(Stream).

write_prob_templates_r(_Stream, [], []).
% Param can be a list of >= 2 elems
write_prob_templates_r(Stream, [(Param, NCat, NInst) | T],
    [(_Pred, _Idx, Type) | T2]) :-
    Param = [_,_|_],
    write_list(Stream, Param, ' '),
    format(Stream, ' ~p ~p ~p\n', [NCat, NInst, Type]),
    write_prob_templates_r(Stream, T, T2).
% Or a number for symmetric prior
write_prob_templates_r(Stream, [(Param, NCat, NInst)|T],
    [(_Pred, _Idx, Type) | T2]) :-
    number(Param),
    format(Stream, '~p ~p ~p ~p\n', [Param, NCat, NInst, Type]),
    write_prob_templates_r(Stream, T, T2).

query_plates_r([], [], _NAcc).
query_plates_r([ pb_plate(OuterQ, Reps, InnerQ) | T], L2, NAcc) :-
    File = '/tmp/peircebayes/aprob/plates/out',
    atom_chars(File, LF),
    number_chars(NAcc, LN),
    Ext = '.plate',
    atom_chars(Ext, LE),
    append([LF,LN,LE], LF1),
    atom_chars(File1,LF1),
    open(File1, write, Stream),
    %print(OuterQ),nl,
    %print(InnerQ),nl,
    query_plate(Stream, OuterQ, Reps, InnerQ),
    close(Stream),
    % do the bdd stuff
    NAcc1 is NAcc+1,
    query_plates_r(T, L2, NAcc1).

query_plate(Stream, OuterQ, Reps, InnerQ) :-
    %set_value(bdd_created, 0),
    findall(AbdTask,
        (
            q_pb(OuterQ,_),
            findall( AbdSol,
                (
                q_pb(InnerQ, (AbdSol,_,_))
                %process_as(As, As1),
                %AbdSol = (As, []),
                %format('AbdSol:\n~p\n',[AbdSol])
                %format('Dict:\n~p\n',[AbdSol])
                ),
            AbdSolL),
            %agg_sol(AbdSolL, AggL, []),
            %format('AbdSolL:\n~p\n',[AbdSolL]),
            %format('AggL:\n~p\n',[AggL]),
            %process_agg(AbdSolL, AggL, DeltaList, BDDL, [], []),
            %format('DL\n~p\n',[DeltaList]),
            %format('BDDL:\n~p\n',[BDDL]),
            % create BDD if it doesn't exist
            %(get_value(bdd_created, 0) ->
            %    write_final_formula(BDDL),
            %    set_value(bdd_created, 1)
            %;
            %    true
            %),
            % extract indices from solutions
            %my_unzip(AbdSolL, DeltaList, DeltaNList),
            %sort(DeltaList, SortedDeltaList),
            %AbdTask = (SortedDeltaList, DeltaNList, Reps) %?
            AbdTask = (AbdSolL, Reps)
        ),
    AbdTaskL),
    %format('AbdTaskL:\n~p\n',[AbdTaskL]),
    write_pb_plate(Stream, AbdTaskL).

write_pb_plate(_Stream, []).
write_pb_plate(Stream, [(AbdSolL, Reps)|T]) :-
    ( AbdSolL=[[]] ->
        true
    ;
        write_abd_sol_l(Stream, AbdSolL, ';'),
        format(Stream, '~d\n',[Reps])
    ),
    write_pb_plate(Stream, T).

write_abd_sol_l(_Stream, [], _Op).
write_abd_sol_l(Stream, [AbdSol|T], Op) :-
    write_abd_sol(Stream, AbdSol, '.'),
    format(Stream, '~p', [Op]),
    write_abd_sol_l(Stream, T, Op).

write_abd_sol(Stream, [(Pred, PDIdx, Cat, _)], _Op) :-
    my_get_value(prob_pred_l, ProbPredL),
    nth1(PredIdx, ProbPredL, (Pred, _)),
    PredIdx1 is PredIdx-1,
    PDIdx1 is PDIdx-1,
    Cat1 is Cat-1,
    format(Stream, '~d,~d,~d', [PredIdx1, PDIdx1, Cat1]).
write_abd_sol(Stream, [(Pred, PDIdx, Cat, _),Next|T], Op) :-
    my_get_value(prob_pred_l, ProbPredL),
    nth1(PredIdx, ProbPredL, (Pred, _)),
    PredIdx1 is PredIdx-1,
    PDIdx1 is PDIdx-1,
    Cat1 is Cat-1,
    format(Stream, '~d,~d,~d~p', [PredIdx1, PDIdx1, Cat1, Op]),
    write_abd_sol(Stream, [Next|T], Op).

process_agg([], AggL, DeltaList, BDDL, DeltaList, BDDL).
process_agg([(As, [])|T], AggL, DeltaList, BDDL, DLAcc, BDDLAcc) :-
    process_delta(As, AggL, DLAcc1, BDDLNew, DLAcc, ([], [], [])),
    BDDLAcc1 = [BDDLNew|BDDLAcc],
    process_agg(T, AggL, DeltaList, BDDL, DLAcc1, BDDLAcc1).

process_delta([], _, DL, BDDL, DL, BDDL).
process_delta([(Pred,PDIdx,_,Idx)|T], AggL, DL, BDDL, DLAcc, BDDLAcc) :-
    member((Pred, PDIdx, _, IdxL), AggL),
    format('here\n',[]),
    ((IdxL = [Pref|[Idx]],length(Pref, PrefL),PrefL>0,\+member(Idx, DLAcc)) ->
        DLAcc1 = [Idx|DLAcc]
    ;
        DLAcc1 = DLAcc
    ),
    format('here ~p ~p\n',[Idx, IdxL]),
    (get_value(bdd_created, 0) ->
        append(Pref1, [Idx|_], IdxL),
        format('pref ~p\n',[Pref1]),
        format('BDDLAcc ~p\n',[BDDLAcc]),
        BDDLAcc = (DeltaL, DeltaNL, []),
        format('here\n',[]),
        DeltaL1 = [Idx|DeltaL],
        format('here\n',[]),
        append(Pref1, DeltaNL, DeltaNL1),
        format('here\n',[]),
        BDDLAcc1 = (DeltaL1, DeltaNL1, [])
    ;
        BDDLAcc1 = BDDLAcc
    ),
    format('here\n',[]),
    process_delta(T, AggL, DL, BDDL, DLAcc1, BDDLAcc1).

agg_sol([], AggL, AggL).
agg_sol([(DeltaL, [])|T], AggL, AggLAcc) :-
    agg_delta(DeltaL, AggLAcc1, AggLAcc),
    agg_sol(T, AggL, AggLAcc1).

agg_delta([], AggL, AggL).
agg_delta([(Pred, PDIdx, Cat, Idx)|T], AggL, AggLAcc) :-
    (nth0(N, AggLAcc, (Pred, PDIdx, CatL, IdxL), RestAggL) ->
        (member(Cat, CatL) ->
            AggLAcc1 = AggLAcc
        ;
            CatL1 = [Cat|CatL],
            IdxL1 = [Idx|IdxL],
            nth0(N, AggLAcc1, (Pred, PDIdx, CatL1, IdxL1), RestAggL)
        )
    ;
        AggLAcc1 = [(Pred, PDIdx, [Cat], [Idx])|AggLAcc]
    ),
    agg_delta(T, AggL, AggLAcc1).

process_as([], []).
process_as([(Pred, Cat, PDIdx, Idx)|T1], [Idx|T2]) :-
    process_as(T1, T2).

assert_prob_distributions(L1, L2) :-
    ((get_value(dbg_read, N), N>=1) ->
        print(L1),nl ; true
    ),
    set_value(pa_index, 0),
    assert_probs(L1, L2, ProbTemplates),
    set_value(prob_template, ProbTemplates),
    ((get_value(dbg_read, N), N>=1,
        get_value(prob_template, ProbTemplates)
    ) ->
        print(ProbTemplates),nl ; true
    ).

% base case
assert_probs([], [], []).

% categorical distribution
assert_probs([( aprob_cat(Params, Pred) :- Body ) |T],
            L2, [ProbTemplate | TProbTemplates]) :-
    %% checks
    Pred =.. [_F, _CatVar | _Rest],
    flatten_body(Body, [CatLit]),
    copy_term(Pred, CPred),
    copy_term(CatLit, CCatLit),
    CPred =..[_CF, CCatVar | _CRest],
    CCatLit =.. [_CFCat, CCatVar],
    cat_rules(Params, CCatLit, CPred, ProbTemplate1),
    prob_def([ProbTemplate1], [ProbTemplate]),
    assert_probs(T, L2, TProbTemplates).

% categorical distribution with indep var
assert_probs([( aprob_cat(Params, IndepVar, Pred) :- Body ) |T],
            L2, ProbTemplates) :-
    cat_probs(Params, IndepVar, Pred, Body, CurrProbTemplates1),
    prob_def(CurrProbTemplates1, CurrProbTemplates),
    append(CurrProbTemplates, TProbTemplates, ProbTemplates),
    assert_probs(T, L2, TProbTemplates).

% categorical distribution with shared var
assert_probs([( aprob_cat_share(Params, IndepVar, Pred) :- Body ) |T],
            L2, [ShareProbTemplate | TProbTemplates]) :-
    cat_probs(Params, IndepVar, Pred, Body, CurrProbTemplates),
    cat_indep_to_share(CurrProbTemplates, ShareProbTemplate1),
    prob_def([ShareProbTemplate1], [ShareProbTemplate]),
    assert_probs(T, L2, TProbTemplates).

% dirichlet distribution
assert_probs([( aprob_dirichlet(Params, Pred) :- Body ) |T],
            L2, [ProbTemplate | TProbTemplates]) :-
    %% checks
    Pred =.. [_F, _CatVar],
    flatten_body(Body, [CatLit]),
    copy_term(Pred, CPred),
    copy_term(CatLit, CCatLit),
    CPred =..[_CF, CCatVar],
    CCatLit =.. [_CFCat, CCatVar],
    dirichlet_rules(Params, CCatLit, CPred, ProbTemplate1),
    prob_def([ProbTemplate1], [ProbTemplate]),
    assert_probs(T, L2, TProbTemplates).

% dirichlet distribution with indep var
assert_probs([( aprob_dirichlet(Params, IndepVar, Pred) :- Body ) |T],
            L2, ProbTemplates) :-
    cat_probs_prepare(IndepVar, Pred, Body,
        CIndepLit, CCatLit, CPred),
    dirichlet_probs_create_ads(Params, CIndepLit, CCatLit, CPred,
        CurrProbTemplates1),
    prob_def(CurrProbTemplates1, CurrProbTemplates),
    append(CurrProbTemplates, TProbTemplates, ProbTemplates),
    assert_probs(T, L2, TProbTemplates).

% dirichlet distriubtion with shared var
assert_probs([( aprob_dirichlet_share(Params, IndepVar, Pred) :- Body ) |T],
            L2, [ShareProbTemplate | TProbTemplates]) :-
    cat_probs_prepare(IndepVar, Pred, Body,
        CIndepLit, CCatLit, CPred),
    dirichlet_probs_create_ads(Params, CIndepLit, CCatLit, CPred,
        CurrProbTemplates),
    cat_indep_to_share(CurrProbTemplates, ShareProbTemplate1),
    prob_def([ShareProbTemplate1], [ShareProbTemplate]),
    assert_probs(T, L2, TProbTemplates).

% no prob definition
assert_probs([H|T], [H|T2], ProbTemplates) :-
    assert_probs(T, T2, ProbTemplates).

prob_def([], []).
prob_def([Template|T1], [ProbDef|T2]) :-
    Template = [ProbDef|_T],
    prob_def(T1, T2).

cat_indep_to_share(CurrProbTemplates, ShareProbTemplate) :-
    CurrProbTemplates = [[ProbDef|_Rest] |_T],
    append(First, [_NInstances], ProbDef),
    length(CurrProbTemplates, LenCurrProbTemplates),
    append(First, [LenCurrProbTemplates], ShareProbDef),
    ShareProbTemplate = [ShareProbDef| T2],
    cat_share(CurrProbTemplates, T2).

cat_share([], []).
cat_share([[_Def, VarList]|T], [VarList|T2]) :-
    cat_share(T, T2).

cat_probs(Params, IndepVar, Pred, Body, CurrProbTemplates) :-
    cat_probs_prepare(IndepVar, Pred, Body,
        CIndepLit, CCatLit, CPred),
    cat_probs_create_ads(Params, CIndepLit, CCatLit, CPred,
        CurrProbTemplates).

cat_probs_prepare(IndepVar, Pred, Body,
    CIndepLit, CCatLit, CPred) :-
    %% checks
    \+ground(IndepVar),
    Pred =.. [_F|Args],
    length(Args, LenArgs),
    nth0_no_bind(NPred, Args, IndepVar),
    (NPred ==1 ->
        NPredCat is 0
    ;
        NPredCat is 1
    ),
    nth0(NPredCat, Args, CatVar),
    % CatVar is the variable that describes the domain of the distribution
    % NPredCat is the position of CatVar in Pred
    LenArgs = 2,
    flatten_body(Body, Lits),
    Lits = [B1, B2],
    ( B1 =.. [_F1|Args1],
        nth0_no_bind(NBody, Args1, IndepVar),
        IndepLit = B1,
        CatLit = B2,
        CatLit =.. [_FCat1| ArgsCat1],
        nth0_no_bind(NBodyCat, ArgsCat1, CatVar)
    ;
     B2 =.. [_F2|Args2],
        nth0_no_bind(NBody, Args2, IndepVar),
        IndepLit = B2,
        CatLit = B1,
        CatLit =.. [_FCat2| ArgsCat2],
        nth0_no_bind(NBodyCat, ArgsCat2, CatVar)
    ),
    % IndepLit is the literal that contains IndepVar
    % NBody is the position of IndepVar in IndepLit
    % CatLit is the literal containing CatVar
    % NBodyCat is the position of CatVar in CatLit
    %% generate probs
    copy_term(Pred, CPred),
    CPred =.. [_CF|CArgs],
    % CPred is a copy of Pred
    copy_term(IndepLit, CIndepLit),
    CIndepLit =.. [_CIndepF|CIndepArgs],
    nth0(NBody,  CIndepArgs, CIndepVar),
    % CIndepLit is a copy of IndepLit
    % CIndepVar is a copy of IndepVar
    nth0(NPred, CArgs, CIndepVar),
    % unifies CIndepVar in CIndepLit and CPred
    copy_term(CatLit, CCatLit),
    CCatLit =.. [_CCatF|CCatArgs],
    nth0(NPredCat, CArgs, CCatVar),
    nth0(NBodyCat, CCatArgs, CCatVar).
    % unifies CCatVar in CCatLit and CPred

cat_probs_create_ads(Params, CIndepLit, CCatLit, CPred,
    CurrProbTemplates) :-
    %% create and assert ads
    findall(ProbTemplate, ( q_p([CIndepLit]), %rule((d, CIndepLit), []),
                    cat_rules(Params, CCatLit, CPred, ProbTemplate)),
            CurrProbTemplates).

dirichlet_probs_create_ads(Params, CIndepLit, CCatLit, CPred,
    CurrProbTemplates) :-
    %% create and assert ads
    findall(ProbTemplate, (q_p([CIndepLit]), %rule((d, CIndepLit), []),
                    dirichlet_rules(Params, CCatLit, CPred, ProbTemplate)),
            CurrProbTemplates).

cat_rules(Params, CCatLit, CPred, ProbTemplate) :-
    % create prob template
    get_value(pa_index, OldPAIdx),
    length(Params, LenParams),
    PAIdx is OldPAIdx+LenParams,
    set_value(pa_index, PAIdx),
    append([cat|Params], [1], Header),
    gen_interval(OldPAIdx, PAIdx, Interval),
    maplist(add_pa, Interval, AbdList),
    append([Header], [AbdList], ProbTemplate),
    % assert abducibles
    maplist(assert_pas, AbdList, _),
    % create and assert rules
    compile_ad(CCatLit, CPred, AbdList).

dirichlet_rules(Params, CCatLit, CPred, ProbTemplate) :-
    % create prob template
    get_value(pa_index, OldPAIdx),
    length(Params, LenParams),
    PAIdx is OldPAIdx+LenParams-1,
    set_value(pa_index, PAIdx),
    append([dirichlet|Params], [1], Header),
    gen_interval(OldPAIdx, PAIdx, Interval),
    maplist(add_pa, Interval, AbdList),
    append([Header], [AbdList], ProbTemplate),
    % assert abducibles
    maplist(assert_pas, AbdList, _),
    % create and assert rules
    compile_ad(CCatLit, CPred, AbdList).

compile_ad(CCatLit, CPred, AbdList) :-
    findall(CPred,  q_p([CCatLit]), HList),%rule((d, CCatLit), []), HList),
    length(AbdList, LenAbdList),
    create_ad_rules(HList, AbdList, LenAbdList, 0).

create_ad_rules([], _AbdList, _LenAbdList, _Idx).
create_ad_rules([H|T], AbdList, LenAbdList, Idx) :-
    (LenAbdList = Idx ->
        maplist(add_neg, AbdList, BodyList)
    ;
        length(NegList, Idx),
        append(NegList, [PosPA|_RestPAs], AbdList),
        maplist(add_neg, NegList, NegBody),
        append(NegBody, [PosPA], BodyList)
    ),
    length(BodyList, LenBodyList),
    (LenBodyList = 1 ->
        BodyList = [Body]
    ;
        list_to_tuple(BodyList, Body)
    ),
    Rule = (H :- Body),
    ((get_value(dbg_read,N), N>=1) ->
        print(Rule),nl ; true
    ),
    transform_and_assert_clauses([Rule], []),
    Idx1 is Idx +1,
    create_ad_rules(T, AbdList, LenAbdList, Idx1).

assert_pas(PA, PA) :-
    assertz(abducible(PA)).

add_pa(N, Out) :-
    atom_number_concat(pa, N, Out).

nth0_no_bind(N, List, El) :-
    nth0_no_bind(N, List, El, 0).
nth0_no_bind(N, [H|T], El, Idx) :-
    (H == El ->
        N is Idx
    ;
        Idx1 is Idx+1,
        nth0_no_bind(N, T, El, Idx1)
    ).

% ground predicate Pred with respect to Vars yielding a list of partially ground predicates GPreds
ground_type(Pred, Vars, GPreds) :-
    copy_term((Pred, Vars), (CPred, CVars)),  % because the copies go in findall
    findall(CPred, force_types(CPred, CVars), GPreds).
% for the case of params
ground_type(Pred, Vars, Params, GPreds, GParams) :-
    copy_term((Pred, Vars, Params), (CPred, CVars, CParams)),  % because the copies go in findall
    findall((CPred, CParams), force_types(CPred, CVars, CParams), L),
    unpack(L, [GPreds, GParams]).

unpack(L1,L2) :-
    L1 = [H|_T],
    H =.. [','|Args],
    length(Args, N),
    gen_list([], N, Acc),
    unpack(L1,L2,Acc).

gen_list(_El, 0, []).
gen_list(El, N, [El|T]) :-
    N1 is N-1,
    gen_list(El, N1, T).

gen_list_var(_El, 0, []).
gen_list_var(El, N, [El|T]) :-
    N1 is N-1,
    gen_list(_NewEl, N1, T).

unpack([], L2, L2).
unpack([H|T], L2, Acc) :-
    H =.. [','|L],
    length(Acc, N),
    gen_list_var(_X, N, Acc1),
    append_pos(L, Acc, Acc1, 0),
    unpack(T, L2, Acc1).

append_pos([], _Acc, _Acc1, _N).
append_pos([H1|T1], Acc, Acc1,N):-
    nth0(N, Acc, L),
    nth0(N, Acc1, V),
    V = [H1|L],
    N1 is N+1,
    append_pos(T1, Acc, Acc1, N1).

split_lists([], []).
split_lists([H|T], [[H]|T2]) :-
    split_lists(T,T2).

% compile annotated disjunction into a set of rules
compile_ad([Head], [NewClause], [], NegAcc) :-
    NewClause = (Head:-Body),
    add_neg(NegAcc, BodyList),
    Body =.. [','|BodyList].
compile_ad([Head, H1|T1], [NewClause|T2], [PosPA|T3], NegAcc) :-
    (NegAcc = [] ->
        NewClause = (Head:-PosPA)
    ;
        NewClause = (Head:-Body),
        add_neg(NegAcc, NegBody),
        append(NegBody, [PosPA], BodyList),
        Body =.. [','|BodyList]
    ),
    compile_ad([H1|T1], T2, T3, [PosPA|NegAcc]).

% same as before, only the PAs have vars
compile_ad([Head], [NewClause], [], [Params], NegAcc) :-
    NewClause = (Head:-Body),
    maplist(add_vars(Params), NegAcc, NegAccVar),
    add_neg(NegAccVar, BodyList),
    Body =.. [','|BodyList].
compile_ad([Head, H1|T1], [NewClause|T2], [PosPA|T3], [Params|T4], NegAcc) :-
    (NegAcc = [] ->
        maplist(add_vars(Params), [PosPA], [PosPAVar]),
        NewClause = (Head:-PosPAVar)
    ;
        maplist(add_vars(Params), [PosPA], [PosPAVar]),
        NewClause = (Head:-Body),
        maplist(add_vars(Params), NegAcc, NegAccVar),
        add_neg(NegAccVar, NegBody),
        append(NegBody, [PosPAVar], BodyList),
        Body =.. [','|BodyList]
    ),
    compile_ad([H1|T1], T2, T3, T4,[PosPA|NegAcc]).


one_minus(X,Y) :- Y is 1-X.

% pad with zeros to get 7 length
atom_number_concat(X,Y,Z) :-
    atom_chars(X, LX),
    number_chars(Y, LY),
    length(LY, LenLY),
    (LenLY > 8 ->
        write('ERROR: Too many abducibles!'),nl
    ;
        length(PaddedLY, 8),
        append(PaddedZeros, LY, PaddedLY),
        make_list_of_zeros(PaddedZeros)
    ),
    append(LX, PaddedLY, LZ),
    atom_chars(Z, LZ).

make_list_of_zeros([]).
make_list_of_zeros(['0'|T]) :-
    make_list_of_zeros(T).

add_vars(X, Y, Z) :-
    Z=..[Y|X].

%add_neg([],[]).
%add_neg([H|T], [\+ H|T1]) :-
%    add_neg(T,T1).

add_neg(X, \+ X).

% [A,B)
gen_interval(A, B, L) :- gen_interval(A,B,L,A).

gen_interval(_A,B,[], Acc) :- Acc>=B,!.
gen_interval(A,B,[Acc|T], Acc) :-
    Acc1 is Acc+1,
    gen_interval(A,B,T,Acc1).

prod(L, Prod) :- prod(L, Prod, 1).
prod([], Prod, Prod).
prod([H|T], Prod, Acc) :-
    Acc1 is Acc*H,
    prod(T, Prod, Acc1).

% for now, there is only one plate/query, see comment below
assert_and_query_plates(L1, L2) :-
    (L1\=[], set_value(abd_plate, on) ; true),
    length(L1, LenL1),
    open('../out/plates/n_plates',write,Stream),
    write(Stream, LenL1),
    close(Stream),
    assert_plates(L1, L2, 1).

assert_plates([], [], _NAcc).
assert_plates([ aprob_plate(IteratorQuery, Reps, TemplateInitQuery, ProbQuery)
    | T],
    L2, NAcc
    ) :-
    % setup plate file
    File = '../out/plates/out',
    atom_chars(File, LF),
    number_chars(NAcc, LN),
    Ext = '.plate',
    atom_chars(Ext, LE),
    append([LF, LN, LE], LF1),
    atom_chars(File1,LF1),
    open(File1,write,Stream),
    % write to plate
    copy_term([TemplateInitQuery, ProbQuery],
        [TemplateInitQuery1, ProbQuery1]),
    nl,print(IteratorQuery),nl,
    print(ProbQuery),nl,
    write_iterator_list(Stream, IteratorQuery, Reps, ProbQuery),
    close(Stream),
    % setup bdd file
    Ext1 = '.sat',
    atom_chars(Ext1, LE2),
    append([LF, LN, LE2], LF2),
    atom_chars(File2, LF2),
    set_value(out_file, File2),
    % query
    q_p(TemplateInitQuery1, _),
    print(ProbQuery1),nl,
    query_prob(ProbQuery1),
    NAcc1 is NAcc+1,
    assert_plates(T, L2, NAcc1).
assert_plates([_H|T], L2, NAcc) :-
    assert_plates(T, L2, NAcc).

write_iterator_list(Stream, IteratorQuery, Reps, ProbQuery) :-
    get_value(prob_template, ProbTemplates),
    findall(PAList,
        (
        q_p(IteratorQuery, _),
        %print(IteratorQuery),nl,
        %halt,
        %append(IteratorQuery, ProbQuery, Query),
        findall( PAs,
            (
            q_p(ProbQuery, (Dict, _, _, _)),
            format('Dict ~s\n', [Dict])
            %pa_list_to_number(DeltaN, DeltaNNo),
            %pa_list_to_number(Delta, DeltaNo),
            %print(DeltaNNo),nl,
            %list_to_ord_set(DeltaNNo, ODeltaNNo),
            %once(check_deltan(ODeltaNNo, ProbTemplates, DeltaNNos)),
            %PAs = (DeltaNo, DeltaNNos)
            ),
            QPAs
        )
        %print(QPAs),nl,
        %my_unzip(QPAs, DeltaList, DeltaNList),
        %list_to_ord_set(DeltaList, ODeltaList),
        %list_to_ord_set(DeltaNList, ODeltaNList),
        %( (IteratorQuery = [observe(d19, [w0,w1,w3,w3,w3,w6,w7,w9,w12,w16,w16,w20,w21,w22,w23]), member(w0, [w0,w1,w3,w3,w3,w6,w7,w9,w12,w16,w16,w20,w21,w22,w23])]) ->
        %    print('okkkk'),nl,print(ODeltaList),nl ; true
        %),
        %once(check_row_delta(ODeltaList, ODeltaNList, ODeltaNList1)),
        %PAList = (ODeltaList, ODeltaNList1, Reps)
        ),
    ItList),
    !,fail,
    %length(ItList, LenIL),
    %write(LenIL),nl,
    %write(ItList),nl,
    write_plate(Stream, ItList).

my_unzip(QPAs, DList, DNList) :-
    my_unzip(QPAs, DList, DNList, [], []).

my_unzip([], DList, DNList, DList, DNList).
my_unzip([(Delta, DeltaN)|T], DList, DNList, DListAcc, DNListAcc) :-
    append(DListAcc, Delta, DListAcc1),
    append(DNListAcc, DeltaN, DNListAcc1),
    my_unzip(T, DList, DNList, DListAcc1, DNListAcc1).

check_row_delta(ODeltaList, ODeltaNList, ODeltaNList1) :-
    get_value(prob_template, ProbTemplates),
    check_deltan(ODeltaList, ProbTemplates, VarsToRemove),
    subtract(ODeltaNList, VarsToRemove, ODeltaNList1).

check_deltan(DeltaN, ProbTemplates, DeltaNNos) :-
    check_deltan(DeltaN, ProbTemplates, DeltaNNos, []).

check_deltan([], _ProbTemplates, DNNos, DNNos).
check_deltan([N|T], ProbTemplates, DNNos, DNNosAcc) :-
    get_consec_interval(N, T, N2),!,
    check_interval(N, N2, ProbTemplates, T, TRest, N2s, []),
    append(DNNosAcc, N2s, DNNosAcc1),
    check_deltan(TRest, ProbTemplates, DNNos, DNNosAcc1).
check_deltan([_H|T], ProbTemplates, DNNos, DNNosAcc) :-
    check_deltan(T, ProbTemplates, DNNos, DNNosAcc).

get_consec_interval(N, L, N2) :- get_consec_interval(N, L, N2, N).
get_consec_interval(_N, [], NAcc, NAcc).
get_consec_interval(_N, [H|_T], N2, N2) :-
    H =\= N2+1.
get_consec_interval(N, [H|T], N2, _NAcc) :-
    get_consec_interval(N, T, N2, H).

check_interval(N, N2, ProbTemplates, T, TRest, N2s, N2sAcc) :-
    aprob_is_row(N,NX, ProbTemplates),
    NX =< N2,
    !,
    Rest is NX+1,
    append(_Pref,[NX|TRest1], T),
    (Rest =< N2 ->
        check_interval(Rest, N2, ProbTemplates, T, TRest1, N2s, [NX|N2sAcc])
    ;
        N2s=[NX|N2sAcc],
        TRest = TRest1
    ).
check_interval(_N, _N2, _ProbTemplates, T, TRest, N2s, N2s) :-
    ground(T),
    ground(TRest).
check_interval(_N, _N2, _ProbTemplates, T, T, N2s, N2s).

aprob_is_row(N, NX, ProbTemplates) :- aprob_is_row(N, NX, ProbTemplates, 0).
aprob_is_row(N, NX, [ProbTemplate | T], Idx) :-
    append(_, [NoInstances], ProbTemplate),
    length(ProbTemplate,  LenPT),
    NoVars is LenPT-3,
    MaxVal is Idx+NoVars*NoInstances,
    DiffN is N-Idx,
    ( (DiffN>=0, N < MaxVal, DiffN mod NoVars =:= 0 ) ->
        NX is N+NoVars-1
    ;
        NextIdx is MaxVal,
        aprob_is_row(N, NX, T, NextIdx)
    ).

write_plate(_Stream, []).
write_plate(Stream, [(DList, DNList, Reps)|T]) :-
    ( (DList=[], DNList=[]) ->
        true
    ;
        write_dlist(Stream, DList, DNList, ' '),
        format(Stream, ' ~d\n',[Reps])
    ),
    write_plate(Stream, T).

write_dlist(_Stream, [], [], _Op).
write_dlist(Stream, DList, [], Op) :-
    write_list(Stream, DList, Op).
write_dlist(Stream, [], DNList, Op) :-
    maplist(add_plus_to_number, DNList, PlusDNList),
    write_list(Stream, PlusDNList, Op).
write_dlist(Stream, [N1|T1], [N2|T2], Op) :-
    (N1 < N2 ->
        format(Stream, '~d ', [N1]),
        write_dlist(Stream, T1, [N2|T2], Op)
    ;
        add_plus_to_number(N2, N2Plus),
        format(Stream, '~s ', [N2Plus]),
        write_dlist(Stream, [N1|T1], T2, Op)
    ).

add_plus_to_number(N, Atom) :-
    number_codes(N, NL),
    atom_codes('+', PlusL),
    append(NL, PlusL, LongL),
    atom_codes(Atom, LongL).

pa_list_to_number(PAList, NumberList) :-
    maplist(pa_to_number, PAList, NumberList).

pa_to_number(PA, Number) :-
    atom_concat('pa', NumberString, PA),
    atom_chars(NumberString, NumberList),
    number_chars(Number, NumberList).

assert_plate_list(_Stream, []).
assert_plate_list(Stream, [[DistribNumber,Elem] |T]) :-
    % for now Elem is either "row" or "col"
    % and is written as such in plate_file
    format(Stream, '~d ~s\n', [DistribNumber, Elem]),
    assert_plate_list(Stream, T).
% End Rares

% --- Abductive Meta-Interpreter (Depth-First) ---

% Ans: (As, Cs, Ns)
query(Query, Ans) :-
    initialise(Query, Gs, As, Cs, Ns, Info),
    solve_all(Gs, As, Cs, Ns, Info, Ans).

q_p(Query) :- q_p(Query, _Ans).
q_p(Query, Ans) :-
    (get_value(no_ics, off) ->
        initialise(Query, Gs, As, Cs, Ns, Info)
    ; get_value(no_ics, on) ->
        initialise_no_ics(Query, Gs, As, Cs, Ns, Info)
    ),
    solve_all(Gs, As, Cs, Ns, Info, AnsPre),
    (get_value(post_proc, on) ->
        post_process(AnsPre, Deltas, NDeltas, NsOut, NewInfo),
        Ans = (Deltas, NDeltas, NsOut, NewInfo)
    ; get_value(post_proc, off) ->
        Ans = AnsPre
    ).

q_pb(Query) :- q_pb(Query, _Ans).
q_pb(Query, Ans) :-
    (get_value(no_ics, off) ->
        initialise(Query, Gs, As, Cs, Ns, Info)
    ; get_value(no_ics, on) ->
        initialise_no_ics(Query, Gs, As, Cs, Ns, Info)
    ),
    solve_all(Gs, As, Cs, Ns, Info, AnsPre),
    (get_value(post_proc, on) ->
        post_process(AnsPre, Deltas, NDeltas, NsOut, NewInfo),
        Ans = (Deltas, NDeltas, NsOut, NewInfo)
    ; get_value(post_proc, off) ->
        Ans = AnsPre
    ).


query_prob(Query) :-
    %current_directory(Dir, Dir),
    %format('~s \n', [Dir]),
    (get_value(log_inf, exact) ->
        (get_value(solve_method, all) ->
            (get_value(prob_inf, joint) ->
                findall((As, NAs, Ns), q_p(Query, (As, NAs, Ns,_)), L1),
                %write(L1),nl,
                write_final_formula(L1)
            ; get_value(prob_inf, cond) ->
               true
            )
        ; get_value(solve_method, serial) ->
            true % TODO serial
        )
    ;
        true % TODO approx inf
    ).

write_final_formula(L1) :-
    ((get_value(dbg_write,V0), V0>0) ->
        write('q and write'),nl,
        write(L1),nl ; true
    ),
    %write(L1),nl,
    (get_value(remove_abds, []) ->
        L2=L1
    ;
        remove_temp_abds(L1, L2)
    ),
    %write(L2),nl,
    ((get_value(dbg_write, V1), V1>0) ->
        write('L2'),nl,
        write(L2),nl ; true
    ),
    remove_neg(L2, L2NoNeg),
    ((get_value(dbg_write, V1), V1>0) ->
        write(L2NoNeg),nl ; true
    ),
    order_abducibles(L2NoNeg, PreAbdOrder),
    list_to_ord_set(PreAbdOrder, AbdOrder),
    ((get_value(dbg_write,V2), V2>0) ->
        write('AbdOrder'),nl,
        write(AbdOrder),nl ; true
    ),
    (get_value(abd_plate, off) ->
        get_value(prob_template, ProbTemplates),
        write_probs_file(AbdOrder, ProbTemplates),
        write_abducible_dict(AbdOrder)
    ;
        get_value(prob_template, ProbTemplates),
        write_probs_file_plate(ProbTemplates)
    ),
    get_value(out_file, OutFile),
    open(OutFile, write, Stream),
    length(AbdOrder, LenAbdOrder),
    (get_value(fformat, sat) ->
        [And, Or, Not] = ['*', '+', '-'],
        format(Stream, 'p sat ~d\n', [LenAbdOrder]),
        write_formula(Stream, L2, AbdOrder, And, Or, Not)
    ; get_value(fformat, bat) ->
        [And, Or, Not] = ['and', 'or', 'not'],
        write_probs_bat(Stream, ProbL),
        write(Stream, ':exists\n'),
        format(Stream, '(:vars (v ~d) )\n', [LenProbL]),
        write(Stream, '()\n'),
        write_formula(Stream, L2, AbdOrder, And, Or, Not)
    ; get_value(fformat, a) ->
        [And, Or, Not] = ['&', '|', '~'],
        write_probs_sat(Stream, ProbL),
        format(Stream, 'p sat ~d\n', [LenProbL]),
        write_formula(Stream, L2, AbdOrder, And, Or, Not)
    ),
    close(Stream).

remove_temp_abds([], []).
remove_temp_abds([(As, NAs, Ns)|T], [(NewAs, NewNAs, NewNs)|T2]) :-
    exclude(temp_abd, As, NewAs),
    exclude(temp_abd, NAs, NewNAs),
    exclude(member_temp_abd,  Ns, NewNs),
    remove_temp_abds(T, T2).

temp_abd(At) :-
    At =.. [Abd|T],
    length(T, Ar),
    get_value(remove_abds, AbdL),
    member(Abd/Ar, AbdL).
member_temp_abd(X) :-
    is_list(X),
    get_value(remove_abds, AbdL),
    member(Abd/Ar, AbdL),
    member(Lit, X),
    ( Lit = \+(At) ->
        At =.. [Abd|T],
        length(T, Ar)
    ;
        Lit =.. [Abd|T],
        length(T, Ar)
    ).

write_abducible_dict(AbdOrder) :-
    get_value(dict_file, OutFile),
    open(OutFile, write, Stream),
    write_abducible_dict(Stream, AbdOrder, 1),
    close(Stream).

write_abducible_dict(_Stream, [], _N).
write_abducible_dict(Stream, [Abd|T], N) :-
    N1 is N+1,
    format(Stream, '~d ~w\n', [N, Abd]),
    write_abducible_dict(Stream, T, N1).

write_probs_file(AbdOrder, ProbTemplates) :-
    get_value(probs_file, ProbsFile),
    open(ProbsFile, write, Stream),
    length(AbdOrder, LenAbdOrder),
    set_value(abd_index, LenAbdOrder),
    write_probs_file(Stream, ProbTemplates, AbdOrder),
    close(Stream).

write_probs_file_plate(ProbTemplates) :-
    get_value(probs_file, ProbsFile),
    open(ProbsFile, write, Stream),
    write_probs_file_plate(Stream, ProbTemplates),
    close(Stream).

write_probs_file(_Strem, [], _AbdOrder).
write_probs_file(Stream, [ProbTemplate|T], AbdOrder) :-
    ProbTemplate = [_ProbDef | Instances],
    my_flatten(Instances, Vars),
    ( ( list_to_ord_set(Vars, SVars),
        list_to_ord_set(AbdOrder, SAbdOrder),
        ord_intersection(SVars, SAbdOrder, [])
        )  ->
        true ; write_template(Stream, ProbTemplate, AbdOrder)
    ),
    write_probs_file(Stream, T, AbdOrder).

write_probs_file_plate(_Strem, []).
write_probs_file_plate(Stream, [ProbTemplate|T]) :-
    write_template_plate(Stream, ProbTemplate),
    write_probs_file_plate(Stream, T).

write_template(Stream, [ProbDef|Instances], AbdOrder) :-
    ProbDef = [ProbDistribType|ParamsNInstances],
    append(Params, [NInstances], ParamsNInstances),
    get_value(pd_string(ProbDistribType), ProbDistribString),
    format(Stream, '~s ', [ProbDistribString]),
    write_list(Stream, Params, ' '),
    format(Stream, ' ~d\n', [NInstances]),
    write_template_instances(Stream, Instances, AbdOrder).

write_template_plate(Stream, ProbDef) :-
    ProbDef = [ProbDistribType|ParamsNInstances],
    append(Params, [NInstances], ParamsNInstances),
    get_value(pd_string(ProbDistribType), ProbDistribString),
    format(Stream, '~s ', [ProbDistribString]),
    write_list(Stream, Params, ' '),
    format(Stream, ' ~d\n', [NInstances]).

write_template_instances(_Stream, [], _AbdOrder).
write_template_instances(Stream, [Instance | T], AbdOrder) :-
    write_template_instance(Stream, Instance, AbdOrder),
    write_template_instances(Stream, T, AbdOrder).

write_template_instance(Stream, [Var], AbdOrder) :-
    get_var_number(Var, AbdOrder, N),
    format(Stream, '~d\n', [N]).
write_template_instance(Stream, [Var, H|T], AbdOrder) :-
    get_var_number(Var, AbdOrder, N),
    format(Stream, '~d ', [N]),
    write_template_instance(Stream, [H|T], AbdOrder).

get_var_number(Var, AbdOrder, N) :-
    (
        nth1(N, AbdOrder, Var)
    ;
        get_value(abd_index, AbdIdx),
        N is AbdIdx+1,
        set_value(abd_index, N)
    ).

% only works when all elements of NestedL are (un-nested) lists
my_flatten(NestedL, L) :- my_flatten(NestedL, L, []).
my_flatten([], L, L).
my_flatten([HL|T], L, Acc) :-
    append(Acc, HL, Acc1),
    my_flatten(T, L, Acc1).


write_probs_sat(Stream, ProbL) :-
    write(Stream, 'c probs '),
    write_probs_list(Stream, ProbL).
write_probs_bat(Stream, ProbL) :-
    write(Stream, '; probs '),
    write_probs_list(Stream, ProbL).

write_probs_list2(_Stream, [], _N).
write_probs_list2(Stream, [(_Abd, Prob)|T], N) :-
    format(Stream, 'Categorical ~h 1\n~d\n', [Prob, N]),
    N1 is N+1,
    write_probs_list2(Stream, T, N1).

write_probs_list(Stream, [(_Abd, Prob)]) :-
    !,
    write(Stream, Prob),
    write(Stream, '\n').
write_probs_list(Stream, [(_Abd, Prob), H|T]) :-
    write(Stream, Prob),
    write(Stream, ' '),
    write_probs_list(Stream, [H|T]).

write_formula(Stream, L2, AbdOrder, And, Or, Not) :-
    length(L2, LenL2),
    (LenL2 > 1 ->
        (get_value(fformat, sat) ->
            format(Stream, '~s(\n', [Or]),
            write_form(Stream, L2, '\n', AbdOrder, And, Not),
            write(Stream, ')\n')
        ; get_value(fformat, bat) ->
            format(Stream, '(~s \n', [Or]),
            write_form(Stream, L2, '\n', AbdOrder, And, Not),
            write(Stream, ')\n')
        ; get_value(fformat, a) ->
            atom_codes(Or, OrStr),
            append(OrStr, "\n", OrNewLineStr),
            atom_codes(Sep, OrNewLineStr),
            write_form(Stream, L2, Sep, AbdOrder, And, Not)
        )
    ;
        write_form(Stream, L2, ' ', AbdOrder, And, Not)
    ).

write_form(_Stream, [], _Sep, _AbdOrder, _And, _Not).
write_form(Stream, [(As, NAs, Ns)|T], Sep, AbdOrder, And, Not) :-
    exclude(is_empty_list, [As, NAs, Ns], NonEmptyL),
    length(NonEmptyL, LenNonEmptyL),
    (LenNonEmptyL > 1 ->
        (get_value(fformat, sat) ->
            format(Stream, '~s( \n', [And])
        ; get_value(fformat, bat) ->
            format(Stream, '(~s \n', [And])
        ; get_value(fformat, a) ->
            true
        )
    ;
        true
    ),
    (As = [] ->
        true
    ;
        length(As, LenAs),
        (LenAs > 1 ->
            (get_value(fformat, sat) ->
                format(Stream, '~s( \n', [And]),
                write_lit_list(Stream, As, ' ', AbdOrder, Not),
                write(Stream, ' )\n')
            ; get_value(fformat, bat) ->
                format(Stream, '(~s \n', [And]),
                write_lit_list(Stream, As, ' ', AbdOrder, Not),
                write(Stream, ' )\n')
            ; get_value(fformat, a) ->
                write_lit_list(Stream, As, And, AbdOrder, Not),
                write(Stream, '\n')
            )
        ;
            write_lit_list(Stream, As, ' ', AbdOrder, Not),
            write(Stream, '\n')
        )
    ),
    (NAs = [] ->
        true
    ;
        length(NAs, LenNAs),
        (LenNAs > 1 ->
            (get_value(fformat, sat) ->
                format(Stream, '~s( \n', [And]),
                write_neg_lit_list(Stream, NAs, ' ', AbdOrder, Not),
                write(Stream, ' )\n')
            ; get_value(fformat, bat) ->
                format(Stream, '(~s \n', [And]),
                write_neg_lit_list(Stream, NAs, ' ', AbdOrder, Not),
                write(Stream, ' )\n')
            ; get_value(fformat, a) ->
                write_neg_lit_list(Stream, NAs, And, AbdOrder, Not),
                write(Stream, '\n')
            )
        ;
            write_neg_lit_list(Stream, NAs, ' ', AbdOrder, Not),
            write(Stream, '\n')
        )
    ),
    (Ns = [] ->
        true
    ;
        length(Ns, LenNs),
        (LenNs > 1 ->
            (get_value(fformat, sat) ->
                format(Stream, '~s( \n', [And]),
                write_ns_list(Stream, Ns, ' ', AbdOrder, And, Not),
                write(Stream, ' )\n')
            ; get_value(fformat, bat) ->
                format(Stream, '(~s \n', [And]),
                write_ns_list(Stream, Ns, ' ', AbdOrder, And, Not),
                write(Stream, ' )\n')
            ; get_value(fformat, a) ->
                write_ns_list(Stream, Ns, '\n', AbdOrder, And, Not),
                write(Stream, '\n')
            )
        ;
            write_ns_list(Stream, Ns, ' ', AbdOrder, And, Not),
            write(Stream, '\n')
        )
    ),
    (LenNonEmptyL > 1 ->
        (get_value(fformat, sat) ->
            write(Stream, ')\n')
        ; get_value(fformat, bat) ->
            write(Stream, ')\n')
        ; get_value(fformat, a) ->
            true
        )
    ;
        true
    ),
    (T=[] ->
        true
    ;
        write(Stream, Sep)
    ),
    write_form(Stream, T, Sep, AbdOrder, And, Not).

is_empty_list([]).

write_lit_list(_Stream, [], _Sep, _AbdOrder, _Not).
write_lit_list(Stream, [Lit], _Sep, AbdOrder, Not) :-
    write_lit(Stream, Lit, AbdOrder, Not),!.
write_lit_list(Stream, [H|T], Sep, AbdOrder, Not) :-
    write_lit(Stream, H, AbdOrder, Not),
    write(Stream, Sep),
    write_lit_list(Stream, T, Sep, AbdOrder, Not).

write_lit(Stream, L, AbdOrder, Not) :-
    L =.. ['\\+',Atom],
    !,
    (get_value(fformat, sat) ->
        format(Stream, '~s', [Not]),
        write_at(Stream, Atom, AbdOrder)
    ; get_value(fformat, bat) ->
        format(Stream, '(~s ', [Not]),
        write_at(Stream, Atom, AbdOrder),
        write(Stream, ')')
    ; get_value(fformat, a) ->
        write(Stream, Not),
        write_at(Stream, Atom, AbdOrder)
    ).
write_lit(Stream, L, AbdOrder, _Not) :-
    write_at(Stream, L, AbdOrder).

write_neg_lit(Stream, L, AbdOrder, _Not) :-
    L =.. ['\\+',Atom],
    !,
    write_at(Stream, Atom, AbdOrder).
write_neg_lit(Stream, L, AbdOrder, Not) :-
    (get_value(fformat, sat) ->
        format(Stream, '~s', [Not]),
        write_at(Stream, L, AbdOrder)
    ; get_value(fformat, bat) ->
        format(Stream, '(~s ', [Not]),
        write_at(Stream, L, AbdOrder),
        write(Stream, ')')
    ; get_value(fformat, a) ->
        write(Stream, Not),
        write_at(Stream, L, AbdOrder)
    ).

write_at(Stream, Atom, AbdOrder) :-
    (get_value(fformat, sat) ->
        nth1(N, AbdOrder, Atom),
        format(Stream,'~d', [N])
    ; get_value(fformat, bat) ->
        nth0(N, AbdOrder, Atom),
        format(Stream, '(v ~d)', [N])
    ; get_value(fformat, a) ->
        nth0(N, AbdOrder, Atom),
        format(Stream, 'v~d', [N])
    ).

write_neg_lit_list(_Stream, [], _Sep, _AbdOrder, _Not).
write_neg_lit_list(Stream, [Lit], _Sep, AbdOrder, Not) :-
    write_neg_lit(Stream, Lit, AbdOrder, Not),!.
write_neg_lit_list(Stream, [H|T], Sep, AbdOrder, Not) :-
    write_neg_lit(Stream, H, AbdOrder, Not),
    write(Stream, Sep),
    write_neg_lit_list(Stream, T, Sep, AbdOrder, Not).

write_ns_list(_Stream, [], _Sep, _AbdOrder, _And, _Not).
write_ns_list(Stream, [Denial], _Sep, AbdOrder, And, Not) :-
    (get_value(fformat, sat) ->
        format(Stream, '~s( ~s( ', [Not, And]),
        write_lit_list(Stream, Denial, ' ', AbdOrder, Not),
        write(Stream, ' ) )')
    ; get_value(fformat, bat) ->
        format(Stream, '(~s (~s ', [Not, And]),
        write_lit_list(Stream, Denial, ' ', AbdOrder, Not),
        write(Stream, ' ) )')
    ; get_value(fformat, a) ->
        format(Stream ,'~s(', [Not]),
        write_lit_list(Stream, Denial, And, AbdOrder, Not),
        write(Stream, ')')
    ),
    !.
write_ns_list(Stream, [H|T], Sep, AbdOrder, And, Not) :-
    (get_value(fformat, sat) ->
        format(Stream, '~s( ~s( ', [Not, And]),
        write_lit_list(Stream, H, ' ', AbdOrder, Not),
        write(Stream, ' ) )')
    ; get_value(fformat, bat) ->
        format(Stream, '(~s (~s ', [Not, And]),
        write_lit_list(Stream, H, ' ', AbdOrder, Not),
        write(Stream, ' ) )')
    ; get_value(fformat, a) ->
        format(Stream ,'~s(', [Not]),
        write_lit_list(Stream, H, And, AbdOrder, Not),
        write(Stream, ')')
    ),
    write(Stream, Sep),
    write_ns_list(Stream, T, Sep, AbdOrder, And, Not).

%query_no_ics(Query, Ans) :-
%    initialise_no_ics(Query, Gs, As, Cs, Ns, Info),
%    solve_all(Gs, As, Cs, Ns, Info, Ans).

%query_process(Query, (Deltas, NDeltas, NsOut, NewInfo)) :-
  % - post-process the denial store, i.e.
  %   > all atoms in delta (As) are removed from the ics
  %   > ics which contain another ic as a subset are removed
  %   > ics like <- a. are collected in a special list
%    initialise(Query, Gs, As, Cs, Ns, Info),
%    solve_all(Gs, As, Cs, Ns, Info, Ans),
%    post_process(Ans, Deltas, NDeltas, NsOut, NewInfo).

%query_process_no_init(Gs, As, Cs, Ns, Info, (Deltas, NDeltas, NsOut, NewInfo)) :-
  % - post-process the denial store, i.e.
  %   > all atoms in delta (As) are removed from the ics
  %   > ics which contain another ic as a subset are removed
  %   > ics like <- a. are collected in a special list
%    solve_all(Gs, As, Cs, Ns, Info, Ans),
%    post_process(Ans, Deltas, NDeltas, NsOut, NewInfo).

post_process(Ans, Deltas, NDeltas, NsOut, NewInfo) :-
  % Rares
  Ans = (Deltas, _CsFinal, Ns2, Info1),
  remove_delta_denials(Ns2, Ns3, Deltas),
  (get_value(dbg_query,2) ->
    write('q_process'),nl,
    write('Ns3'),nl,
    write(Ns3),nl ; true
  ),
  remove_subset_denials(Ns3, Ns4),
  (get_value(dbg_query,2) ->
    write('Ns4'),nl,
    write(Ns4),nl ; true
  ),
  split_false_atoms(Ns4, NDeltas, NsOut),
  (get_value(dbg_query,2) ->
    write('Deltas'),nl,
    write(Deltas),nl,
    write('NsOut'),nl,
    write(NsOut),nl ; true
  ),
  inspect_post_process(Info1, Deltas, NDeltas, NsOut, NewInfo).


% TODO remove duplicate solutions (in terms of delta + denial)
%query_process_all(Query) :-
%  findall((D,N),query_process(Query, (D,_C,N,_Info)), L),
%  write('*** Query Process All ***\nsolution format = Delta, Denial Store\n'),
%  write_list(user_output, L, '\n').

% solve_all(Gs, As, Cs, Ns, Info, Solution)
%   Gs: set of pending goals
%   As: set of collected abducibles
%   Cs: set of constraints, of the form (Es, Fs, Rs), where
%     Es: set of inequalities
%     Fs: set of finite domain constraints
%     Rs: set of real domain constraints
%   Info: any user-defined data structure for storing
%     computational information, such as debug data
%   Solution: when the whole abductive computation succeeds,
%     this will contain the final answer

% base case: no more pending goals.  so it succeeds.
solve_all([], As, (Es, Fs, Rs), NsW, Info, (As, Cs, Ns, NewInfo)) :-
  % Rares
  % post process, i.e.
  % - remove non-abducibles from the denial store Ns
  % - ground remaining abducibles in the denial store
  (get_value(dbg_query,2) ->
    write('post-process'),nl,
    write(NsW),nl,
    write(As),nl ; true
  ),
  remove_non_abducibles(NsW, NsW1),
  (get_value(dbg_query,2) ->
    write('NsW1'),nl,
    write(NsW1),nl ; true
  ),
  ground_abducibles(NsW1, NsW2),
  (get_value(dbg_query,2) ->
    write('NsW2'),nl,
    write(NsW2),nl ; true
  ),
  % End Rares
    extract_constraints((Es, Fs, Rs), Cs),
    maplist(unwrap_denial, NsW2, Ns),
    inspect_successful_state(As, Cs, Ns, Info, NewInfo).

% recursive case: simple depth-first left-right search strategy
solve_all([(Type, G)|RestGs], As, Cs, Ns, Info, Sol) :-
    ((get_value(dbg_query, X), X>0) ->
        write('Goal '), write(Type), write(' '), write(G),nl,
        write('RestGs '), write(RestGs),nl ; true
    ),
    ( (get_value(log_inf, bound),get_value(bound_theta, _Theta)) ->
        % TODO
        true
        % evaluate P(S)
        %Ans = (As, Cs, Ns, Info),
        %post_process(Ans, Deltas, NDeltas, NsOut, NewInfo),
        %L1 = [(Deltas, NDeltas, NsOut)],
        %remove_temp_r(L1, L2), % remove temp_r abducibles from solutions
        %write('L2'),nl,
        %write_L2(L2),
        %write('End L2'),nl,
        %((get_value(dbg_write, V1), V1>0) ->
        %    write(L2NoNeg),nl ; true
        %),
        %remove_neg(L2, L2NoNeg),
        %((get_value(dbg_write, V1), V1>0) ->
        %    write(L2NoNeg),nl ; true
        %),
        %order_abducibles(L2NoNeg, AbdOrder),
        %((get_value(dbg_write,V2), V2>0) ->
        %    write('AbdOrder'),nl,
        %    write(AbdOrder),nl ; true
        %),
        %get_probs(AbdOrder, ProbL),
        %write_to_file(L2, ProbL, F),
        %((get_value(dbg_write,V4), V4>0) ->
        %    write('write_ok'),nl ; true
        %),
        %call_bdd_script,
        % process bdd output
        %call_py_bdd_proc_script,
        % read from file
        %read_from_bdd(Paths, QueryProb, '../temp_files/aprob_bdd_proc.txt'),
        %QueryProb = prob(P),
        % decide to continue or stop
        %(P>=theta ->
        %    solve_one(Type, G, RestGs, As, Cs, Ns, Info, Sol)
        %;
        %    Sol = ([(Type, G)|RestGs], As, Cs, Ns, Info)
        %)
    ; get_value(log_inf, exact) ->
        solve_one(Type, G, RestGs, As, Cs, Ns, Info, Sol)
    ).

solve_one(a, A, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((a, A), RestGs, As, Cs, Ns, Info, NewInfo),
    (
        % reuse hypotheses
        member(A, As),
        solve_all(RestGs, As, Cs, Ns, NewInfo, Sol)
    ;   % a create new hypothesis
        force_types(A), % EXP
        resolve_abducible_with_delta(As, A, Inequalities),
        propagate_inequalities(Inequalities, Cs, NewCs),
        resolve_abducible_with_dynamic_ics(Ns, A, FailureGoals),
        append(FailureGoals, RestGs, NewGs),
        solve_all(NewGs, [A|As], NewCs, Ns, NewInfo, Sol)
    ).

% DEFINED stuff

% special pb predicate
solve_one(d, D, RestGs, As, Cs, Ns, Info, Sol) :-
    D =.. [pb_call, Term],
    inspect_solve_one((d, D), RestGs, As, Cs, Ns, Info, NewInfo),
    NewGs = [(d,Term) | RestGs],
    solve_all(NewGs, As, Cs, Ns, NewInfo, Sol).

% probabilistic abducible
solve_one(d, D, RestGs, As, Cs, Ns, Info, Sol) :-
    D =.. [Pred, Cat, PDIdx],
    my_get_value(prob_pred_l, ProbPredL),
    nth0(N, ProbPredL, (Pred, PredIdx, _Type)),
    inspect_solve_one((d, D), RestGs, As, Cs, Ns, Info, NewInfo),
    % N is the position in ProbTemplates of Pred info
    % PredIdx is the start index for Pred
    %format('D prob: ~p\n', [D]),
    % TODO ground checking of the args
    % TODO encode As as a hash
    (
        % case 1: the same category was chosen before
        % NB: if a different category was chosen this fails
        member((Pred, PDIdx, Cat, _), As),
        NewAs = As
    ;
        % case 2: a choice from the pd was not made
        \+member((Pred, PDIdx, Cat, _), As),
        % extract NCat the number of categories in the pd
        my_get_value(prob_templates, ProbTemplates),
        nth0(N, ProbTemplates, (_, NCat, _NInst)),
        % compute index Idx of the choice
        Idx is PredIdx+Cat+NCat*(PDIdx-1)-1,
        NewAs = [(Pred, PDIdx, Cat, Idx)|As]
    ),
    solve_all(RestGs, NewAs, Cs, Ns, NewInfo, Sol).

% other defined predicates
solve_one(d, D, RestGs, As, Cs, Ns, Info, Sol) :-
    my_get_value(prob_pred_l, ProbPredL),
    D =.. [Pred | _],
    \+member((Pred, _), ProbPredL),
    %format('D normal: ~p\n', [D]),
    % original
    inspect_solve_one((d, D), RestGs, As, Cs, Ns, Info, NewInfo),
    rule((d,D), B), % pick a rule
    append(B, RestGs, NewGs), % FIXME: solve constraints first?
    solve_all(NewGs, As, Cs, Ns, NewInfo, Sol).


% END DEFINED stuff

solve_one(b, B, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((b, B), RestGs, As, Cs, Ns, Info, NewInfo),
    call(B), % backtrackable
    solve_all(RestGs, As, Cs, Ns, NewInfo, Sol).

solve_one(c, C, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((c, C), RestGs, As, Cs, Ns, Info, NewInfo),
    (C = current_abducibles(X) ->
        X = As
    ; C = current_remaining_goals(X) ->
        X = RestGs
    ;
        call(C) % FIXME
    ),
    solve_all(RestGs, As, Cs, Ns, NewInfo, Sol).

solve_one(e, X = Y, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((e, X = Y), RestGs, As, Cs, Ns, Info, NewInfo),
    call(X = Y),
    solve_all(RestGs, As, Cs, Ns, NewInfo, Sol).

solve_one(i, X=/=Y, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((i, X=/=Y), RestGs, As, Cs, Ns, Info, NewInfo),
    propagate_inequalities([X=/=Y], Cs, NewCs),
    solve_all(RestGs, As, NewCs, Ns, NewInfo, Sol).

solve_one(n, \+ G, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((n, \+ G), RestGs, As, Cs, Ns, Info, NewInfo),
    solve_one(-, fail([], [G]), RestGs, As, Cs, Ns, NewInfo, Sol).

solve_one(f, FC, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((f, FC), RestGs, As, Cs, Ns, Info, NewInfo),
    propagate_finite_domain_constraints([FC], Cs, NewCs),
    solve_all(RestGs, As, NewCs, Ns, NewInfo, Sol).

solve_one(r, RC, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_solve_one((r, RC), RestGs, As, Cs, Ns, Info, NewInfo),
    propagate_real_domain_constraints([RC], Cs, NewCs),
    solve_all(RestGs, As, NewCs, Ns, NewInfo, Sol).

solve_one(-, fail(Vs, Lits), RestGs, As, Cs, Ns, Info, Sol) :-
    (safe_select_failure_literal(Lits, Vs, (Type, L), RestLits, RestGs, NewGs) ->
    %fail_one(Type, L, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol)
    % Rares
    (RestGs=NewGs ->
      % no grounding
          fail_one(Type, L, Vs, RestLits, NewGs, As, Cs, Ns, Info, Sol)
    ;
      % abducibles grounded
      solve_all(NewGs, As, Cs, Ns, Info, Sol)
    )
    % End Rares
    ;
        inspect_solve_one((-, fail(Vs, Lits)), RestGs, As, Cs, Ns, Info, NewInfo),
        (Lits == [] ->
            fail % fail normally
        ;
            (backup_safe_select_failure_literal(Lits, Vs, (Type, L), RestLits) ->
                fail_one(Type, L, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol)
            ;
                % if "Lits" contains only finite domain constraints and/or
                % real constraints, then we can try to test for satisfiability.
                % if it is satisfiable, then it fails normally (i.e. falsity
                % can be derived); otherwise if it is not satisfiable, then
                % it succeeds.
                (partition_failure_literals(Lits, FCs, RCs) ->
                    (\+ \+ (
                                % can be satisfied?
                                propagate_finite_domain_constraints(FCs, Cs, Cs1),
                                propagate_real_domain_constraints(RCs, Cs1, _)
                            ) ->
                        % yes, satisfiable.  so fail
                        fail
                    ;
                        % no, and hence no floundering, so continue the reasoning
                        solve_all(RestGs, As, Cs, Ns, NewInfo, Sol)
                    )
                ; % indeed, it flounders.
                    write('Floundered: '), write(fail(Vs, Lits)), nl, !,
                    fail
                )
            )
        )
    ).

fail_one(a, A, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((a, A), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    resolve_failure_abducible_with_delta(As, A, Vs, RestLits, FailureGoals),
    append(FailureGoals, RestGs, NewGs),
    NewNs = [fail(Vs, [(a, A)|RestLits])|Ns],
    solve_all(NewGs, As, Cs, NewNs, NewInfo, Sol).

% Rares
fail_one(d, D, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    D =.. [Pred, Cat, PDIdx],
    my_get_value(prob_pred_l, ProbPredL),
    nth0(N, ProbPredL, (Pred, PredIdx, _Type)),
    inspect_fail_one((d, D), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    (
        % case 1: the same category was chosen before
        % NB: if a different category was chosen this fails
        % here we fail
        member((Pred, PDIdx, Cat, _), As),
        fail
    ;
        % case 2: a choice from the pd was not made
        % here we generate the sample space \setminus Cat
        % as alternative solutions
        \+member((Pred, PDIdx, _Cat, _), As),
        % extract NCat the number of categories in the pd
        my_get_value(prob_templates, ProbTemplates),
        nth0(N, ProbTemplates, (_, NCat, _NInst)),
        % compute index Idx of the choice
        X in 1..NCat,
        X #\= Cat,
        label([X]),
        NewD =.. [Pred, X, PDIdx],
        NewGs = [(d,NewD)|RestGs]
    ;
        % case 3: a choice different from Cat exists
        % here we do nothing
        member((Pred, PDIdx, Cat1, _), As),
        Cat =\= Cat1,
        NewGs = RestGs
    ),
    solve_all(NewGs, As, Cs, Ns, NewInfo, Sol).

fail_one(d, D, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    my_get_value(prob_pred_l, ProbPredL),
    D =.. [Pred | _],
    \+member((Pred, _), ProbPredL),
    inspect_fail_one((d, D), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    findall(H-B, (rule((d, H), B), unifiable(H, D)), Rules),
    resolve_failure_non_abducible_with_rules(Rules, D, Vs, RestLits, FailureGoals),
    append(FailureGoals, RestGs, NewGs),
    solve_all(NewGs, As, Cs, Ns, NewInfo, Sol).
% End Rares

fail_one(b, B, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((b, B), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    findall(B-[], call(B), Rules),
    resolve_failure_non_abducible_with_rules(Rules, B, Vs, RestLits, FailureGoals),
    append(FailureGoals, RestGs, NewGs),
    solve_all(NewGs, As, Cs, Ns, NewInfo, Sol).

fail_one(c, C, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((c, C), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    findall(C-[], (
        C = current_abducibles(X) ->
            X = As
        ; C = current_remaining_goals(X) ->
            RestGs
        ;
            call(C) % FIXME
    ), Rules),
    resolve_failure_non_abducible_with_rules(Rules, C, Vs, RestLits, FailureGoals),
    append(FailureGoals, RestGs, NewGs),
    solve_all(NewGs, As, Cs, Ns, NewInfo, Sol).

fail_one(e, X = Y, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((e, X = Y), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    ((var(X), strictmember(Vs, X)) ->
        % X is uni. quant. and we don't care about Y
        strictdelete(Vs, X, NewVs),
        call(X = Y),
        solve_one(-, fail(NewVs, RestLits), RestGs, As, Cs, Ns, NewInfo, Sol)
    ; (var(Y), strictmember(Vs, Y)) ->
        % Y is uni. quant. but X is not
        strictdelete(Vs, Y, NewVs),
        call(Y = X),
        solve_one(-, fail(NewVs, RestLits), RestGs, As, Cs, Ns, NewInfo, Sol)
    ; var(X) ->
        % X is ex. quant.
        % NB: by the safe selection strategy, "Y" doesn't contain any
        % universally quantified variable
        (
            % try to succeed in the inequality
            propagate_inequalities([X=/=Y], Cs, NewCs),
            solve_all(RestGs, As, NewCs, Ns, NewInfo, Sol)
        ;
            % try to succeed in the equality and fail one of the rest literals
            call(X = Y),
            solve_one(-, fail(Vs, RestLits), RestGs, As, Cs, Ns, NewInfo, Sol)
        )
    ; var(Y) ->
        % Y is ex. quant. but X is not
        % NB: by the safe selection strategy, "X" doesn't contain any
        % universally quantified variable
        (
            % try to succeed in the inequality
            propagate_inequalities([Y=/=X], Cs, NewCs),
            solve_all(RestGs, As, NewCs, Ns, NewInfo, Sol)
        ;
            % try to succeed in the equality and fail one of the rest literals
            call(Y = X),
            solve_one(-, fail(Vs, RestLits), RestGs, As, Cs, Ns, NewInfo, Sol)
        )
    ; % one of them is a variable
        (unifiable(X, Y, Es) ->
            maplist(wrap_atom, Es, EsW),
            append(EsW, RestLits, NewBody),
            solve_one(-, fail(Vs, NewBody), RestGs, As, Cs, Ns, NewInfo, Sol)
        ;
            % this literal fails trivially
            solve_all(RestGs, As, Cs, Ns, NewInfo, Sol)
        )
    ).

fail_one(i, X=/=Y, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((i, X=/=Y), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    (
        call(X = Y),
        solve_all(RestGs, As, Cs, Ns, NewInfo, Sol)
    ;
        propagate_inequalities([X=/=Y], Cs, NewCs),
        solve_one(-, fail(Vs, RestLits), RestGs, As, NewCs, Ns, NewInfo, Sol)
    ).

fail_one(n, \+ G, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((n, \+ G), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    (
        % Case 1:
        solve_all([G|RestGs], As, Cs, Ns, NewInfo, Sol)
    ;
        % case 2:
        G = (Type, L),
        fail_one(Type, L, [], [], [(-, fail(Vs, RestLits))|RestGs], As, Cs, Ns, NewInfo, Sol)
    ).

fail_one(f, C, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((f, C), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    (special_domain_condition(C, Vs) ->
        % need to treat this as a built-in for X in Dom
        C = (X in Dom),
        findall((X in Dom)-[], call((X in Dom, fd_label([X]))), Rules),
        resolve_failure_non_abducible_with_rules(Rules, C, Vs, RestLits, FailureGoals),
        append(FailureGoals, RestGs, NewGs),
        solve_all(NewGs, As, Cs, Ns, NewInfo, Sol)
    ;
        fd_entailment(C, B), % reitification
        (B == 0 ->
            % the constraint can never hold anyway
            solve_all(RestGs, As, Cs, Ns, NewInfo, Sol)
        ; B == 1 ->
            % the constraint always hold
            solve_one(-, fail(Vs, RestLits), RestGs, As, Cs, Ns, NewInfo, Sol)
        ; % the constraint can hold and can unhold
            % then we either (a) fail it or (b) succeed it and fail the rest
            (
                negate_finite_domain_constraint(C, NC),
                propagate_finite_domain_constraints([NC], Cs, NewCs),
                solve_all(RestGs, As, NewCs, Ns, NewInfo, Sol)
            ;
                propagate_finite_domain_constraints([C], Cs, NewCs),
                solve_one(-, fail(Vs, RestLits), RestGs, As, NewCs, Ns, NewInfo, Sol)
            )
        )
    ).

fail_one(r, C, Vs, RestLits, RestGs, As, Cs, Ns, Info, Sol) :-
    inspect_fail_one((r, C), Vs, RestLits, RestGs, As, Cs, Ns, Info, NewInfo),
    negate_real_domain_constraint(C, NC),
    C = {Cons}, NC = {NCons},
    (entailed(NCons) ->
        % the constraint can never hold anyway
        solve_all(RestGs, As, Cs, Ns, NewInfo, Sol)
    ; entailed(Cons) ->
        % the constraint always hold
        solve_one(-, fail(Vs, RestLits), RestGs, As, Cs, Ns, NewInfo, Sol)
    ; % we either (a) fail it, or (b) succeed it and fail the rest.
        (
            propagate_real_domain_constraints([NC], Cs, NewCs),
            solve_all(RestGs, As, NewCs, Ns, NewInfo, Sol)
        ;
            propagate_real_domain_constraints([C], Cs, NewCs),
            solve_one(-, fail(Vs, RestLits), RestGs, As, NewCs, Ns, NewInfo, Sol)
        )
    ).

% --- Auxiliary Predicates for the Meta-Interpreters ---

% initialise(Query, InitGs, Abducibles, Constraints, Denials, Info)
initialise(Query, InitGs, [], ([], [], []), InitNs, NewInfo) :-
    maplist(wrap_literal, Query, QueryW),
    collect_static_ics(FGs, InitNs),
    append(QueryW, FGs, InitGs),
    inspect_initial_state(Query, NewInfo).

% Rares
% discard ICs
initialise_no_ics(Query, InitGs, [], ([], [], []), InitNs, NewInfo) :-
    maplist(wrap_literal, Query, QueryW),
  InitGs = QueryW,
  InitNs = [],
    inspect_initial_state(Query, NewInfo).
% End Rares

collect_static_ics(FailureGoals, DynamicDenials) :-
    findall(IC, ic(IC), ICs),
    partition_ics(ICs, FailureGoals, DynamicDenials).

partition_ics([], [], []).
partition_ics([IC|Rest], AllFs, AllNs) :-
  % Rares
  % modified so that all ics are added to the goal
    term_variables(IC, Vs),
  AllFs = [(-, fail(Vs, IC))|Fs],
  AllNs = Ns,
    partition_ics(Rest, Fs, Ns).

resolve_abducible_with_delta([], _, []).
resolve_abducible_with_delta([H|T], A, [(A=/=H)|IEs]) :-
    unifiable(H, A), !,
    resolve_abducible_with_delta(T, A, IEs).
resolve_abducible_with_delta([_|T], A, IEs) :-
    resolve_abducible_with_delta(T, A, IEs).

resolve_abducible_with_dynamic_ics([], _, []).
resolve_abducible_with_dynamic_ics([fail(Vs, [(a, Left)|Rest])|RestNs], A, [F|Fs]) :-
    unifiable(Left, A), !,
    rename_universal_variables(Vs, Left-Rest, RnVs, RnLeft-RnRest),
    unifiable(RnLeft, A, Bindings),
    bind_universal_variables(Bindings, RnVs, RemainingBindings, RemainingRnVs),
    maplist(wrap_atom, RemainingBindings, Es),
    append(Es, RnRest, NewRest),
    F = (-, fail(RemainingRnVs, NewRest)),
    resolve_abducible_with_dynamic_ics(RestNs, A, Fs).
resolve_abducible_with_dynamic_ics([_|RestNs], A, Fs) :-
    resolve_abducible_with_dynamic_ics(RestNs, A, Fs).

bind_universal_variables([], Vs, [], Vs).
bind_universal_variables([(A=B)|T], Vs, NewEs, NewVs) :-
    (strictmember(Vs, A) ->
        % A is universally quantified, so bind it
        strictdelete(Vs, A, Vs1),
        call(A=B),
        bind_universal_variables(T, Vs1, NewEs, NewVs)
    ;
        % A is not universally quantified, so keep the equality
        bind_universal_variables(T, Vs, Es1, NewVs),
        NewEs = [(A=B)|Es1]
    ).

% Rares
bind_universal_variables_2([], Vs, NewUnivLits, [], NewVs, NewUnivLits) :-
  term_variables(Vs, NewVs).
bind_universal_variables_2([(A=B)|T], Vs, UnivLits, NewEs, NewVs, NewUnivLits) :-
    (strictmember(Vs, A) ->
        % A is universally quantified, so bind it
        strictdelete(Vs, A, Vs1),
        call(A=B),
        bind_universal_variables_2(T, Vs1, UnivLits, NewEs, NewVs, NewUnivLits)
    ;
    ( (ground(A), ground(B)) ->
      % A and B are ground
      call(A=B),
          bind_universal_variables_2(T, Vs, UnivLits, Es1, NewVs, NewUnivLits)
    ;
      % A is not universally quantified, so keep the equality
          bind_universal_variables_2(T, Vs, UnivLits, Es1, NewVs, NewUnivLits),
          NewEs = [(A=B)|Es1]
    )
    ).
% End Rares

% rename all the universally quantified variables in a given term.
% HACK: the term_variables/2 in YAP and that in SICStus behave
% differently.  Therefore, the implementation of "rename_universal_variables"
% can't rely on them any more.  Below is a new implementation that should work
% on both systems

:- dynamic mycopy/1.
rename_universal_variables(VsIn, FormIn, VsOut, FormOut) :-
    assert(mycopy(VsIn-FormIn)), retract(mycopy(VsOut-FormOut)),
    rebind_existential_variables(FormIn, VsIn, FormOut).
rebind_existential_variables(Pin, VsIn, Pout) :-
    (atomic(Pin) ->
        true % nothing to bind
    ; var(Pin) ->
        (strictmember(VsIn, Pin) ->
            % this is a universally quantified variable. so do not bind it
            true
        ;
            % this is an existentially quantified variable, so BIND it.
            Pout = Pin
        )
    ; % this is a compound term
        Pin =.. [F|ArgsIn],
        Pout =.. [F|ArgsOut],
        rebind_existential_variables_recursively(ArgsIn, VsIn, ArgsOut)
    ).
rebind_existential_variables_recursively([], _, []).
rebind_existential_variables_recursively([H1|T1], VsIn, [H2|T2]) :-
    rebind_existential_variables(H1, VsIn, H2),
    rebind_existential_variables_recursively(T1, VsIn, T2).

% Rares
safe_select_failure_literal(Lits, Vs, (Type, L), RestLits, RestGs, NewGs) :-
  (get_value(ss, u) ->
    unfold_select((Type, L), Lits, RestLits, Vs, RestGs, NewGs)
  ; get_value(ss, u_old) ->
    unfold_select_old((Type, L), Lits, RestLits, Vs, RestGs, NewGs)
  ; get_value(ss, o) ->
    % original def
    select((Type, L), Lits, RestLits),
    safe_failure_literal(Type, L, Vs), !
  ;
    write('Invalid option for the ss flag! Failing!'),fail
  ).

unfold_select((Type,L), Lits, RestLits, Vs, RestGs, NewGs) :-
  Lits \= [],
  split_denial(Lits, Vs, ABL, NGL, OL),
  (get_value(dbg_query,2) ->
    write('split_denial'),nl,
    write(Lits),nl,
    write(ABL),nl,
    write(NGL),nl,
    write(OL),nl ; true
  ),
  uss((Type,L), ABL, NGL, OL, RestLits, Vs, RestGs, NewGs).

% splits the set of literals in 3 disjoint sets
split_denial([], _Vs, [], [], []).
% ABL case (abducible literals)
split_denial([H|T], Vs, [H|TABL], NGL, OL) :-
  H = (Type1, Lit1),
  (
    Type1 == a ;
    Type1 == n, Lit1 =.. [_Neg, Atm2], Atm2 = (Type2, _Lit2), Type2 == a
  ),
  !,
  split_denial(T, Vs, TABL, NGL, OL).
% NGL case (neg non-g def, non-g constraints)
split_denial([H|T], Vs, ABL, [H|TNGL], OL) :-
  H = (Type1, Lit1),
  \+ safe_failure_literal(Type1, Lit1, Vs),
  !,
  split_denial(T, Vs, ABL, TNGL, OL).
% OL case (pos def, neg ground def, ground constraints)
split_denial([H|T], Vs, ABL, NGL, [H|TOL]) :-
  split_denial(T, Vs, ABL, NGL, TOL).

% OL not empty => select an element from it
uss((Type,L), ABL, NGL, OL, RestLits, _Vs, RestGs, NewGs) :-
  OL \= [],
  !,
  OL = [(Type, L)|T],
  append(ABL, NGL, RL1),
  append(RL1, T, RestLits),
  NewGs = RestGs.
% otherwise, intersect Y_NGL and Y_ABL to ground abducibles
uss((_Type,_L), ABL, NGL, [], RestLits, Vs, RestGs, NewGs) :-
  get_vars(ABL, YABL),
  get_vars(NGL, YNGL),
  var_intersect(YABL, YNGL, Inter),
  Inter \= [],
  !,
  (get_value(dbg_query,2) ->
    write('inter'),nl,
    write(Inter),nl ; true
  ),
  get_abducibles_with_var(ABL, Inter, ABLtoGround, RestAbds),
  (get_value(dbg_query,2) ->
    write('before ground'),nl,
    write(ABLtoGround),nl,
    write(RestAbds),nl,
    write(NGL),nl ; true
  ),
  append(RestAbds, NGL, RestLits),
  findall((NewVs, ABLtoGround, UnivLits), force_types_list(ABLtoGround, Vs, RestLits, NewVs, UnivLits), L1),
  (get_value(dbg_query,2) ->
    write('after ground'),nl,
    write(L1),nl ; true
  ),
  make_denials(L1, GroundedGs),
  append(RestGs, GroundedGs, NewGs),
  (get_value(dbg_query,2) ->
    write(NewGs),nl ; true
  ).
% otherwise select from ABL, ground or positive non-ground
uss((Type,L), ABL, NGL, [], RestLits, _Vs, RestGs, NewGs) :-
  ABL \= [],
  select((Type, L), ABL, RestABL),
  (ground(L) ; Type == a),
  !,
  append(RestABL, NGL, RestLits),
  NewGs = RestGs.
% otherwise select from ABL, negative non-ground => ground
uss((_Type,_L), ABL, NGL, [], RestLits, Vs, RestGs, NewGs) :-
  ABL \= [],
  %write('ground neg nonground ABL'),nl,
  RestLits = NGL,
  %write(ABL), nl,
  findall((NewVs, ABL, UnivLits), force_types_list(ABL, Vs, RestLits, NewVs, UnivLits), L1),
  %write('after ground'),nl,
  %write(L1),nl,
  make_denials(L1, GroundedGs),
  append(RestGs, GroundedGs, NewGs).

get_vars(L, Vars) :- get_vars(L, Vars, []).
get_vars([], Vars, Vars).
get_vars([(_Type, Lit)|T], Vars, VarAcc) :-
  term_variables(Lit, VarLit),
  append(VarAcc, VarLit, VarAcc1),
  get_vars(T, Vars, VarAcc1).

var_intersect([], _Var2, []).
var_intersect([H|T], Var2, [H|T2]) :-
  member(V, Var2),
  H == V,
  !,
  var_intersect(T, Var2, T2).
var_intersect([_H|T], Var2, Inter) :-
  var_intersect(T, Var2, Inter).

get_abducibles_with_var([], _Inter, [], []).
get_abducibles_with_var([(Type, Lit)|T], Inter, [(Type, Lit)|T2], RestAbds) :-
  term_variables(Lit, VarLit),
  var_intersect(VarLit, Inter, Inter2),
  Inter2 \= [],
  !,
  get_abducibles_with_var(T, Inter, T2, RestAbds).
get_abducibles_with_var([H|T], Inter, ABLtoGround, [H|T2]) :-
  get_abducibles_with_var(T, Inter, ABLtoGround, T2).

% old unfold select

unfold_select_old(X, Lits, RestLits, Vs, RestGs, NewGs) :-
  unfold_select_old(X, Lits, RestLits, Vs, RestGs, NewGs, [], []).

% base - no abducibles, should fail
unfold_select_old((Type,L), [], RestLits, Vs, NewGs, NewGs, [], UnivLits) :-
  !,
  select((Type, L), UnivLits, RestLits),
  safe_failure_literal(Type, L, Vs).

% 'base' - there are abducibles
unfold_select_old((Type,L), [], RestLits, Vs, RestGs, NewGs, Abds, UnivLits) :-
  %write('Abds: '),write(Abds),nl,
  %write('UnivLits: '),write(UnivLits),nl,
  %write('Vars: '),write(Vs),nl,
  %force_types_list(Abds, Vs, UnivLits, NewVs, UnivLits),
  ((UnivLits = [] ; ground_list(Abds)) ->
    Abds = [(Type, L)|RestLits],
    NewGs = RestGs
  ;
    findall((NewVs, Abds, UnivLits), force_types_list(Abds, Vs, UnivLits, NewVs, UnivLits), L1),
    %write('L1: '),write(L1),nl,
    %write('unfold NewUnivLits: '),write(UnivLits),nl,
    ground_denials(L1, GroundedGs),
    %write('RestGs: '),
    %write(RestGs),nl,
    %write('L: '), write(L), nl,fail,
    %findall(NewGoal, (force_types_list(Abds, Vs, NewVs), append(Abds, RestLits, L1), append(L1, UnivLits, NewGoalLits), NewGoal = (-, fail(NewVs, NewGoalLits)) ), GroundedGs),
    append(RestGs, GroundedGs, NewGs)
  ).
  %write('NewGs: '), write(NewGs), nl.
  %write('Abds: '),write(Abds),nl,
  %write('NewVs: '),write(NewVs),nl.
%fail,
% append(Abds, UnivLits, NewLits),
% unfold_select((Type, L), NewLits, RestLits, Vs).

% base - select if non abducible and non Univ
unfold_select_old((Type,L), [(Type, L)|T], RestLits, Vs, NewGs, NewGs, Abds, UnivLits) :-
  (
    Type \== a ;
    Type == n, L=(Type2, _L2), Type2 \== a
  ),
  safe_failure_literal(Type, L, Vs),!,
  append(T, Abds, List1),
  append(List1, UnivLits, RestLits).

unfold_select_old((Type,L), [(Type1, L1)|T],  RestLits, Vs, RestGs, NewGs, Abds, UnivLits) :-
  safe_failure_literal(Type1, L1, Vs),
  (
    Type1 == a ;
    Type1 == n, L1=(Type2, _L2), Type2 == a
  ), % this should be unnecessary
  unfold_select_old((Type, L), T, RestLits, Vs, RestGs, NewGs, [(Type1, L1)|Abds], UnivLits).

unfold_select_old((Type,L), [(Type1, L1)|T], RestLits, Vs, RestGs, NewGs, Abds, UnivLits) :-
  \+ safe_failure_literal(Type1, L1, Vs), % this should be unnecessary
  unfold_select_old((Type, L), T, RestLits, Vs, RestGs, NewGs, Abds, [(Type1, L1)|UnivLits]).

ground_list([]).
ground_list([H|T]) :-
  ground(H),
  ground_list(T).

ground_denials([], []).
ground_denials([(Vs, Abds, UnivLits)|T], [GroundedGoal| TGrounded]) :-
  append(UnivLits, Abds, GroundedLits),
  GroundedGoal = (-, fail(Vs, GroundedLits)),
  ground_denials(T, TGrounded).








%%% Post processing predicates
% remove non-abducibles from the denial store (first arg.) to get second arg.
remove_non_abducibles([], []).
remove_non_abducibles([fail(_Vs, Body)|T], [fail(Vs2, Body2)|T2]) :-
  remove_non_abducibles_denial(Body, Body2),
  term_variables(Body2, Vs2), % update variables in Body2
  remove_non_abducibles(T, T2).

% same, but for a single denial
remove_non_abducibles_denial([], []).
% the literal in the body is abducible => copy it
remove_non_abducibles_denial([(Type, Lit)|T], [(Type, Lit)|T2]) :-
  (
    Type == a
  ;
    Type == n, Lit =.. [_Neg,(Type2, _Lit2)], Type2 == a
  ),
  !,
  remove_non_abducibles_denial(T, T2).
% the literal in the body is non abducible => remove it
% (and possibly some vars, but that's done in remove_non_abducibles)
remove_non_abducibles_denial([_H|T], Body2) :-
  remove_non_abducibles_denial(T, Body2).

% ground abducibles after proof
ground_abducibles(Ns, GroundNs) :- once(ground_abducibles(Ns, GroundNs, [])).
ground_abducibles([], GroundNs, GroundNs).
ground_abducibles([fail(Vs, Body)|T], GroundNs, GNsAcc) :-
  findall((NewVs, Body, []), force_types_list(Body, Vs, [], NewVs, []), L1),
  make_denials_2(L1, GroundedGs),
  append(GNsAcc, GroundedGs, GNsAcc1),
  ground_abducibles(T, GroundNs, GNsAcc1).

% remove abducibles in delta (As) from the body of a denial (i.e. set difference)
% or remove the denial if it contains \+ a and a is in As
remove_delta_denials([], [], _As).
% remove denial if it contains \+ a and a is in As
remove_delta_denials([fail([], Body)|T], Body2, As) :-
  member(Lit, Body),
  Lit =.. [_Neg, A],
  member(A, As),
  !,
  remove_delta_denials(T, Body2, As).
% remove literal from the body of the denial
remove_delta_denials([fail([], Body)|T], [Body2|T2], As) :-
  subtract(Body, As, Body2),
  remove_delta_denials(T, T2, As).

% remove denials which contain as subsets other denials
remove_subset_denials(Ns, NsClean) :- once(remove_subset_denials(Ns, NsClean, [])).
remove_subset_denials([], NsClean, NsClean).
remove_subset_denials([Body|T], NsClean, NsAcc) :-
  append(NsAcc, T, Denials),
  ( (member(D, Denials), subset(D, Body)) ->
    NsAcc1 = NsAcc
  ;
    NsAcc1 = [Body|NsAcc]
  ),
  remove_subset_denials(T, NsClean, NsAcc1).

% separate the denials of type <- a., i.e. of length 1, from the rest
split_false_atoms([], [], []).
% false atom case (copy to second argument)
split_false_atoms([[Atom]|T], [Atom|TNAs], OtherNs) :-
  !,
  split_false_atoms(T, TNAs, OtherNs).
% other denials are simply copied to 3rd argument
split_false_atoms([Denial|T], NAs, [Denial|TOtherNs]) :-
  split_false_atoms(T, NAs, TOtherNs).

% End Rares

% 11/07/2011 Bug fixed: floundering case
backup_safe_select_failure_literal(Lits, Vs, (f, X in Dom), RestLits) :-
    select((f, X in Dom), Lits, RestLits),
    strictmember(Vs, X),
    ground(Dom).

special_domain_condition(X in Dom, Vs) :-
    strictmember(Vs, X),
    ground(Dom).

% 17/09/2010 Bug fixed: thanks Domenico Corapi
partition_failure_literals([], [], []).
partition_failure_literals([(f,C)|T], [C|FCs], RCs) :-
    partition_failure_literals(T, FCs, RCs).
partition_failure_literals([(r,C)|T], FCs, [C|RCs]) :-
    partition_failure_literals(T, FCs, RCs).

safe_failure_literal(n, L, Vs) :-
    % negative literal must not contain universally quantified
    % variables
    variables_within_term(Vs, L, []).
safe_failure_literal(i, L, Vs) :-
    % simliary inequality must not involve universally quantified
    % variables
    variables_within_term(Vs, L, []).
safe_failure_literal(f, L, Vs) :-
    % finite domain constraint with universally quantified variables
    % should not be selected at first
    variables_within_term(Vs, L, []).
safe_failure_literal(r, L, Vs) :-
    % similarly real domain constraint with universally quantified
    % variables should not be selected at first
    variables_within_term(Vs, L, []).
safe_failure_literal(e, X = Y, Vs) :-
    % for equalitiy it is a bit trickier
    % FIXME (low priority): at the moment, the equality store does
    % not handle inequalities with universally quantified variables.
    % Thus, equalities between one existentially quantified variable
    % and a compound term with universally quantified variables should
    % not be selected.
    \+ (
        % the following means "X is an existentially quantified variable
        % but Y is a compound term with universally quantified variables"
        var(X), \+ strictmember(Vs, X),
        \+ var(Y), \+ variables_within_term(Vs, Y, [])
    ),
    \+ (
        % the following means "Y is an existentially quantified variable
        % but X is a compound term with universally quantified variables"
        var(Y), \+ strictmember(Vs, Y),
        \+ var(X), \+ variables_within_term(Vs, X, [])
    ).
% finally, everything else is safe.
safe_failure_literal(a, _, _).
safe_failure_literal(b, _, _).
safe_failure_literal(d, _, _).

resolve_failure_abducible_with_delta([], _, _, _, []).
resolve_failure_abducible_with_delta([H|T], A, Vs, RestLits, [FailureGoal|FailureGoals]) :-
    unifiable(H, A), !,
    rename_universal_variables(Vs, A-RestLits, RnVs, RnA-RnRestLits),
    unifiable(RnA, H, Es),
    bind_universal_variables(Es, RnVs, RemainingEs, RemainingRnVs),
    maplist(wrap_atom, RemainingEs, EsW),
    append(EsW, RnRestLits, NewBody),
    FailureGoal = (-, fail(RemainingRnVs, NewBody)),
    resolve_failure_abducible_with_delta(T, A, Vs, RestLits, FailureGoals).
resolve_failure_abducible_with_delta([_|T], A, Vs, RestLits, FailureGoals) :-
    resolve_failure_abducible_with_delta(T, A, Vs, RestLits, FailureGoals).

resolve_failure_non_abducible_with_rules([], _, _, _, []).
resolve_failure_non_abducible_with_rules([H-B|T], P, Vs, RestLits,
        [FailureGoal|FailureGoals]) :-
    rename_universal_variables(Vs, P-RestLits, RnVs, RnP-RnRestLits),
    term_variables(H-B, NewVs),
    append(NewVs, RnVs, AllVs), % no need to use union/3 here as "NewVs" must be fressh
    unifiable(H, RnP, Es),
    bind_universal_variables(Es, AllVs, RemainingEs, RemainingVs),
    maplist(wrap_atom, RemainingEs, EsW),
    append(EsW, B, NewB), append(NewB, RnRestLits, AllBody),
    FailureGoal = (-, fail(RemainingVs, AllBody)),
    resolve_failure_non_abducible_with_rules(T, P, Vs, RestLits, FailureGoals).

% --- Utilities for Inequality and Constraint stores ---

enforce_labeling(true) :-
    set_value(lbl, true).
enforce_labeling(false) :-
    set_value(lbl, false).

finite_domain_inequality(X=/=Y) :-
    fd_var(X) ; fd_var(Y) ; % already be fd var
    (var(X), integer(Y)) ; (var(Y), integer(X)). % about to become fd var

:- if(current_prolog_flag(dialect, yap)).
%{
fd_entailment(C, B) :-
    call(C #<==> B).

real_domain_inequality(X=/=Y) :-
    get_attr(X, itf, _) ; get_attr(Y, itf, _) ; % already be rd var
    (var(X), float(Y)) ; (var(Y), float(X)). % about to become rd var

fd_label(Vs) :-
    label(Vs).
%}
:- elif(current_prolog_flag(dialect, sicstus)).
%{
fd_entailment(C, B) :-
    call(C #<=> B).

real_domain_inequality(X=/=Y) :-
    % get_atts(X, itf) ; get_atts(Y, itf) ; % already be rd var % FIXME: can't get it working
    (var(X), float(Y)) ; (var(Y), float(X)). % about to become rd var

fd_label(Vs) :-
    labeling([], Vs).
%}
:- endif.

propagate_inequalities(Es, (E, F, R), (NewE, NewF, NewR)) :-
    add_inequalities(Es, (E, F, R), (NewE, NewF, NewR)).

add_inequalities([], Cs, Cs).
add_inequalities([(X=/=Y)|T], (Ei, Fi, Ri), (Eo, Fo, Ro)) :-
    (finite_domain_inequality(X=/=Y) ->
        % push it to the finite domain constraint store
        fd_entailment((X #\= Y), B),
        (B == 1 ->
            % already reitified, so can discard it
            add_inequalities(T, (Ei, Fi, Ri), (Eo, Fo, Ro))
        ; % otherwise, try to add it to the finite domain constraint store
            call(X #\= Y),
            add_inequalities(T, (Ei, [(X #\= Y)|Fi], Ri), (Eo, Fo, Ro))
        )
    ; real_domain_inequality(X=/=Y) ->
        % push it to the real domain constraint store
        (entailed(X =\= Y) ->
            % already entailed, so can discard it
            add_inequalities(T, (Ei, Fi, Ri), (Eo, Fo, Ro))
        ; % otherwise, try to add it to the real domain constraint store
            call({X =\= Y}),
            add_inequalities(T, (Ei, Fi, [{X =\= Y}|Ri]), (Eo, Fo, Ro))
        )
    ; % a real inequality?
        (unifiable(X, Y) ->
            % still need to keep it as the inequality could be falsified
            call(X=/=Y),
            add_inequalities(T, ([(X=/=Y)|Ei], Fi, Ri), (Eo, Fo, Ro))
        ;
            add_inequalities(T, (Ei, Fi, Ri), (Eo, Fo, Ro))
        )
    ).

propagate_finite_domain_constraints(Fs, (E,F,R), (E,NewF,R)) :-
    add_finite_domain_constraints(Fs, F, NewF),
    finite_domain_soundness_check(NewF).

add_finite_domain_constraints([], F, F).
add_finite_domain_constraints([H|T], Fin, Fout) :-
    fd_entailment(H, B),
    (B == 1 ->
        % can discard it
        add_finite_domain_constraints(T, Fin, Fout)
    ;
        call(H),
        add_finite_domain_constraints(T, [H|Fin], Fout)
    ).

propagate_real_domain_constraints(Rs, (E,F,R), (E,F,NewR)) :-
    add_real_domain_constraints(Rs, R, NewR).

add_real_domain_constraints([], R, R).
add_real_domain_constraints([H|T], Rin, Rout) :-
    H = {C},
    (entailed(C) ->
        % can discard it
        add_real_domain_constraints(T, Rin, Rout)
    ;
        call(H),
        add_real_domain_constraints(T, [H|Rin], Rout)
    ).

negate_finite_domain_constraint(C1 #\/ C2, NC1 #/\ NC2) :-
    negate_finite_domain_constraint(C1, NC1),
    negate_finite_domain_constraint(C2, NC2).
negate_finite_domain_constraint(C1 #/\ C2, NC1 #\/ NC2) :-
    negate_finite_domain_constraint(C1, NC1),
    negate_finite_domain_constraint(C2, NC2).
negate_finite_domain_constraint(#\ C, C).
negate_finite_domain_constraint(X #< Y, X #>= Y).
negate_finite_domain_constraint(X #> Y, X #=< Y).
negate_finite_domain_constraint(X #=< Y, X #> Y).
negate_finite_domain_constraint(X #>= Y, X #< Y).
negate_finite_domain_constraint(X #\= Y, X #= Y).

negate_real_domain_constraint({C1 , C2}, {NC1 ; NC2}) :-
    negate_real_domain_constraint({C1}, {NC1}),
    negate_real_domain_constraint({C2}, {NC2}).
negate_real_domain_constraint({C1 ; C2}, {NC1 , NC2}) :-
    negate_real_domain_constraint({C1}, {NC1}),
    negate_real_domain_constraint({C2}, {NC2}).
negate_real_domain_constraint({X < Y}, {X >= Y}).
negate_real_domain_constraint({X > Y}, {X =< Y}).
negate_real_domain_constraint({X =< Y}, {X > Y}).
negate_real_domain_constraint({X >= Y}, {X < Y}).
negate_real_domain_constraint({X =\= Y}, {X = Y}).
negate_real_domain_constraint({X = Y}, {X =\= Y}).
negate_real_domain_constraint({X =:= Y}, {X =\= Y}).

domain_bounded(X) :- fd_size(X, S), S \== sup.

finite_domain_soundness_check(F) :-
    term_variables(F, Vs),
    selectlist(domain_bounded, Vs, BVs),
    (get_value(lbl, true) ->
        fd_label(BVs)
    ;
        \+ \+ fd_label(BVs)
    ).

extract_constraints((Es, Fs, Rs), Cs) :-
    % finite domain constraints soundness check
    finite_domain_soundness_check(Fs),
    % collect inequalities
    term_variables(Es, EsVs),
    maplist(inequalities, EsVs, Ess),
    my_flatten(Ess, Es1),
    list_to_ord_set(Es1, Es2),
    % collect ground finite domain constraints
    selectlist(nonground, Fs, Fs1),
    % collect real domain constraints
    term_variables(Rs, RsVs),
    dump(RsVs, RsVs, Rs1),
    append(Fs1, Rs1, Cs1),
    append(Es2, Cs1, Cs).

% --- Inequality Store ---

% public
X =/= Y :-
    (var(X) ; var(Y)), !,
    X \== Y,
    reinforce_neq(X, Y),
    reinforce_neq(Y, X).
X =/= Y :-
    (unifiable(X, Y, Eqs) ->
        (Eqs = [A = B] ->
            A =/= B % no choice point
        ;
            member(A = B, Eqs), % backtrackable
            A =/= B
        )
    ;
        true
    ).

reinforce_neq(A, B) :-
    var(A), !,
    (get_atts(A, aliens(S)) ->
        (\+ strictmember(S, B) -> NewS = [B|S] ; NewS = S),
        put_atts(A, aliens(NewS))
    ;
        put_atts(A, aliens([B]))
    ).
reinforce_neq(_, _).

strictmember([H|T], X) :-
    (X == H ->
        true
    ;
        strictmember(T, X)
    ).

% hook
verify_attributes(Var, Val, Goals) :-
    get_atts(Var, aliens(S1)), !, % are we involved?
    \+ strictmember(S1, Val), % is it an alien?
    ((var(Val), get_atts(Val, aliens(S2))) ->
    % thanks Domenico Corapi for helping with fixing the bug, 2010/03/31
    %(var(Val) ->
        %get_atts(Val, aliens(S2)),
        % \+ strictmember(S2, Var) % this should be implied by the previous test
        list_to_ord_set(S2, NewS2),
        list_to_ord_set(S1, NewS1),
        ord_union(NewS2, NewS1, S3), % copy forward aliens
        put_atts(Val, aliens(S3)),
        Goals = []
    ;
        generate_goals(S1, Val, Goals)
    ).
verify_attributes(_, _, []).

generate_goals([], _, []).
generate_goals([H|T], Val, Gs) :-
    generate_goals(T, Val, Gs1),
    (var(H) ->
        Gs = Gs1
    ;
        Gs = [(Val =/= H)|Gs1]
    ).

% hook
attribute_goal(Var, Goal) :-
    get_atts(Var, aliens(S)),
    list_to_ord_set(S, S1),
    construct_body(S1, Var, Goal).

construct_body([H|T], Var, Goal) :-
    (T = [] ->
        Goal = (Var =/= H)
    ;
        construct_body(T, Var, G),
        Goal = ((Var =/= H),G)
    ).

% public
inequalities(Var, Ineqs) :-
    get_atts(Var, aliens(S)), !,
    list_to_ord_set(S, S1),
    collect_inequalities(S1, Var, Ineqs).
inequalities(_, []).

collect_inequalities([], _, []).
collect_inequalities([H|T], Var, [N|Rest]) :-
    (Var @=< H ->
        N = (Var =/= H)
    ;
        N = (H =/= Var)
    ),
    collect_inequalities(T, Var, Rest).

% ------------------------------------------------------------------------
% ---------- Extensions (Experimental) ---------------
% EXP

:- use_module(library(timeout)).
:- use_module(library(system)).

% similar to eval/2 (see below) but with timeout
eval(Query, TimeOut, Delta) :-
    time_out(query_with_minimal_solutions(Query, MinSols), TimeOut, Result),
    (Result == success ->
        member((_Length, Query-Delta), MinSols)
    ;
        write('TIMEOUT! Cannot compute all ground solutions and global minimality '),
        write('of explanations cannot be guaranteed.  Please use "query('),
        write(Query), write('), (Delta, Constraints, Denials))." instead.'), nl, fail
    ).

% eval(Query, Delta) holds if and only if Delta is the minimal ground explanation
% for Query.  Currently it only works if we can compute all the ground solutions
% for the given query.
eval(Query, Delta) :-
    query_with_minimal_solutions(Query, MinSols),
    member((_Length, Query-Delta), MinSols).

query_with_minimal_solutions(Query, SortedSolutions) :-
    % 1. compute all the ground answers
    findall(Query-SortedAs, (
            query(Query, (As, Cs, _Ns)),
            term_variables(Cs, Vs),
            selectlist(domain_bounded, Vs, BVs),
            fd_label(BVs),
            list_to_ord_set(As, SortedAs)
    ), Solutions),
    % 2. make sure everything is ground
    ground(Solutions),
    % 3. select the minimal ones
    select_minimal_solutions(Solutions, [], MinSolutions),
    % 4. sort it according to its length
    sort_solutions_according_to_length(MinSolutions, SortedSolutions).

select_minimal_solutions([], MinSols, MinSols).
select_minimal_solutions([H|T], Sols, MinSols) :-
    ((is_minimal(T, H), is_minimal(Sols, H)) ->
        NewSols = [H|Sols]
    ;
        NewSols = Sols
    ),
    select_minimal_solutions(T, NewSols, MinSols).

is_minimal([], _).
is_minimal([H|T], X) :-
    \+ ground_subsumes(H, X),
    is_minimal(T, X).

ground_subsumes(Q-D1, Q-D2) :-
    ord_intersection(D1, D2, Inter),
    D1 = Inter.

sort_solutions_according_to_length(MinSols, SortedSols) :-
    maplist(attach_length, MinSols, Sols),
    sort(Sols, SortedSols).

attach_length(Q-D, (L, Q-D)) :-
    length(D, L).

% forcing the argument of P to be typed.
force_types(P) :-
    (\+ types(P, _) ->
        true
    ;
        types(P, Conds),
        force_all_type_conditions(Conds)
    ).

% Rares

% force_types/2 (P, Vars) is the same as force_types/1
% except Vars should NOT be typed
force_types(P, Vars) :-
    (\+ types(P, _) ->
        true
    ;
        types(P, Conds),
        force_all_type_conditions(Conds, Vars)
    ).
% same as force_tpyes/2, ensure Param vars are unified
force_types(P, Vars, Params) :-
    (\+ types(P, _) ->
        true
    ;
        types(P, Conds),
        force_all_type_conditions(Conds, Vars, Params)
    ).


% types supported:
% - type(Var, Type)
% - X = Y
% - * type(Pred), where Pred appears as a fact and grounds the variable in it
force_all_type_conditions([]).
force_all_type_conditions([C|T]) :-
    (C = (X = Y) ->
        call(C)
    ; C = type(X, Y) ->
        enum(Y, D),
        member(X, D)
  ; C = type(Pred),
    rule((d,Pred), []),
    ground(Pred)
    ),
    force_all_type_conditions(T).

force_all_type_conditions([], _Vars).
force_all_type_conditions([C|T], Vars) :-
    ( (C = (X = Y), member(Z, Vars), \+ ground(Z), X==Z) ->
        call(C)
  ;
    (C = (X = Y), member(Z, Vars), \+ ground(Z), Y==Z) ->
        call(C)
    ; (C = type(X, Y), member(Z, Vars), \+ ground(Z), X==Z) ->
        enum(Y, D),
        member(X, D)
  ;
    % for type type(Pred)
    (C=type(Pred), Pred=..[_F|Args], member(Z, Args), \+ ground(Z), X==Z) ->
    rule((d,Pred), []),
    ground(Pred)
  ;
    true
    ),
    force_all_type_conditions(T, Vars).
% with params, the def is the same as force_all_type_conditions/2
force_all_type_conditions([], _Vars, _Params).
force_all_type_conditions([C|T], Vars, Params) :-
    ( (C = (X = Y), member(Z, Vars), \+ ground(Z), X==Z) ->
        call(C)
  ;
    (C = (X = Y), member(Z, Vars), \+ ground(Z), Y==Z) ->
        call(C)
    ; (C = type(X, Y), member(Z, Vars), \+ ground(Z), X==Z) ->
        enum(Y, D),
        member(X, D)
  ;
    % for type type(Pred)
    (C=type(Pred), Pred=..[_F|Args], member(Z, Args), \+ ground(Z), X==Z) ->
    rule((d,Pred), []),
    ground(Pred)
  ;
    true
    ),
    force_all_type_conditions(T, Vars, Params).

% force types on a list of literals
% Vs acts as accumulator
force_types_list([], NewVs, NewUnivLits, NewVs, NewUnivLits).
force_types_list([(Type, A)|T], Vs, UnivLits, NewVs, NewUnivLits) :-
  Type \== n,
  !,
  force_types_univ(A, Vs, UnivLits, UpdVs, NewUnivLits),
  force_types_list(T, UpdVs, UnivLits, NewVs, NewUnivLits).
force_types_list([(n, A)|T], Vs, UnivLits, NewVs, NewUnivLits) :-
  A =.. [_Neg, (_Type, Lit)],
  !,
  force_types_univ(Lit, Vs, UnivLits, UpdVs, NewUnivLits),
  force_types_list(T, UpdVs, UnivLits, NewVs, NewUnivLits).

force_types_univ(P, Vs, UnivLits, NewVs, NewUnivLits) :-
    (\+ types(P, _) ->
    % warning - no type found in this case
    (get_value(dbg_query,2) ->
      write('!!!!!!!!!!!!!!!!! NO TYPE '),nl ; true
    ),
    NewVs = Vs,
        true
    ;
        types(P, Conds),
        force_all_type_conditions_univ(Conds, Vs, UnivLits, NewVs, NewUnivLits)
    ).

force_all_type_conditions_univ([], NewVs, NewUnivLits, NewVs, NewUnivLits).
force_all_type_conditions_univ([C|T], Vs, UnivLits, NewVs, NewUnivLits) :-
    (
    C = (_X1 = _Y1) ->
        bind_universal_variables_2([C], Vs, UnivLits, _Es, UpdVs, NewUnivLits)
    ;
        enum(Y, D),
        member(Z, D),
    C = type(X, Y),
    % call Z = X
    bind_universal_variables_2([(X=Z)], Vs, UnivLits, _Es, UpdVs, NewUnivLits)
  ;
    C = type(Pred),
    rule((d,Pred), []),
    ground(Pred),
    term_variables(Vs, UpdVs)
    ),
    force_all_type_conditions_univ(T,UpdVs, UnivLits, NewVs, NewUnivLits).

% re-make denials after grounding
make_denials([], []).
make_denials([(Vs, Abds, UnivLits)|T], [GroundedGoal| TGrounded]) :-
  append(UnivLits, Abds, GroundedLits),
  GroundedGoal = (-, fail(Vs, GroundedLits)),
  make_denials(T, TGrounded).
% without wrapping (-, ...)
make_denials_2([], []).
make_denials_2([(Vs, Abds, UnivLits)|T], [GroundedGoal| TGrounded]) :-
  append(UnivLits, Abds, GroundedLits),
  GroundedGoal = fail(Vs, GroundedLits),
  make_denials_2(T, TGrounded).

% End Rares


% -- enhanced querying interface

eval_all(Query) :-
    nl,
    write('+++++++++++++++ Start +++++++++++++++++'), nl, nl,
    now(Time1),
    findall(Query-Delta, eval(Query, Delta), Ans),
    now(Time2),
    Diff is Time2 - Time1,
    length(Ans, Len),
    (ground(Query) ->
        write('Query: '), portray_clause(Query),
        forall(member(_-D, Ans), (
            write(' <= '), portray_clause(D)
        ))
    ;
        forall(member(Q-D, Ans), (
            write('Query: '), portray_clause(Q),
            write(' <= '), portray_clause(D)
        ))
    ),
    nl,
    write('Total execution time (seconds): '), write(Diff), nl,
    write('Total minimal explanations found: '), write(Len), nl, nl,
    write('---------------  End  -----------------'), nl,
    nl.

eval_all(Query, GroundQueryHypothesesPairs) :-
    findall((Query, Delta), eval(Query, Delta), GroundQueryHypothesesPairs).

eval_all_with_ground_query(Query, AllGroundHypotheses) :-
    ground(Query),
    findall(Delta, eval(Query, Delta), AllGroundHypotheses).

query_all(Query) :-
    nl,
    write('+++++++++++++++ Start +++++++++++++++++'), nl,
    write('Original Query: '), write(Query), nl, nl,
    now(Time1),
    findall(Query-Delta, query(Query, (Delta,_,_)), Ans),
    now(Time2),
    Diff is Time2 - Time1,
    length(Ans, Len),
    (ground(Query) ->
        write('Query: '), portray_clause(Query),
        forall(member(_-D, Ans), (
            write(' <= '), portray_clause(D)
        ))
    ;
        forall(member(Q-D, Ans), (
            write('Query: '), portray_clause(Q),
            write(' <= '), portray_clause(D)
        ))
    ),
    nl,
    write('Total execution time (seconds): '), write(Diff), nl,
    write('Total explanations found: '), write(Len), nl, nl,
    write('---------------  End  -----------------'), nl,
    nl.

% Rares



/*
query_ss(Q, Ans, SelectionStrategy, Debug) :-
  (Debug == no_debug ->
   set_value(dbg_read, 0),
   set_value(dbg_query, 0),
   set_value(dbg_write, 0)
 ; Debug == debug_read(N) ->
    set_value(dbg_read, N)
  ; Debug == debug_write(N) ->
    set_value(dbg_write, N)
 ; Debug == debug_query(N) ->
    set_value(dbg_query, N)
 ;
   write('Invalid debug option')
 ),
 (SelectionStrategy == classic ->
   set_value(ss, o),
   query(Q, Ans)
 ; SelectionStrategy == uss1 ->
   set_value(ss, u_old),
   query(Q, Ans)
 ; SelectionStrategy == aprob ->
   set_value(ss, u),
   query_process(Q, Ans)
 ;
   write('Invalid Selection Strategy')
 ).
*/

query_exact_prob_joint(Q, P, D) :-
  query_exact_prob_joint(Q, '../temp_files/aprob_bdd.cpp', _Paths, _ProbL, P, D).

query_exact_prob_joint(Q, F, Paths, ProbL, P, D) :-
    findall((As, NAs, Ns), query_ss(Q, (As, NAs, Ns,_), aprob, D), L1),
    ((get_value(dbg_write,V0), V0>0) ->
      write('q and write'),nl,
      write(L1),nl ; true
    ),
    remove_temp_r(L1, L2), % remove temp_r abducibles from solutions
    write('L2'),nl,
    write_L2(L2),
    write('End L2'),nl,
    ((get_value(dbg_write, V1), V1>0) ->
        write(L2NoNeg),nl ; true
    ),
    remove_neg(L2, L2NoNeg),
    ((get_value(dbg_write, V1), V1>0) ->
        write(L2NoNeg),nl ; true
    ),
    order_abducibles(L2NoNeg, AbdOrder),
    ((get_value(dbg_write,V2), V2>0) ->
        write('AbdOrder'),nl,
        write(AbdOrder),nl ; true
    ),
    get_probs(AbdOrder, ProbL),
    ((get_value(dbg_write,V3), V3>0) ->
        write(ProbL),nl ; true
    ),
    set_value(solution_bdd, 1),
    set_value(print_bdd, 1),
    write('ProbL'),nl,
    write(ProbL),nl,
    (get_value(solution_bdd, 1) ->
        write_bdd_sols(L2, ProbL)
    ;
        write_to_file(L2, ProbL, F),
        ((get_value(dbg_write,V4), V4>0) ->
            write('write_ok'),nl ; true
        ),
        call_bdd_script,
        % process bdd output
        call_py_bdd_proc_script,
        % read from file
        read_from_bdd(Paths, QueryProb, '../temp_files/aprob_bdd_proc.txt'),
        QueryProb = prob(P),
        ((get_value(dbg_write,V5), V5>1) ->
            write('+++++++++++++++ Query Probability +++++++++++++++++'), nl,
        write('  P = '), write(P), nl; true
        )
    ).

query_exact_prob(Q, P, D) :-
  query_exact_prob(Q, '../temp_files/aprob_bdd.cpp', _Paths, _Paths_IC, _ProbL, _ProbL_IC, P, D).

query_exact_prob(Q, F, Paths, PathsNew, ProbL, ProbLNew, P, D) :-
  % part 1 query IC
  statistics(walltime, [Time1|_]),
  findall((As, NAs, Ns), query_ss([], (As, NAs, Ns,_), aprob, D), L1),
  statistics(walltime, [Time2|_]),
  Diff12 is Time2-Time1,
  %format(user_output, 'Abduction 1 ~3d sec.~n', [Diff12]),
  statistics(walltime, [Time3|_]),
  ((get_value(dbg_write,V0), V0>0) ->
    write('q and write'),nl,
    write(L1),nl ; true
  ),
  %remove_temp_r(L1, L2), % remove temp_r abducibles from solutions
  %write_L2(L2),
  ((get_value(dbg_write, V1), V1>0) ->
    write(L2),nl ; true
  ),
  %remove_neg(L2, L2NoNeg),
  ((get_value(dbg_write, V1), V1>0) ->
    write(L2NoNeg),nl ; true
  ),
  %order_abducibles(L2NoNeg, AbdOrder),
  ((get_value(dbg_write,V2), V2>0) ->
    write('AbdOrder'),nl,
    write(AbdOrder),nl ; true
  ),
  %get_probs(AbdOrder, ProbL),
  ((get_value(dbg_write,V3), V3>0) ->
    write(ProbL),nl ; true
  ),
  %write_to_file(L2, ProbL, F),
  ((get_value(dbg_write,V4), V4>0) ->
    write('write_ok'),nl ; true
  ),
  %statistics(walltime, [Time4|_]),
  %Diff34 is Time4-Time3,
  %format(user_output, 'Prepare 1 ~3d sec.~n', [Diff34]),
  %statistics(walltime, [Time5|_]),
  %call_bdd_script,
  %statistics(walltime, [Time6|_]),
  %Diff56 is Time6-Time5,
  %format('BDD 1 ~3d sec.~n', [Diff56]),
  %statistics(walltime, [Time7,_]),
  % process bdd output
  %call_py_bdd_proc_script,
  % read from file
  %read_from_bdd(Paths, QueryProb, '../temp_files/aprob_bdd_proc.txt'),
  %QueryProb = prob(P_IC),
  %statistics(walltime, [Time8|_]),
  %Diff78 is Time8-Time7,
  %format('Read_BDD 1 ~3d sec.~n', [Diff78]),
  % part 2 - query P
  statistics(walltime, [Time9|_]),
  maplist(wrap_literal, Q, QW),
  expand_tree(QW, L1, LNew),
  statistics(walltime, [Time10|_]),
  Diff910 is Time10-Time9,
  %format('Abduction 2 ~3d sec.~n', [Diff910]),
  % post process and bdd stuff
  %statistics(walltime, [Time11|_]),
  %remove_temp_r(LNew, L2New), % remove temp_r abducibles from solutions
  ((get_value(dbg_write, V5), V5>0) ->
    write(L2New),nl ; true
  ),
  %remove_neg(L2New, L2NoNegNew),
  ((get_value(dbg_write, V6), V6>0) ->
    write(L2NoNegNew),nl ; true
  ),
  %order_abducibles(L2NoNegNew, AbdOrderNew),
  ((get_value(dbg_write,V7), V7>0) ->
    write('AbdOrder'),nl,
    write(AbdOrderNew),nl ; true
  ),
  %get_probs(AbdOrderNew, ProbLNew),
  ((get_value(dbg_write,V8), V8>0) ->
    write(ProbLNew),nl ; true
  ),
  %write_to_file(L2New, ProbLNew, F),
  ((get_value(dbg_write,V9), V9>0) ->
    write('write_ok'),nl ; true
  ),
  %statistics(walltime, [Time12|_]),
  %Diff1112 is Time12-Time11,
  %write('Prepare 2: '),write(Diff1112),nl,
  %statistics(walltime, [Time13|_]),
  %call_bdd_script,
  %statistics(walltime, [Time14|_]),
  %Diff1314 is Time14-Time13,
  %write('BDD 2: '),write(Diff1314),nl,
  % process bdd output
  %statistics(walltime, [Time15|_]),
  %call_py_bdd_proc_script,
  % read from file
  %read_from_bdd(PathsNew, QueryProb_IC, '../temp_files/aprob_bdd_proc.txt'),
  %QueryProb_IC = prob(P_Joint),
  % part 3 - P(Q|IC) = P(Q,IC)/P(IC)
  %P is P_Joint/P_IC,
  %statistics(walltime, [Time16|_]),
  %Diff1516 is Time16-Time15,
  %write('Read_BDD 2: '),write(Diff1516),nl,
  AbductionTotal is (Diff12+Diff910)/1000,
  %ProcessTotal is (Diff34+Diff78+Diff1112+Diff1516)/1000,
  %BDDTotal is (Diff56+Diff1314)/1000,
  %TotalTime is AbductionTotal+ProcessTotal+BDDTotal,
  open('../out_files/aprob_times.log', append, Stream),
  format(Stream, 'Abduction Total|~4f~n', [AbductionTotal]),
  %format(Stream, 'Process Total|~4f~n', [ProcessTotal]),
  %format(Stream, 'Bdd Total|~4f~n', [BDDTotal]),
  %format(Stream, 'Total Time|~4f~n', [TotalTime]),
  close(Stream),
  write(P),nl.

write_L2([]).
write_L2([(As, NAs, Ns)|T]) :-
  write(As),nl,
  write(NAs),nl,
  write(Ns),nl,
  nl,
  write_L2(T).

%expand_tree_thresh(L, NewLSucc, NewLPend) :-
%    expand_tree_thresh(L, NewLSucc, NewLPend, [], [], []).

%expand_tree_thresh(OldL, NewLSucc, NewLPend, CurrentL, SuccAcc, PendAcc) :-
%    expand_tree_thresh(OldL, ).


expand_tree(QW, L1, LNew) :- expand_tree(QW, L1, LNew, []).

expand_tree(_QW, [], LNew, LNew).
expand_tree(QW, [(As, NAs, Ns)|T], LNew, LAcc) :-
  merge_ics(NAs, Ns, AllNs),
  wrap_ics(AllNs, WNs),
  findall((AsNew, NAsNew, NsNew), query_process_no_init(QW, As, ([], [], []), WNs, [], (AsNew, NAsNew, NsNew, _Info)), LNew1),
  append(LAcc, LNew1, LAcc1),
  expand_tree(QW, T, LNew, LAcc1).

merge_ics(NAs, Ns, AllNs) :- merge_ics(NAs, Ns, AllNs, []).
merge_ics([], Ns, AllNs, LAcc) :- append(LAcc, Ns, AllNs).
merge_ics([H|T], Ns, AllNs, LAcc) :-
  append(LAcc, [[H]], LAcc1),
  merge_ics(T, Ns, AllNs, LAcc1).

wrap_ics([], []).
wrap_ics([H|T], [H2|T2]) :-
  maplist(wrap_literal, H, HW),
  H2 = fail([], HW),
  wrap_ics(T, T2).

query_exact_prob_nets(Q, P, D) :-
  set_value(print_bdd, 1),
  set_value(print_paths, 1),
  query_exact_prob_joint(Q, '../temp_files/aprob_bdd.cpp', Paths, ProbL, P, D),
  process_bdd_paths(Paths, ProcessedPaths, ProbL),
  ((get_value(dbg_write,V1), V1>1) ->
    write('process bdd ok'),nl ; true
  ),
  plot_nets(ProcessedPaths).

query_exact_prob_paths(Q, P, D) :-
  query_exact_prob_joint(Q, '../temp_files/aprob_bdd.cpp', Paths, ProbL, P, D),
  process_bdd_paths(Paths, ProcessedPaths, ProbL),
  ((get_value(dbg_write,V1), V1>1) ->
    write('process bdd ok'),nl,
      write('+++++++++++++++ BDD Paths +++++++++++++++++'), nl ; true
  ),
  write_paths(ProcessedPaths).

query_exact_prob_paths_file(Q, P, D) :-
  query_exact_prob_joint(Q, '../temp_files/aprob_bdd.cpp', Paths, ProbL, P, D),
  process_bdd_paths(Paths, ProcessedPaths, ProbL),
  ((get_value(dbg_write,V1), V1>1) ->
    write('process bdd ok'),nl ; true
  ),
  write_paths(ProcessedPaths, '../out_files/bdd_paths.txt').

query_nets(Q, SS, D) :-
  findall(As, query_ss(Q, (As, _NAs, _Ns,_), SS, D), L1),
  plot_nets(L1).

query_exact_prob_joint_max_nets(Q, P, D) :-
  query_exact_prob_joint(Q, '../temp_files/aprob_bdd.cpp', Paths, ProbL, P, D),
  get_probs2(ProbL2),
  atoms_from_probl(ProbL2, Atoms2),
  atoms_from_probl(ProbL, Atoms),
  subtract(Atoms2, Atoms, DontCareAtoms2),
  process_bdd_paths(Paths, ProcessedPaths, ProbL),
  get_max_nets(ProcessedPaths, MaxNets, DontCareAtoms2),
  write(DontCareAtoms2),nl,
  write('Max Nets'),nl,
  write(MaxNets),nl,
  plot_nets(MaxNets).

atoms_from_probl([], []).
atoms_from_probl([(Atom, _Prob)|T1], [Atom|T2]) :-
    atoms_from_probl(T1, T2).

get_max_nets(PPaths, MaxNets, DontCareAtoms2) :-
    get_max_nets(PPaths, MaxNets, DontCareAtoms2, [], 0).

get_max_nets([], MaxNets, _DCA2, MaxNets, _MaxSize).
get_max_nets([(TrueAtoms, DontCareAtoms)|T], MaxNets, DontCareAtoms2, MaxNetsAcc, MaxSize) :-
    append(TrueAtoms, DontCareAtoms, MaxNet1),
    append(MaxNet1, DontCareAtoms2, MaxNet),
    length(MaxNet, LenMaxNet),
    ((LenMaxNet > MaxSize) ->
        MaxSize1 is LenMaxNet,
        MaxNetsAcc1 = [MaxNet]
    ; (LenMaxNet =:= MaxSize) ->
        MaxSize1 is MaxSize,
        MaxNetsAcc1 = [MaxNet|MaxNetsAcc]
    ;
        MaxSize1 is MaxSize,
        MaxNetsAcc1 = MaxNetsAcc
    ),
    get_max_nets(T, MaxNets, DontCareAtoms2, MaxNetsAcc1, MaxSize1).



plot_nets(ProcessedPaths) :- plot_nets(ProcessedPaths, 0).

plot_nets([], N) :-
  open('../out_files/nets/all_nets.dot', write, Stream),
  write(Stream, 'graph{\nnode [shape=none]\n'),
  write_nets(Stream, N, 0),
  write(Stream, '}\n'),
  close(Stream),
  call_big_net_script.
plot_nets([(TrueAtoms, DontCareAtoms)|T], NAcc) :-
  !,
  number_codes(NAcc, NStr),
  append("../out_files/nets/net_", NStr, Str1),
  append(Str1, ".dot", FName),
  name(FNameStr, FName),
  open(FNameStr, write, Stream),
  write(Stream, 'digraph G{\n'),
  write(Stream, 'graph [fontsize=36];\n'),
  write(Stream, 'node [fontsize=36];\n'),
  write(Stream, 'rankdir=LR;\n'),
  %% for net prob
  %write(Stream, 'labelloc="t";\nlabel="P={}";\n'),
  write_net_edges(Stream, TrueAtoms, ',color="black"'),
  write_net_edges(Stream, DontCareAtoms, ',color="red"'),
  write(Stream, '}\n'),
  close(Stream),
  call_net_script(NStr),
  NAcc1 is NAcc+1,
  plot_nets(T, NAcc1).
% to work for Deltas only
plot_nets([TrueAtoms|T], NAcc) :-
  !,
  number_codes(NAcc, NStr),
  append("../out_files/nets/net_", NStr, Str1),
  append(Str1, ".dot", FName),
  name(FNameStr, FName),
  open(FNameStr, write, Stream),
  write(Stream, 'digraph G{\n'),
  write(Stream, 'graph [fontsize=36];\n'),
  write(Stream, 'node [fontsize=36];\n'),
  write(Stream, 'rankdir=LR;\n'),
  %% for net prob
  %write(Stream, 'labelloc="t";\nlabel="P={}";\n'),
  write(TrueAtoms),nl,
  write_net_edges(Stream, TrueAtoms, ',color="black"'),
  write(Stream, '}\n'),
  close(Stream),
  call_net_script(NStr),
  NAcc1 is NAcc+1,
  plot_nets(T, NAcc1).

write_paths(Paths) :- write_paths(user_output, Paths, 1).
write_paths(Paths, File) :-
  open(File, write, Stream),
  write_paths(Stream, Paths, 1),
  close(Stream).

write_paths(_Stream, [], _NAcc).
write_paths(Stream, [(TrueAtoms, DontCareAtoms)|T], NAcc) :-
  write(Stream, 'Solution '),
  write(Stream, NAcc),
  write(Stream, '\n'),
  write(Stream, 'True: '),
  write_list(Stream, TrueAtoms, ','),
  write(Stream, '\n'),
  write(Stream, 'Don\'t Care: '),
  write_list(Stream, DontCareAtoms, ','),
  write(Stream, '\n'),
  NAcc1 is NAcc+1,
  write_paths(Stream, T, NAcc1).
write_nets(_Stream, N, N) :- !.
write_nets(Stream, N, Acc) :-
  write(Stream, 'd'),
  write(Stream, Acc),
  write(Stream, ' [image="net_'),
  write(Stream, Acc),
  write(Stream,'.png",label=""]\n'),
  Acc1 is Acc+1,
  write_nets(Stream, N, Acc1).

write_net_edges(_Stream, [], _Color).
write_net_edges(Stream, [H|T], Color) :-
  H =.. [PredName, G1, G2, S],
  (PredName==r ->
    write(Stream, G2),
    write(Stream, ' -> '),
    write(Stream, G1),
    write(Stream, '[style=filled'),
    (S=1 ->
      write(Stream, ',arrowhead="normal"')
    ;
      write(Stream, ',arrowhead="inv"')
    ),
    write(Stream, Color),
    write(Stream, ']\n')
  ; PredName == or ->
    write(Stream, G2),
    write(Stream, ' -> '),
    write(Stream, G1),
    write(Stream, '[style=dotted'),
    (S=1 ->
      write(Stream, ',arrowhead="normal"')
    ;
      write(Stream, ',arrowhead="inv"')
    ),
    write(Stream, Color),
    write(Stream, ']\n')
  ;
    write('Error! Unexpected predicate name: '), write(PredName),write(' !!!'),nl
  ),
  write_net_edges(Stream, T, Color).

call_net_script(N) :-
  absolute_file_name('$SHELL', Shell),
  append("dot -Tpng ../out_files/nets/net_", N, Str1),
  append(Str1, ".dot > ../out_files/nets/net_", Str2),
  append(Str2, N, Str3),
  append(Str3, ".png", Command),
  name(CommandStr, Command),
  process_create(Shell,['-c', [CommandStr]], [process(Proc)]),
  process_wait(Proc, exit(ExitCode)),
  (ExitCode =:= 0  ->
    true
  ;
    write('!!!!!!!!!!!!!! Script run failed!'),nl
  ).

call_big_net_script :-
  current_directory(_C1, '../out_files/nets'),
  absolute_file_name('$SHELL', Shell),
  process_create(Shell,['-c', ['dot -Tpng all_nets.dot > all_nets.png']], [process(Proc)]),
  process_wait(Proc, exit(ExitCode)),
  current_directory(_C2, '../../src'),
  ((get_value(dbg_write,V1), V1>1) ->
    write(ExitCode),nl ; true
  ),
  (ExitCode =:= 0  ->
    write('big net script_ok'),nl
  ;
    write('!!!!!!!!!!!!!! Script run failed!'),nl
  ).


call_bdd_script :-
  absolute_file_name('$SHELL', Shell),
  process_create(Shell,['-c', ['../sh_files/aprob_script.sh']], [process(Proc)]),
  process_wait(Proc, exit(ExitCode)),
  ((get_value(dbg_write,V1), V1>1) ->
    write(ExitCode),nl ; true
  ),
  (ExitCode =:= 0  ->
    write('bdd script_ok'),nl
  ;
    write('!!!!!!!!!!!!!! Script run failed!'),nl
  ).

call_bdd_script(F) :-
    ScriptPathStr = "../sh_files/aprob_script.sh",
    atom_codes(ScriptPath, ScriptPathStr),
    % create script
    open(ScriptPath, write, Stream),
    atom_codes(F, Fstr),
    append(Fstr, ".cpp", FcppStr),
    append(Fstr, "_out.txt", FoutStr),
    atom_codes(Fcpp, FcppStr),
    atom_codes(Fout, FoutStr),
    format(Stream, 'g++ ~s -o ~s -lbdd\n', [Fcpp, F]),
    format(Stream, './~s > ~s', [F, Fout]),
    close(Stream),
    absolute_file_name('$SHELL', Shell),
    process_create(Shell,['-c', [ScriptPath]], [process(Proc)]),
    process_wait(Proc, exit(ExitCode)),
    ((get_value(dbg_write,V1), V1>1) ->
      write(ExitCode),nl ; true
    ),
    (ExitCode =:= 0  ->
      write('bdd script_ok'),nl
    ;
      write('!!!!!!!!!!!!!! Script run failed!'),nl
    ).

call_py_bdd_proc_script :-
  absolute_file_name('$SHELL', Shell),
  process_create(Shell,['-c', ['../py_files/post_proc_bdd.py']], [process(Proc)]),
  process_wait(Proc, exit(ExitCode)),
  ((get_value(dbg_write,V5), V5>1) ->
    write(ExitCode),nl ; true
  ),
  (ExitCode =:= 0  ->
    write('bdd proc script_ok'),nl
  ;
    write('!!!!!!!!!!!!!! Script run failed!'),nl
  ).

read_from_bdd(Paths, QueryProb, F) :-
  open(F, read, Stream),
  read_file(Stream, L),
  close(Stream),
  L=[QueryProb|Paths].

read_file(Stream,L) :-
    read(Stream,X),
    (X = end_of_file ->
      L = []
    ;
      L = [X|T],
      read_file(Stream,T)
    ).

process_bdd_paths(Paths1, Paths, ProbL) :-
  length(ProbL, Len),
  process_bdd_paths(Paths1, Paths, ProbL, Len).
process_bdd_paths([], [], _ProbL, _Len).
process_bdd_paths([bdd_path(L)|T], [(TrueAtoms, DontCareAtoms)|T1], ProbL, Len) :-
  process_bdd_path(L, TrueAtoms, DontCareAtoms, ProbL, Len),
  process_bdd_paths(T, T1, ProbL, Len).

process_bdd_path(L, TrueAtoms, DontCareAtoms, ProbL, Len) :-
  get_bdd_path_true_atoms(L, TrueAtoms, AllIdx, ProbL),
  gen_interval(0, Len, Interval),
  subtract(Interval, AllIdx, DontCareIdx),
  get_bdd_path_dont_care_atoms(DontCareIdx, DontCareAtoms, ProbL).

get_bdd_path_true_atoms([], [], [], _ProbL).
get_bdd_path_true_atoms([(Idx, 1)|T1], [Atom|T2], [Idx|T3], ProbL) :-
  once(nth0(Idx, ProbL, (Atom, _Prob))),
  get_bdd_path_true_atoms(T1, T2, T3, ProbL).
get_bdd_path_true_atoms([(Idx, 0)|T1], TrueAtoms, [Idx|T3], ProbL) :-
  get_bdd_path_true_atoms(T1, TrueAtoms, T3, ProbL).

get_bdd_path_dont_care_atoms([], [], _ProbL).
get_bdd_path_dont_care_atoms([Idx|T1], [Atom|T2], ProbL) :-
  once(nth0(Idx, ProbL, (Atom, _Prob))),
  get_bdd_path_dont_care_atoms(T1, T2, ProbL).

remove_temp_r([], []).
remove_temp_r([(As, NAs, Ns)|T], [(NewAs, NewNAs, NewNs)|T2]) :-
  findall(X, (member(X,As), X\=temp_r(_,_,_)), NewAs),
  findall(X, (member(X,NAs), X\=temp_r(_,_,_)), NewNAs),
  exclude(member_temp_r,  Ns, NewNs),
  remove_temp_r(T, T2).

member_temp_r(X) :-
  is_list(X),
  member(temp_r(_,_,_), X).

remove_neg([], []).
remove_neg([(As, NAs, Ns)|T], [(As, NAs, NsNoNeg)|T2]) :-
  remove_neg_denials(Ns, NsNoNeg),
  remove_neg(T, T2).

remove_neg_denials([], []).
remove_neg_denials([H|T], [H2|T2]) :-
  remove_neg_denial(H, H2),
  remove_neg_denials(T, T2).

remove_neg_denial([], []).
remove_neg_denial([H|T], [H2|T2]) :-
  (H=..['\\+', Arg] ->
    H2 = Arg
  ;
    H2 = H
  ),
  remove_neg_denial(T, T2).

order_abducibles2(AbdOrder) :-
    get_value(pa_index, PaIdx),
    gen_interval(0, PaIdx, Interval),
    maplist(add_pa, Interval, AbdOrder).

order_abducibles(L1, AbdOrder) :- order_abducibles(L1, AbdOrder, [], []).
order_abducibles([], AbdOrder, FirstAcc, LastAcc) :-
  subtract(LastAcc, FirstAcc, LastDiff),
  append(FirstAcc, LastDiff, AbdOrder).
order_abducibles([(As, NAs, Ns)|T], AbdOrder, FirstAcc, LastAcc) :-
  list_to_set(As, AsSet),
  list_to_set(NAs, NAsSet),
  union(FirstAcc, AsSet, FATemp),
  union(FATemp, NAsSet, FirstAcc1),
  my_flatten(Ns, NsFlat),
  list_to_set(NsFlat, NsFlatSet),
  (get_value(dbg_query,2) ->
    write(NsFlatSet),nl ; true
  ),
  union(LastAcc,NsFlatSet,LastAcc1),
  order_abducibles(T, AbdOrder, FirstAcc1, LastAcc1).

get_probs([], []).
get_probs([Atom|T], [(Atom, Prob)|T2]) :-
    (
        rule(Head,[]),
        Head=(d,pr(Prob, Atom))
    ;
        pr(Prob, Atom)
    ),
    get_probs(T, T2).


% TO DO
ground_prob_abd(Atom, Prob, ProbL) :-
    (
    rule((d,pr(Prob,Atom)), [])
    ;
    pr(Prob, Atom)
    ),
    Atom =.. [PredName|Args],
    length(Args, LenArgs),
    get_value(remove_abds, RemoveAbds),
    \+ member(PredName/LenArgs, RemoveAbds),
    force_types(Atom),
    \+ member((Atom, _Prob2), ProbL).


get_all_probs(AllProbL, ProbL) :-
    findall((Atom, Prob), ground_prob_abd(Atom, Prob, ProbL), AllProbL1),
    list_to_set(AllProbL1, AllProbL), % this should be unnecessary
    write(AllProbL),nl.


get_probs2(ProbL) :-
    findall((Atom, Prob), ( (rule((d,pr(Prob,Atom)), []); pr(Prob, Atom)), Atom\=temp_r(_,_,_), force_types(Atom) ), ProbL1),
    list_to_set(ProbL1, ProbL),
    write(ProbL),nl.

write_bdd_sols(L, ProbL) :- write_bdd_sols(L, ProbL, 1).

write_bdd_sols([], _ProbL, _N).
write_bdd_sols([H|T], ProbL, N) :-
    number_codes(N, NStr),
    append("../temp_files/aprob_bdd_sol_",NStr, F1),
    append(F1, ".cpp", FStr),
    atom_codes(F, FStr),
    write_to_file([H], ProbL, F, N),
    atom_codes(BddF, F1),
    call_bdd_script(BddF),
    N1 is N+1,
    write_bdd_sols(T, ProbL, N1).

write_to_file(L1, ProbL, File) :-
    open(File, write, Stream),
    write_bdd_header(Stream),
    write_function_bdd_prob(Stream),
    write_bdd_main(Stream, L1, ProbL),
    close(Stream).

write_to_file(L1, ProbL, File, N) :-
    open(File, write, Stream),
    write_bdd_header(Stream),
    write_function_bdd_prob(Stream),
    write_bdd_main(Stream, L1, ProbL, N),
    close(Stream).

write_bdd_header(Stream) :-
  write(Stream, '#include<bdd.h>\n#include<iostream>\nusing namespace std;\n\n').
write_function_bdd_prob(Stream) :-
  write(Stream, 'float bdd_prob(bdd r, float* probs){\n\
if(r==bdd_false()){\n\
return 0.0;\n\
}\n\
if(r==bdd_true()){\n\
return 1.0;\n\
}\n\
return probs[bdd_var(r)]*bdd_prob(bdd_high(r), probs)+(1-probs[bdd_var(r)])*bdd_prob(bdd_low(r), probs);\n\
}\n').

write_bdd_main(Stream, L1, ProbL) :-
  length(ProbL, LenProbL),
  % bdd_main + settings
  write(Stream, 'main(void)\n{\n\
bdd_init(100000, 10000);\nbdd_setvarnum('),
  write(Stream, LenProbL),
  write(Stream, ');\n'),
  % bdd_vars and probs
  write_bdd_vars_and_probs(Stream, ProbL),
  % bdd_subf
  write_bdd_subf(Stream, L1, ProbL),
  % bdd_f
  length(L1, LenL1),
  write_bdd_f(Stream, LenL1),
  % bdd_dump_prob
  write_bdd_dump_prob(Stream),
  % bdd_end_main
  write_bdd_end_main(Stream).
write_bdd_main(Stream, L1, ProbL, N) :-
  length(ProbL, LenProbL),
  write(Stream, 'main(void)\n{\n\
bdd_init(100000, 10000);\nbdd_setvarnum('),
  write(Stream, LenProbL),
  write(Stream, ');\n'),
  write_bdd_vars_and_probs(Stream, ProbL),
  write_bdd_subf(Stream, L1, ProbL),
  length(L1, LenL1),
  write_bdd_f(Stream, LenL1),
  write_bdd_dump_prob(Stream, N),
  write_bdd_end_main(Stream).

write_list(_Stream, [], _Sep).
write_list(Stream, [H], _Sep) :- write(Stream,H),!.
write_list(Stream, [H1,H2|T], Sep) :-
    write(Stream, H1),
    write(Stream, Sep),
    write_list(Stream, [H2|T], Sep).

write_literal(Stream, L) :-
  L =.. ['\\+',Atom],
  !,
  write(Stream, '!'),
  write_atom(Stream, Atom).
write_literal(Stream, L) :-
  write_atom(Stream, L).

write_atom(Stream, Atom) :-
  Atom =.. [Pred|Args],
  write(Stream, Pred),
  (Args \= [] ->
    write(Stream, '_'),
    write_list(Stream, Args, '_')
  ;
    true
  ).
write_var(Stream, Lit, ProbL) :-
  (Lit =.. ['\\+',Atom] ->
    write(Stream, '!'),
    once(nth0(N, ProbL, (Atom, _Prob))),
    write(Stream, 'v'),
    write(Stream, N)
  ;
    once(nth0(N, ProbL, (Lit, _Prob))),
    write(Stream, 'v'),
    write(Stream, N)
  ).

write_literal_list(_Stream, [], _Sep).
write_literal_list(Stream, [Lit], _Sep) :- write_literal(Stream, Lit),!.
write_literal_list(Stream, [H|T], Sep) :-
  write_literal(Stream, H),
  write(Stream, Sep),
  write_literal_list(Stream, T, Sep).

write_var_literal_list(_Stream, [], _Sep, _ProbL).
write_var_literal_list(Stream, [Lit], _Sep, ProbL) :- write_var(Stream, Lit, ProbL),!.
write_var_literal_list(Stream, [H|T], Sep, ProbL) :-
  write_var(Stream, H, ProbL),
  write(Stream, Sep),
  write_var_literal_list(Stream, T, Sep, ProbL).

write_var_neg_literal_list(_Stream, [], _Sep, _ProbL).
write_var_neg_literal_list(Stream, [Lit], _Sep, ProbL) :-
  write(Stream, '!'),
  write_var(Stream, Lit, ProbL),
  !.
write_var_neg_literal_list(Stream, [H|T], Sep, ProbL) :-
  write(Stream, '!'),
  write_var(Stream, H, ProbL),
  write(Stream, Sep),
  write_var_neg_literal_list(Stream, T, Sep, ProbL).

write_var_denials(_Stream, [], _Sep, _ProbL).
write_var_denials(Stream, [Denial], _Sep, ProbL) :-
  write(Stream, '!('),
  write_var_literal_list(Stream, Denial, '&', ProbL),
  write(Stream, ')').
write_var_denials(Stream, [Denial|T], Sep, ProbL) :-
  write(Stream, '!('),
  write_var_literal_list(Stream, Denial, '&', ProbL),
  write(Stream, ')'),
  write(Stream, Sep),
  write_var_denials(Stream, T, Sep, ProbL).

write_denials2(_Stream, [], _DenialSep, _BodySep).
write_denials2(Stream, [fail(_Vs, Body)], _DenialSep, BodySep) :- write_list(Stream, Body, BodySep),!.
write_denials2(Stream, [fail(_Vs, Body), H2|T], DenialSep, BodySep) :-
  write_list(Stream, Body, BodySep),
  write(Stream, DenialSep),
  write_denials2(Stream, [H2|T], DenialSep, BodySep).

write_bdd_vars_and_probs(Stream, ProbL) :- write_bdd_vars_and_probs(Stream, ProbL, [],0).
write_bdd_vars_and_probs(Stream, [], Probs, _NAcc) :-
  write(Stream, 'float probs[] = {'),
  write_list(Stream, Probs,','),
  write(Stream, '};\n').
write_bdd_vars_and_probs(Stream, [(_Atom, Prob)|T], Probs, NAcc) :-
  write(Stream, 'bdd '),
  %write_atom(Stream, Atom),
  write(Stream, 'v'),
  write(Stream, NAcc),
  write(Stream, ' = bdd_ithvar('),
  write(Stream, NAcc),
  write(Stream, ');\n'),
  NAcc1 is NAcc+1,
  append(Probs, [Prob], Probs1),
  write_bdd_vars_and_probs(Stream, T, Probs1, NAcc1).

write_bdd_subf(Stream, L1, ProbL) :- write_bdd_subf(Stream, L1, ProbL, 0).
write_bdd_subf(_Stream, [], _ProbL, _NAcc).
write_bdd_subf(Stream, [(As, NAs, Ns)|T], ProbL, NAcc) :-
  write(Stream, 'bdd f'),
  write(Stream, NAcc),
  write(Stream, ' = '),
  (As \= [] ->
    write(Stream, '('),
    write_var_literal_list(Stream, As, '&', ProbL),
    write(Stream, ')&')
  ;
    write(Stream, 'bddtrue&')
  ),
  (NAs \= [] ->
    write(Stream, '('),
    write_var_neg_literal_list(Stream, NAs, '&', ProbL),
    write(Stream, ')&')
  ;
    write(Stream, 'bddtrue&')
  ),
  (Ns \= [] ->
    write(Stream, '('),
    write_var_denials(Stream, Ns, '&', ProbL),
    write(Stream, ')')
  ;
    write(Stream, 'bddtrue')
  ),
  write(Stream, ';\n'),
  NAcc1 is NAcc+1,
  write_bdd_subf(Stream, T, ProbL, NAcc1).

write_bdd_f(Stream, Len) :-
  write(Stream, 'bdd f = '),
    gen_interval(0, Len, Interval),
    maplist(atom_number_concat('f'), Interval, Fs),
  write_list(Stream, Fs, '|'),
  write(Stream, ';\n').

write_bdd_dump_prob(Stream) :-
    (get_value(print_bdd, 1) ->
        write(Stream, 'bdd_fnprintdot("../out_files/bdd.dot",f);\nstd::cout<<endl;\n')
    ;
        true
    ),
    (get_value(print_paths, 1) ->
        write(Stream, 'bdd_printset(f);\nstd::cout<<endl;\n')
    ;
        true
    ),
    write(Stream, 'std::cout<<bdd_prob(f,probs)<<endl;\n').
write_bdd_dump_prob(Stream, N) :-
    (get_value(print_bdd, 1) ->
        number_codes(N, NStr),
        append("../out_files/bdd_sol_", NStr, F1),
        append(F1, ".dot", F),
        format(Stream, 'bdd_fnprintdot("~s",f);\nstd::cout<<endl;\n', [F])
    ;
        true
    ),
    (get_value(print_paths, 1) ->
        write(Stream, 'bdd_printset(f);\nstd::cout<<endl;\n')
    ;
        true
    ),
    write(Stream, 'std::cout<<bdd_prob(f,probs)<<endl;\n').

write_bdd_end_main(Stream) :-
  write(Stream, 'bdd_done();\n}').

% End Rares

% -- state inspection --

:- set_value(depth_limit, 0).

set_depth_limit(N) :-
    N > 0,
    set_value(depth_limit, N).
clear_depth_limit :-
    set_value(depth_limit, 0).

:- set_value(max_states, 0). % by default, no limit
set_max_states(M) :-
    M >= 0,
    set_value(max_states, M).

query_all_with_trace(Query) :-
    query_all_with_trace(Query, 'trace.graphml').

query_all_with_trace(Query, TraceFile) :-
    set_value(dbg, true),
    initialise_trace(TraceFile),
    query_all(Query),
    finalise_trace,
    set_value(dbg, false).

eval_all_with_trace(Query) :-
    eval_all_with_trace(Query, 'trace.graphml').

eval_all_with_trace(Query, TraceFile) :-
    set_value(dbg, true),
    initialise_trace(TraceFile),
    eval_all(Query),
    finalise_trace,
    set_value(dbg, false).

% Rares
query_stats(Query) :-
  set_value(stats, true),
  findall(NewInfo, query(Query, (_D,_C,_N,NewInfo)), Infos),
  open('../temp_files/aprob_stats', write, Stream),
  write_stats_to_file(Infos, Stream),
  close(Stream),
  call_plot1.

write_stats_to_file([], _Stream).
write_stats_to_file([[GoalSizes, DeltaSizes, DenialSizes, Avg, Std]|T], Stream) :-
  write_list(Stream, GoalSizes, ','),
  write(Stream, '\n'),
  write_list(Stream, DeltaSizes, ','),
  write(Stream, '\n'),
  write_list(Stream, DenialSizes, ','),
  write(Stream, '\n'),
  write_list(Stream, Avg, ','),
  write(Stream, '\n'),
  write_list(Stream, Std, ','),
  write(Stream, '\n'),
  write_stats_to_file(T, Stream).

query_stats_process(Query) :-
  set_value(stats, true),
  set_value(post_process, true),
  findall(NewInfo, query_process(Query, (_D,_C,_N,NewInfo)), Infos),
  open('../temp_files/aprob_stats', write, Stream),
  write_stats_to_file(Infos, Stream),
  close(Stream),
  call_plot1.

call_plot1 :-
  absolute_file_name('$SHELL', Shell),
  process_create(Shell,['-c', ['./../py_files/plot_script.py']], [process(Proc)]),
  process_wait(Proc, exit(ExitCode)),
  write(ExitCode),nl,
  (ExitCode =:= 0  ->
    write('plot_script_ok'),nl
  ;
    write('!!!!!!!!!!!!!! Plot script run failed!'),nl
  ).

% End Rares

reset_state_count :-
    set_value(state_count, 0).

increment_state_count :-
    get_value(max_states, MaxVal),
    (MaxVal == 0 ->
        true
    ;
        inc_value(state_count, NewVal, _),
        NewVal < MaxVal
    ).

inspect_solve_one(SelectedGoal, RestGoals, Abducibles, Constraints, Denials, Info, NewInfo) :-
    ((get_value(depth_limit, DMax), DMax \== 0) ->
        Info = [Log, depth(D)],
        D =< DMax,
        NewD is D + 1,
        Info1 = [Log, depth(NewD)]
    ;
        Info1 = Info
    ),
    (get_value(dbg, true) ->
        increment_state_count,
        Info1 = [log([PID, OldComment]), Depth],
        inc_value(state_id, CID, _),
        trace_file(OutStream),
        open_node(OutStream, CID),
        dump_goals(OutStream, [SelectedGoal|RestGoals]),
        dump_abducibles(OutStream, Abducibles),
        dump_constraints(OutStream, Constraints),
        dump_denials(OutStream, Denials),
        close_node(OutStream),
        output_edge(OutStream, PID, CID, OldComment),
        SelectedGoal = (Type, _),
        goal_type(Type, TypeName),
        unwrap_literal(SelectedGoal, UwGoal),
        (UwGoal = fail(Vs, Body) ->
            NewComment = ['Select ', TypeName, ' goal: forall ', Vs, ' . fail ', Body]
        ;
            NewComment = ['Select ', TypeName, ' goal: ', UwGoal]
        ),
        Info2 = [log([CID, NewComment]), Depth]
    ;
        Info2 = Info1
    ),
  (get_value(stats, true) ->
    length(Denials, LenNs),
    length([SelectedGoal|RestGoals], LenGoals),
    length(Abducibles, LenDelta),
    denial_lengths(Denials, LenDenials),
    my_avg_std(LenDenials, AvgDenial, StdDenial),
    Info = [GoalSizes, DeltaSizes, DenialSizes, AvgDenials, StdDenials],
    NewInfo = [[LenGoals|GoalSizes],[LenDelta|DeltaSizes],[LenNs|DenialSizes], [AvgDenial|AvgDenials], [StdDenial|StdDenials]]
  ;
    NewInfo = Info2
  ).

inspect_fail_one(SelectedLiteral, UniversalVariables, RestDenial, RestGoals, Abducibles, Constraints, Denials, Info, NewInfo) :-
    ((get_value(depth_limit, DMax), DMax \== 0) ->
        Info = [Log, depth(D)],
        D =< DMax,
        NewD is D + 1,
        Info1 = [Log, depth(NewD)]
    ;
        Info1 = Info
    ),
    (get_value(dbg, true) ->
        increment_state_count,
        Info1 = [log([PID, OldComment]), Depth],
        inc_value(state_id, CID, _),
        trace_file(OutStream),
        open_node(OutStream, CID),
        dump_goals(OutStream, [(-, fail(UniversalVariables, [SelectedLiteral|RestDenial]))|RestGoals]),
        dump_abducibles(OutStream, Abducibles),
        dump_constraints(OutStream, Constraints),
        dump_denials(OutStream, Denials),
        close_node(OutStream),
        output_edge(OutStream, PID, CID, OldComment),
        SelectedLiteral = (Type, _),
        goal_type(Type, TypeName),
        unwrap_literal(SelectedLiteral, UwLit),
        maplist(unwrap_literal, RestDenial, UwRestDenial),
        NewComment = ['Select ', TypeName, ' literal ', UwLit, ' in denial goal: forall ', UniversalVariables, ' . fail ', [UwLit|UwRestDenial]],
        Info2 = [log([CID, NewComment]), Depth]
    ;
        Info2 = Info1
    ),
  (get_value(stats, true) ->
    length(Denials, LenNs),
    length([_|RestGoals], LenGoals),
    length(Abducibles, LenDelta),
    denial_lengths(Denials, LenDenials),
    my_avg_std(LenDenials, AvgDenial, StdDenial),
    Info = [GoalSizes, DeltaSizes, DenialSizes, AvgDenials, StdDenials],
    NewInfo = [[LenGoals|GoalSizes],[LenDelta|DeltaSizes],[LenNs|DenialSizes], [AvgDenial|AvgDenials], [StdDenial|StdDenials]]
  ;
    NewInfo = Info2
  ).

inspect_initial_state(Query, NewInfo) :-
    (get_value(dbg, true) ->
        reset_state_count,
        % 1. reset counter for generation of state ids
        set_value(state_id, 0),
        % 2. output new info
        NewLog = log([0, ['Start!']]),
        % 3 output state node
        trace_file(OutStream),
        open_node(OutStream, 0),
        writeln(OutStream, 'Query:'),
        writeln(OutStream, Query),
        close_node(OutStream)
    ;
        NewLog = log([])
    ),
    (get_value(stats, true) ->
    % Info looks like:
    % GoalSizes, DeltaSizes, DenialSizes, AvgDenials, StdDenials
    % AvgDenials is the average body length in the denial store
    NewInfo = [[], [], [], [], []]
  ;
    NewInfo = [NewLog, depth(0)]
  ).

inspect_successful_state(Abducibles, Constraints, Denials, Info, NewInfo) :-
    % ignore depth bound
    (get_value(dbg, true) ->
        Info = [log([PID, Comment]), _Depth],
        inc_value(state_id, CID, _),
        trace_file(OutStream),
        open_node(OutStream, CID),
        dump_abducibles(OutStream, Abducibles),
        dump_extracted_constraints(OutStream, Constraints),
        dump_unwrapped_denials(OutStream, Denials),
        close_node(OutStream),
        output_edge(OutStream, PID, CID, ['Succeeded :::: '|Comment])
    ;
        true
    ),
  (get_value(stats, true) ->
    length(Denials, LenNs),
    length(Abducibles, LenDelta),
    denial_lengths(Denials, LenDenials),
    my_avg_std(LenDenials, AvgDenial, StdDenial),
    Info = [GoalSizes, DeltaSizes, DenialSizes, AvgDenials, StdDenials],
    NewInfo = [[0|GoalSizes],[LenDelta|DeltaSizes],[LenNs|DenialSizes], [AvgDenial|AvgDenials], [StdDenial|StdDenials]]
  ;
    NewInfo = Info
  ).

inspect_post_process(Info, Deltas, NDeltas, NsOut, NewInfo) :-
  (get_value(stats, true) ->
    append(NDeltas, NsOut, Denials),
    length(Denials, LenNs),
    length(Deltas, LenDelta),
    denial_lengths(Denials, LenDenials),
    my_avg_std(LenDenials, AvgDenial, StdDenial),
    Info = [GoalSizes, DeltaSizes, DenialSizes, AvgDenials, StdDenials],
    NewInfo = [[0|GoalSizes],[LenDelta|DeltaSizes],[LenNs|DenialSizes], [AvgDenial|AvgDenials], [StdDenial|StdDenials]]
  ;
    true
  ).

denial_lengths(Denials, DenialLengths) :- denial_lengths(Denials, DenialLengths, []).
denial_lengths([], DenialLengths, DenialLengths) :- !.
denial_lengths([fail(_Vs,L)|T], DenialLengths, Acc) :-
  !,
  length(L, Len),
  denial_lengths(T, DenialLengths, [Len|Acc]).
% for the processed case
denial_lengths([L|T], DenialLengths, Acc) :-
  (is_list(L) ->
    length(L, Len)
  ;
    Len is 1
  ),
  denial_lengths(T, DenialLengths, [Len|Acc]).

my_avg_std(L, Avg, Std) :-
  my_avg_std(L, Avg, Std, 0, 0, 0).
my_avg_std([], Avg, Std, S1, S2, Len) :-
  (Len > 0 ->
    Avg is S1/Len,
    Std is S2/Len-Avg*Avg
  ;
    Avg is 0,
    Std is 0
  ).
my_avg_std([H|T], Avg, Std, S1Acc, S2Acc, LenAcc) :-
  S1Acc1 is S1Acc+H,
  S2Acc1 is S2Acc+H*H,
  LenAcc1 is LenAcc+1,
  my_avg_std(T, Avg, Std, S1Acc1, S2Acc1, LenAcc1).

% End Rares


dump_goals(OutStream, Goals) :-
    maplist(unwrap_literal, Goals, Gs),
    maplist(replace_special_chars, Gs, Gs1),
    (Gs1 \= [] ->
        writeln(OutStream, 'Goals:'),
        forall(member(X, Gs1), (
            X = fail(Vs, Body) ->
                write(OutStream, '  forall '), write(OutStream, Vs), write(OutStream, ' . fail '),
                write(OutStream, Body), nl(OutStream)
            ;
                write(OutStream, '  '), write(OutStream, X), nl(OutStream)
        ))
    ;
        true
    ).

dump_abducibles(OutStream, Abducibles) :-
    maplist(replace_special_chars, Abducibles, As),
    (As \= [] ->
        writeln(OutStream, 'Abducibles:'),
        forall(member(X, As), (
            write(OutStream, '  '), write(OutStream, X), nl(OutStream)
        ))
    ;
        true
    ).

dump_extracted_constraints(OutStream, Constraints) :-
    maplist(replace_special_chars, Constraints, Cs),
    (Cs \= [] ->
        writeln(OutStream, 'Constraints:'),
        forall(member(X, Cs), (
            write(OutStream, '  '), write(OutStream, X), nl(OutStream)
        ))
    ;
        true
    ).

dump_constraints(OutStream, (Es, Fs, Rs)) :-
    append(Fs, Rs, Cons1),
    selectlist(nonground, Cons1, Cons2),
    term_variables(Es, EsVs),
    maplist(inequalities, EsVs, Ess),
    my_flatten(Ess, Es1),
    list_to_ord_set(Es1, Es2),
    append(Es2, Cons2, Constraints),
    maplist(replace_special_chars, Constraints, Cs),
    (Cs \= [] ->
        writeln(OutStream, 'Constraints:'),
        forall(member(X, Cs), (
            write(OutStream, '  '), write(OutStream, X), nl(OutStream)
        ))
    ;
        true
    ).

dump_denials(OutStream, Denials) :-
    maplist(unwrap_denial, Denials, Ns),
    dump_unwrapped_denials(OutStream, Ns).

dump_unwrapped_denials(OutStream, Denials) :-
    maplist(replace_special_chars, Denials, Ns1),
    (Ns1 \= [] ->
        writeln(OutStream, 'Denials:'),
        forall(member(fail(Vs, Body), Ns1), (
            write(OutStream, '  forall '), write(OutStream, Vs), write(OutStream, ' . fail '),
            write(OutStream, Body), nl(OutStream)
        ))
    ;
        true
    ).

open_node(OutStream, StateID) :-
    ttwrite(OutStream, '<node id="n'), write(OutStream, StateID), write(OutStream, '">'), nl(OutStream),
  tttwriteln(OutStream, '<data key="d6">'),
    ttttwriteln(OutStream, '<y:GenericNode configuration="ShinyPlateNode3">'),
    tttttwriteln(OutStream, '<y:Fill color="#FF9900" transparent="false"/>'),
    tttttwriteln(OutStream, '<y:BorderStyle hasColor="false" type="line" width="1.0"/>'),
    tttttwriteln(OutStream, '<y:NodeLabel alignment="left" autoSizePolicy="content" fontFamily="Dialog" fontSize="2" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" modelName="internal" modelPosition="c" textColor="#000000" visible="true">').

close_node(OutStream) :-
    tttttwriteln(OutStream, '</y:NodeLabel>'),
    ttttwriteln(OutStream, '</y:GenericNode>'),
    tttwriteln(OutStream, '</data>'),
    ttwriteln(OutStream, '</node>').

output_edge(OutStream, ParentID, ChildID, Comment) :-
    ttwrite(OutStream, '<edge id="e'), write(OutStream, ChildID), write(OutStream, '" source="n'), write(OutStream, ParentID), write(OutStream, '" target="n'), write(OutStream, ChildID), write(OutStream, '">'), nl(OutStream),
    tttwriteln(OutStream, '<data key="d9">'),
    ttttwriteln(OutStream, '<y:PolyLineEdge>'),
    tttttwriteln(OutStream, '<y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>'),
    tttttwriteln(OutStream, '<y:LineStyle color="#000000" type="line" width="1.0"/>'),
    tttttwriteln(OutStream, '<y:Arrows source="none" target="standard"/>'),
    tttttwriteln(OutStream, '<y:EdgeLabel alignment="center" distance="2.0" fontFamily="Dialog" fontSize="2" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" modelName="six_pos" modelPosition="tail" preferredPlacement="anywhere" ratio="0.5" textColor="#000000" visible="true">'),
    forall(member(C, Comment), (
        write(OutStream, C)
    )), nl(OutStream),
    tttttwriteln(OutStream, '</y:EdgeLabel>'),
    tttttwriteln(OutStream, '<y:BendStyle smoothed="false"/>'),
    ttttwriteln(OutStream, '</y:PolyLineEdge>'),
    tttwriteln(OutStream, '</data>'),
    ttwriteln(OutStream, '</edge>').

initialise_trace(TraceFile) :-
    % open stream to write to TraceFile
    open(TraceFile, write, OutStream),
    retractall(trace_file(_)),
    assertz(trace_file(OutStream)),
    % generate header
    writeln(OutStream, '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'),
    writeln(OutStream, '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"'),
    twriteln(OutStream, 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:y="http://www.yworks.com/xml/graphml"'),
    twriteln(OutStream, 'xmlns:yed="http://www.yworks.com/xml/yed/3"'),
    twriteln(OutStream, 'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">'),
    twriteln(OutStream, '<key for="graphml" id="d0" yfiles.type="resources" />'),
    twriteln(OutStream, '<key for="port" id="d1" yfiles.type="portgraphics" />'),
    twriteln(OutStream, '<key for="port" id="d2" yfiles.type="portgeometry" />'),
    twriteln(OutStream, '<key for="port" id="d3" yfiles.type="portuserdata" />'),
    twriteln(OutStream, '<key attr.name="url" attr.type="string" for="node" id="d4" />'),
    twriteln(OutStream, '<key attr.name="description" attr.type="string" for="node" id="d5" />'),
    twriteln(OutStream, '<key for="node" id="d6" yfiles.type="nodegraphics" />'),
    twriteln(OutStream, '<key attr.name="url" attr.type="string" for="edge" id="d7" />'),
    twriteln(OutStream, '<key attr.name="description" attr.type="string" for="edge" id="d8" />'),
    twriteln(OutStream, '<key for="edge" id="d9" yfiles.type="edgegraphics" />'),
    twriteln(OutStream, '<graph edgedefault="directed" id="G">').

finalise_trace :-
    retract(trace_file(OutStream)),
    % generate footer
    twriteln(OutStream, '</graph>'),
    twriteln(OutStream, '<data key="d0">'),
    ttwriteln(OutStream, '<y:Resources />'),
    twriteln(OutStream, '</data>'),
    writeln(OutStream, '</graphml>'),
    % close stream
    flush_output(OutStream),
    close(OutStream).

twrite(S, X) :- write(S, '\t'), write(S, X).
ttwrite(S, X) :- write(S, '\t\t'), write(S, X).
tttwrite(S, X) :- write(S, '\t\t\t'), write(S, X).

writeln(S, X) :- write(S, X), nl(S).
twriteln(S,X) :- write(S, '\t'), write(S,X), nl(S).
ttwriteln(S,X) :- write(S, '\t\t'), write(S,X), nl(S).
tttwriteln(S,X) :- write(S, '\t\t\t'), write(S,X), nl(S).
ttttwriteln(S,X) :- write(S, '\t\t\t\t'), write(S,X), nl(S).
tttttwriteln(S,X) :- write(S, '\t\t\t\t\t'), write(S,X), nl(S).
ttttttwriteln(S,X) :- write(S, '\t\t\t\t\t\t'), write(S,X), nl(S).

writelist(_, []) :- !.
writelist(S, [H|T]) :-
    write(S, H), writelist(S, T).

% --- for yEd ---
replace_special_chars(X, X) :-
    var(X), !.
replace_special_chars(X, Y) :-
    atomic(X), !,
    replace_special_atom(X, Y).
replace_special_chars(X, Y) :-
    % compound(X),
    X =.. [F|Args],
    replace_special_chars(F, F1),
    replace_special_chars_for_all(Args, Args1),
    Y =.. [F1|Args1].

replace_special_chars_for_all([], []).
replace_special_chars_for_all([H|T], [H1|T1]) :-
    replace_special_chars(H, H1),
    replace_special_chars_for_all(T, T1).

replace_special_atom(<, '&lt;') :- !.
replace_special_atom(=<, '=&lt;') :- !.
replace_special_atom(>, '&gt;') :- !.
replace_special_atom(>=, '&gt;=') :- !.
replace_special_atom(#<, '#&lt;') :- !.
replace_special_atom(#>, '#&gt;') :- !.
replace_special_atom(#=<, '#=&lt;') :- !.
replace_special_atom(#>=, '#&gt;=') :- !.
replace_special_atom(#<=>, '#&lt;=&gt;') :- !.
replace_special_atom(#<==>, '#&lt;==&gt;') :- !.
% replace_special_atom(\+, '\+') :- !.
% replace_special_atom(#\, '#\') :- !.
% replace_special_atom(#\=, '#\=') :- !.
% replace_special_atom(=\=, '=\=') :- !.
% replace_special_atom(#/\, '#/\') :- !.
% replace_special_atom(#\/, '#\/') :- !.
replace_special_atom(X, X).

goal_type(a, abducible).
goal_type(b, builtin).
goal_type(d, 'non-abducible').
goal_type(e, equality).
goal_type(i, inequality).
goal_type(n, negative).
goal_type(f, 'finite domain constraint').
goal_type(r, 'real domain constraint').
goal_type(-, 'denial').
