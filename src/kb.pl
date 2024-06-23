% individui
:- dynamic(area/1).

% proprietà individui
:- dynamic(severity/2).
:- dynamic(nearAreas/2).
:- dynamic(size/2).

:- dynamic(patrolArea/2).


% vicinanza
isNear(area(A1), area(A2)) :- nearAreas(area(A1), NearAreasList), member(A2, NearAreasList).

% severity aggiustata
minSize(S) :- findall(X, size(_, X), L), min_list(L, S).
maxSize(S) :- findall(X, size(_, X), L), max_list(L, S).

adjustedSeverity(area(A), Sev) :- severity(area(A), Sev1), Sev1 =< 1, size(area(A), Size), minSize(Min), maxSize(Max), Size > (Min + Max) / 2, Sev is Sev1 + 1.
adjustedSeverity(area(A), Sev) :- severity(area(A), Sev1), Sev1 is 2, Sev is Sev1.
adjustedSeverity(area(A), Sev) :- severity(area(A), Sev1), size(area(A), Size), minSize(Min), maxSize(Max), Size =< (Min + Max) / 2, Sev is Sev1.

% distanza
distance(area(A1), area(A2), 0) :-
    A1 = A2.

distance(area(A1), area(A2), 1) :-
    isNear(area(A1), area(A2)).

distance(area(A1), area(A2), 2) :-
    isNear(area(A1), area(X)),
    isNear(area(X), area(A2)),
    not(isNear(area(A1), area(A2))),
    not(distance(area(A1), area(A2), 0)).


% massima distanza
maxDistance(area(A1), area(A2), D) :-
    distance(area(A1), area(A2), D1),
    D1 =< D.

% area valutabile
isConsiderable(area(A)) :-
    adjustedSeverity(area(A), S),
    forall(maxDistance(area(A), area(D), 2 - S), patrolArea(area(D), _)).

% area sicura
isSafe(area(A)) :-
    adjustedSeverity(area(A), S),
    maxDistance(area(A), area(AreaP), 2 - S),
    patrolArea(area(AreaP), true).