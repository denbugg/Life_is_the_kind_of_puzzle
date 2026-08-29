# Program

V31 is an isolated research harness around the V30/V29 solver artifacts. It will
cache fused score matrices and candidate boards, expose deterministic structural
objectives and repair operators, assert permutation validity after every accepted
move, and emit JSON reports for validation and fixed-development evaluation.

The production solver is not changed until a V31 candidate passes the declared
validation gate.

