# Solver step 25: recovered historical focal verifier has an opened pair signal

A previously missing historical checkpoint was recovered from the local Kaggle
audit cache: `verify_pair_best.pt`, SHA-256
`3bcc89a12e7b539304484b441688b4b9fb1c3711e918befed9cdef7c17f776e7`.
It is a 200,838-parameter joint seam CNN trained with focal BCE.  At inference
it reads only the two raw dirty boundary strips plus six scores derived from
the current TASKA cost row, and reorders the existing harvest without adding
or deleting edges.

Using the historical repository-tip top-8 scalar contract on opened32 produced
**337.03125 pairs**, recall **0.305281929**, and **3.75 exact tiles**, versus
334.71875 / 0.303187274 / 4.46875 for raw ordering.  Pair delta was +2.3125
with clustered 95% interval `[-0.9070, +6.0]`; exact delta was -0.71875.  All
32 layouts were strict.

This is a genuine pair signal but not yet a promotion: the interval crosses
zero, exact regresses, and the focal training code used top-5 row statistics
while the historical inference tip used top-8.  Both audited fixed contracts
must be separated and replayed unchanged on held300 before choosing one.
