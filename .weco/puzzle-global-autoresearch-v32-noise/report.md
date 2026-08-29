# V32 interim report

V32 is running on the RTX 4060.  It adds an exact, deterministic implementation
of the requested per-tile corruption contract, opaque random filenames and
manifests, three newly generated real-data examples, paired clean/noisy fused
score caches, and a 1,000,924-parameter full-board spatial critic.

The critic consumes 32 planes at 24x24, predicts right/down/cell correctness and
a global board score, and is trained with within-scene ranking, adjacency/local
supervision and clean-teacher/noisy-student consistency.  It never emits a board
directly: it reranks permutation-safe solver candidates and its local error map
can guide LNS.

The exact corruption smoke is materially difficult: on scene 6700 Top-1 neighbor
retrieval fell from `0.4529` clean to `0.2391` and `0.2264` on two replicas.
This confirms that noisy training targets the actual weakness rather than adding
cosmetic augmentation.
