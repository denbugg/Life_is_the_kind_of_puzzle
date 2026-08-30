# V33 task

Replace the failed V32 spatial-CNN board selector with a transformer reranker
over all 576 board cells. Reuse the completed paired clean/noisy score cache and
the 60-scene spatial tensor cache. The transformer must improve group-disjoint
board selection, remain permutation-safe, fit an RTX 4060 with 8 GB VRAM, and
be rejected if it cannot beat the handcrafted baseline selector.
