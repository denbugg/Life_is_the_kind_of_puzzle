## Evidence
- Growing Consensus reports that geometric grid/loop consensus reduces dependence on noisy pairwise scores and reduced assembly error substantially: https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html
- Greedy Asymmetric Block Construction uses two-side verification and disjoint blocks to suppress false-positive single-edge matches: https://www.jstage.jst.go.jp/article/transfun/E109.A/2/E109.A_2025EAP1018/_article/-char/en
- Block-to-block assembly expands the move space beyond attaching one piece to one block: https://doi.org/10.1109/ICPR.2008.4761067
- Successive LP relaxations exploit all pairwise matches and globally position components: https://arxiv.org/abs/1511.04472

## Practical translation
- Preserve locally coherent rectangles during relocation.
- Prefer moves supported by at least two boundary edges.
- Use component/block moves as a second phase after the current strong Hungarian+SA initializer.
