# Experiment ledger

| exp_id | angle | change | status | metric | delta | verified |
|---|---|---|---|---:|---:|---|
| 0 | - | handcrafted V32 baseline selector | passed | 0.3134581 OOF | 0 | yes |
| T01 | C | Transformer-S global reranker, 3.11M | dropped | 0.3134581 OOF | +0.0000000 | yes |
| T02/T03 | C/E | Transformer-M + relative bias/residual ranking, 8.77M | dropped | 0.3143464 OOF | +0.0008884 | yes |
| T04 | B/D | Transformer-MC clean/noisy consistency, 8.77M | dropped | 0.3138413 OOF | +0.0003832 | yes |

Locked validation: baseline `0.3776042`; T-S fallback `0.3776042`, T-M
`0.3716033`, T-MC `0.3766984`. None passes the promotion gates.
