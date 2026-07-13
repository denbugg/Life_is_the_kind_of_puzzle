# Candidate-graph oracle v1: invalid, no result

Date: 2026-07-12

Protocol instance: `2f1f4192e57d39cb1626e7f1eb2ae292`

Final pinned config SHA-256: `6aec4736747ec5350a5bfb27d9f0b1d688e0522e1d74c603afd3e25d201e12f5`

Disposition: `INVALID_NO_RESULT`. This run produced no oracle metrics and must not be interpreted as a graph-ceiling failure or pass.

## Failure

Both Kaggle Phase-A shards stopped before deriving candidate-graph results with:

`RuntimeError: opaque fixture array dtype/shape drift`

The fixture builder stored the `qap_seed` NPZ array correctly as a scalar `uint64`, but `_array_descriptor()` called `numpy.ascontiguousarray()` on the scalar. NumPy promoted the descriptor view to shape `[1]`, so the public input manifest recorded `[1]` while the frozen evaluator and semantic contract require scalar shape `[]`.

The split unit tests covered the builder and evaluator separately but did not run a real builder-to-evaluator manifest compatibility check.

## Evidence

- Input manifest SHA-256: `64a53b170e104ab1e383489259c19e008769fc2b859e800bcebaddeeac247d89`
- Rank-0 failure log SHA-256: `71882dac54eeaf98287bc5d96c3ed566dde95cc7872a53ec6a7778c35cbc3c27`
- Shard invalid-result envelope SHA-256: `e9446c88518b60a2c7db97c00c7d7459a322ae0c32f4e1c62ee14f6ef059bf49`
- Kaggle Phase-A unit-test log SHA-256: `593e67e46d25204a7d8456e681f54adeefc421cdde86307796818cefcb0eaca1` (`17 passed`)

## Blinding state

The persistent lifecycle contains `PREP`, `SEALED`, and `PHASE_A`. It does not contain `LABEL_ACCESS`. The local label root and label records were not opened, listed, hashed, joined, or passed to Phase B after fixture creation. No target-dependent metric was computed.

This protocol instance is retired. Do not modify its pinned config, manifests, fixture roots, lifecycle records, Kaggle archives, or downloaded failure evidence. A replacement instance must use fresh output roots and a new protocol ID, add a real fixture-builder-to-evaluator integration test, and separately account for Kaggle's private-kernel versioned-readback behavior.
