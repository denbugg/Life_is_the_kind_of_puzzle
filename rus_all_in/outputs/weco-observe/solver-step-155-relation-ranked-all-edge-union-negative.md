# Solver step 155: relation-ranked all-edge union is strongly negative

Parent: confirmed relation selector step 143. Fixed contract SHA-256:
`2f0beb7fc071f4aef673267fab348baaeccdb664048e2ac2154edf45fa2723a7`.

The frozen HGB scored all `6×1,104` realised relation occurrences; duplicate
directed edges retained maximum probability with stable arm-order ties; all
unique edges were passed in descending probability order to unchanged raw-tail
global. No threshold, top-k, weight or parameter sweep was allowed.

On HGB-in-sample mechanical local32, pairs fell `326.750→199.500`, delta
`-127.250`, source-CI95 `[-145.625,-108.530]`, W/T/L `1/0/31`. Adjacency fell
`29.597%→18.071%`; exact `5.688→2.219`. All layouts were strict.

Preregistered local gate failed. Held step 156, fresh step 157 and formal step
158 were not created; no new roster was generated or scored. Report SHA-256:
`cee46746aac04e19e4c008d46092611cdd0cc32dfbd1eea6951eab9988d4cca4`.
