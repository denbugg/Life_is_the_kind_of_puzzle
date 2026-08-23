import inspect
import eval_r2l_affinity_union as u
print("--- UNION ---")
print(inspect.getsource(u._union_candidates))
print("--- MAIN ---")
print(inspect.getsource(u.main))
