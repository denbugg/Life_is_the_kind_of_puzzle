$ErrorActionPreference = 'Stop'
$p = 'E:\pazzle_work\pazzle_fixed_orientation_20260813\worktrees\cb1_boundary_buddies\src\p8_context_candidate_graph.py'
$s = [IO.File]::ReadAllText($p)
$old = @'
  for j,m in enumerate(members[q]):
   pos=np.flatnonzero(cn[int(a),int(d)]==m)
   if len(pos)!=1:raise RuntimeError('P8 member score absent/duplicate')
   base[q,j]=sc[int(a),int(d),int(pos[0])]
'@
$new = @'
  for j,m in enumerate(members[q]):
   pos=np.flatnonzero(cn[int(a),int(d)]==m)
   if len(pos)==1:
    base[q,j]=sc[int(a),int(d),int(pos[0])]
   elif int(m)==int(truth):
    # P3 hardlists legally injects an absent true neighbour; rank66 has no
    # score for it, so represent its frozen baseline as strictly worst.
    base[q,j]=-1.0e9
   else:
    raise RuntimeError('P8 unexpected non-rank96 non-target member')
'@
if (-not $s.Contains($old)) { throw 'expected baseline block not found' }
[IO.File]::WriteAllText($p, $s.Replace($old, $new), [Text.UTF8Encoding]::new($false))
Write-Output 'P8 baseline patch applied'
