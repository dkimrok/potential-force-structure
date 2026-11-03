"""Coding sensitivity: a stricter rule that counts as Limited conflict only the eleven years of major incidents
(1967-1969, 1974, 1976, 1983, 1987, 1996, 1999, 2002, 2010); all other years are coded by the Tension rule
from the counts in the posture ledger. Also reports the deep-decline results across war occupancies."""
import sys, json, csv, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'model'))
import potential_structure_lp as M
led = list(csv.DictReader(open(HERE.parents[0] / 'data' / 'posture_ledger.csv')))
MAJOR = {1967, 1968, 1969, 1974, 1976, 1983, 1987, 1996, 1999, 2002, 2010}
code = {}
for r in led:
    y = int(r['year']); nn, nb, ne = int(r['nuclear_tests']), int(r['ballistic_launches']), int(r['csis_events'])
    code[y] = 3 if y in MAJOR else (2 if (nn > 0 or nb >= 3 or ne >= 5) else 1)
ys = sorted(code); C = np.zeros((3, 3))
for a, b in zip(ys[:-1], ys[1:]): C[code[a] - 1, code[b] - 1] += 1
Fs = M.build_F(C)
w, v = np.linalg.eig(Fs.T); p = np.real(v[:, np.argmin(abs(w - 1))]); p /= p.sum()
out = dict(counts=C.astype(int).tolist(), F=np.round(Fs, 3).tolist(), pi=np.round(p, 3).tolist(),
           years={k: sum(1 for c in code.values() if c == i) for k, i in (('P', 1), ('T', 2), ('L', 3))})
def path(F):
    res = {}
    for th in (2.0, 8.0):
        ent = risk = None
        for N in range(400000, 179999, -5000):
            s = M.solve(N, 2.0, rho_scale=th, Fmat=F)
            if ent is None and s['k5sh'] > 0.05: ent = N // 1000
            if risk is None and s['SHORT_s4'] > 1: risk = N // 1000
        rows = []
        for N in (365000, 289000, 219000, 197100):
            s = M.solve(N, 2.0, rho_scale=th, Fmat=F)
            rows.append((N / 1000, round(s['k5sh'], 2), round(s['SHORT'] / 1000, 1), round(s['SHORT_s4'] / 1000, 1), round(s['TC'] / 1000, 1)))
        res[th] = dict(entry=ent, risk=risk, path=rows)
    return res
out['strict'] = path(Fs)
# deep-decline results across war occupancies (baseline coding)
occ = []
for lw, ww in ((0.02, 0.6667), (0.035, 0.6667), (0.06, 0.75), (0.08, 0.80)):
    F2 = M.F.copy(); obs = M.F[2, :3] / M.F[2, :3].sum()
    F2[2, :3] = obs * (1 - lw); F2[2, 3] = lw; F2[3] = [0, 0, 1 - ww, ww]
    row = dict(alpha=lw, persist=ww)
    for th in (2.0, 8.0):
        a = M.solve(219000, 2.0, rho_scale=th, Fmat=F2); b = M.solve(197100, 2.0, rho_scale=th, Fmat=F2)
        row[th] = (round(a['k5sh'], 2), round(b['k5sh'], 2), round(a['TC'] / 1000, 1))
    occ.append(row)
out['occupancy_deep'] = occ
json.dump(json.loads(json.dumps(out, default=str)), open(HERE / 'coding_sensitivity.json', 'w'), indent=1)
print(json.dumps(out, default=str, indent=0))
