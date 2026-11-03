"""N0 one-at-a-time perturbations along gamma (baseline coding): does steady-state cost rise as transition cost falls?"""
import sys, json, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'model'))
import potential_structure_lp as M
N0 = 365000; G = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5, 8]
mphi = np.mean(list(M.phi.values()))
phid = lambda l: {j: max(0.1, mphi + l * (M.phi[j] - mphi)) for j in M.J}
psis = lambda f: {k: M.psi[k] * f for k in M.K}
def Fvol(b):
    X = M.F.copy().astype(float)
    for i in range(4):
        off = X[i].copy(); off[i] = 0; X[i] = b * off; X[i, i] = max(0, 1 - (b * off).sum())
    return X
P = [('lam', v, {'lam': v}) for v in (0.5, 0.75, 1.25, 1.5, 2.0)] + [('phid', v, {'phivec': phid(v)}) for v in (0, 0.5, 1.5, 2)] \
  + [('psi', v, {'psivec': psis(v)}) for v in (0.7, 0.85, 1.15, 1.3)] + [('Fvol', v, {'Fmat': Fvol(v)}) for v in (0.6, 0.8, 1.2, 1.4)] \
  + [('sigma_war', 0.6, {'sigmamap': {0: .3, 1: .5, 2: .8, 3: 0.6}})]
out = {}
for th in (2.0, 8.0):
    for lab, v, kw in P:
        rows = [(g, s['SS'] / 1000, s['TC'] / 1000, s['SHORT'] / 1000) for g in G for s in [M.solve(N0, g, rho_scale=th, **kw)]]
        both_fall = [(a[0], b[0]) for a, b in zip(rows, rows[1:]) if b[1] < a[1] - 0.05 and b[2] < a[2] - 0.05]
        out[f'{th}|{lab}|{v}'] = dict(premium=round(rows[-1][1] - rows[0][1], 1), maxgap=round(max(r[3] for r in rows), 2),
                                      both_fall=both_fall, both_fall_nogap=[p for p in both_fall if max(r[3] for r in rows if r[0] in p) < 1e-3])
json.dump(out, open(HERE / 'perturb.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
