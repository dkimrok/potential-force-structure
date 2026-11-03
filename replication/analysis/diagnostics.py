"""Further checks reported in the text (run after tables.py).
(a) N3 share-zero point on a 0.01 grid; (b) N0 perturbations (lambda = 2, sigma_war = 0.6) along gamma;
(c) gamma = 0 endpoint: solver solution vs. lexicographic minimum of transition cost at the same objective;
(d) path of the share between N2 and N3 at theta = 2;
(e) an alternative monotone profile (each function's requirements sorted ascending) with technical holding cost 1.4;
(f) Table 6 rows when 1986 is coded as Peacetime.
Usage: python diagnostics.py [base|alt]"""
import sys, json, pathlib
import numpy as np, pulp
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'model'))
import potential_structure_lp as M
TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
FM = {'base': M.F, 'alt': M.F_ALT}[TAG]
K, J, PAIRS = M.K, M.J, M.PAIRS
N0, N2, N3 = 365000, 289000, 219000
G = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5, 8]
R = {}

def S(N, th, g=2.0, **kw):
    kw.setdefault('Fmat', FM); return M.solve(N, g, rho_scale=th, **kw)

# (a) share-zero point at N3, gamma = 2
zs = None
for t in np.arange(7.50, 8.201, 0.01):
    if S(N3, float(t))['k5sh'] < 1e-6: zs = round(float(t), 2); break
R['N3_share_zero_theta'] = zs

# (b) perturbations at N0
def sig_war(v): return {0: .3, 1: .5, 2: .8, 3: v}
out = {}
for lab, kw in (('lambda2', dict(lam=2.0)), ('sigma_war_0.6', dict(sigmamap=sig_war(0.6)))):
    for th in (2.0, 8.0):
        out[f'{lab}|{th}'] = [(g, round(s['SS'] / 1000, 1), round(s['TC'] / 1000, 1), round(s['SHORT'] / 1000, 2))
                              for g in G for s in [S(N0, th, g, **kw)]]
R['N0_perturb'] = out

# (c) gamma = 0 endpoint
def lex(N, th):
    W, PI = M._WPI(FM)
    def build():
        m = pulp.LpProblem('L', pulp.LpMinimize)
        n = {k: pulp.LpVariable(f'n_{k}', 0) for k in K}
        x = {(k, j, s): pulp.LpVariable(f'x_{k}_{j}_{s}', 0) for k in K for j in J for s in range(4)}
        dp = {(k, j, p): pulp.LpVariable(f'dp_{k}_{j}_{p}', 0) for k in K for j in J for p in PAIRS}
        dm = {(k, j, p): pulp.LpVariable(f'dm_{k}_{j}_{p}', 0) for k in K for j in J for p in PAIRS}
        u = {(j, s): pulp.LpVariable(f'u_{j}_{s}', 0) for j in J for s in range(4)}
        base = (pulp.lpSum(M.h[k] * n[k] for k in K) + pulp.lpSum(M.LAMBDA * PI[s] * x[k, j, s] for k in K for j in J for s in range(4))
                + pulp.lpSum(th * M.SIGMA[s] * M.rho[j] * u[j, s] for j in J for s in range(4)))
        tc = pulp.lpSum(W[p] * M.psi[k] * M.phi[j] * (dp[k, j, p] + dm[k, j, p]) for k in K for j in J for p in PAIRS)
        for j in J:
            for s in range(4):
                m += pulp.lpSum(M.A[k][j] * x[k, j, s] for k in K) + u[j, s] >= M.r[j][s] * N + M.o[j][s]
        for k in K:
            for s in range(4):
                m += pulp.lpSum(x[k, j, s] for j in J) <= n[k]
        m += pulp.lpSum(n.values()) <= N
        for k in K:
            for j in J:
                for (a, b) in PAIRS:
                    m += x[k, j, a] - x[k, j, b] == dp[k, j, (a, b)] - dm[k, j, (a, b)]
        return m, base, tc, n, x, u
    m, base, tc, n, x, u = build(); m += base; m.solve(pulp.PULP_CBC_CMD(msg=0)); b0 = pulp.value(base)
    m2, base2, tc2, n2, x2, u2 = build(); m2 += tc2; m2 += base2 <= b0 + 1e-3; m2.solve(pulp.PULP_CBC_CMD(msg=0))
    ss = sum(M.h[k] * n2[k].value() for k in K) + sum(M.LAMBDA * PI[s] * x2[k, j, s].value() for k in K for j in J for s in range(4))
    return dict(SS=round(ss / 1000, 3), TC=round(pulp.value(tc2) / 1000, 3), gap=round(sum(v.value() for v in u2.values()) / 1000, 2),
                k5=round(100 * n2['k5_전문기술'].value() / sum(v.value() for v in n2.values()), 2))
R['gamma0'] = {th: dict(solver={k: round(v, 3) for k, v in (('SS', S(N0, th, 0)['SS'] / 1000), ('TC', S(N0, th, 0)['TC'] / 1000), ('k5', S(N0, th, 0)['k5sh']))},
                        lexicographic=lex(N0, th)) for th in (2.0, 8.0)}

# (d) share path between N2 and N3 at theta = 2 and 8
R['path_share'] = {th: [(N // 1000, round(S(N, th)['k5sh'], 2), round(S(N, th)['SHORT'] / 1000, 1)) for N in range(289000, 214999, -5000)] for th in (2.0, 8.0)}
# (e) alternative monotone profile with a lower technical holding cost
import copy
rs = copy.deepcopy(M.r); os_ = copy.deepcopy(M.o)
for j in J: rs[j] = sorted(rs[j]); os_[j] = sorted(os_[j])
R['sorted_monotone_h14_N0'] = {th: (round(s['k5sh'], 3), round(s['SHORT'] / 1000, 2)) for th in (2.0, 8.0)
                               for s in [S(N0, th, rmat=rs, ovec=os_, hvec={**M.h, 'k5_전문기술': 1.4})]}
# (f) 1986 coded as Peacetime
code = json.load(open(HERE.parents[0] / 'data' / ('posture_coding.json' if TAG == 'base' else 'posture_coding_alt.json'))); code['1986'] = 1
ys = sorted(int(k) for k in code); C = np.zeros((3, 3))
for a, b in zip(ys, ys[1:]): C[code[str(a)] - 1, code[str(b)] - 1] += 1
F86 = M.build_F(C)
R['table6_1986P'] = {th: [(N // 1000, round(s['TC'] / 1000, 1), round(s['CI'] / 1000), round(s['SHORT'] / 1000, 1), round(s['k5sh'], 2))
                          for N in (365000, 348000, 289000, 219000, 197100) for s in [M.solve(N, 2.0, rho_scale=th, Fmat=F86)]] for th in (2.0, 8.0)}
json.dump(R, open(HERE / ('diagnostics.json' if TAG == 'base' else f'diagnostics_{TAG}.json'), 'w'), indent=1)
print(json.dumps(R, indent=0)[:6000])
