"""Further numbers reported in the text (run after tables.py).

Usage: python supplement.py [base|alt|orig]
Writes supplement.json (base) or supplement_<tag>.json next to this script.
"""
import sys, json, copy, pathlib, itertools
import numpy as np, pulp
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'model'))
import potential_structure_lp as M

TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
FM = {'base': M.F, 'alt': M.F_ALT, 'orig': M.F_ORIG}[TAG]
K, J = M.K, M.J
N0, N1, N2, N3, N4 = 365000, 348000, 289000, 219000, 197100
r1 = lambda v, d=1: round(float(v), d)
R = {}

def S(N, th, g=2.0, **kw):
    kw.setdefault('Fmat', FM)
    return M.solve(N, g, rho_scale=th, **kw)


# (a) technical-type share above the path
R['k5_above'] = {N // 1000: r1(S(N, 2.0)['k5sh'], 2) for N in (600000, 500000, 450000, 420000, 400000)}

# (b) composition of the last gap just below the minimum weight, and of the first gaps on a 1,000 grid
def solve_u(N, th, g=2.0, **kw):
    """same LP as M.solve but returns u[j,s]"""
    Fm = kw.pop('Fmat', FM); sig = kw.pop('sigmamap', M.SIGMA)
    rm = kw.pop('rmat', M.r); om = kw.pop('ovec', M.o)
    W, PI = M._WPI(Fm)
    m = pulp.LpProblem('PSu', pulp.LpMinimize)
    n = {k: pulp.LpVariable(f'n_{k}', 0) for k in K}
    x = {(k, j, s): pulp.LpVariable(f'x_{k}_{j}_{s}', 0) for k in K for j in J for s in range(4)}
    dp = {(k, j, p): pulp.LpVariable(f'dp_{k}_{j}_{p}', 0) for k in K for j in J for p in M.PAIRS}
    dm = {(k, j, p): pulp.LpVariable(f'dm_{k}_{j}_{p}', 0) for k in K for j in J for p in M.PAIRS}
    u = {(j, s): pulp.LpVariable(f'u_{j}_{s}', 0) for j in J for s in range(4)}
    m += (pulp.lpSum(M.h[k] * n[k] for k in K) + pulp.lpSum(M.LAMBDA * PI[s] * x[k, j, s] for k in K for j in J for s in range(4))
          + pulp.lpSum(th * sig[s] * M.rho[j] * u[j, s] for j in J for s in range(4))
          + g * pulp.lpSum(W[p] * M.psi[k] * M.phi[j] * (dp[k, j, p] + dm[k, j, p]) for k in K for j in J for p in M.PAIRS))
    for j in J:
        for s in range(4):
            m += pulp.lpSum(M.A[k][j] * x[k, j, s] for k in K) + u[j, s] >= rm[j][s] * N + om[j][s]
    for k in K:
        for s in range(4):
            m += pulp.lpSum(x[k, j, s] for j in J) <= n[k]
    m += pulp.lpSum(n.values()) <= N
    for k in K:
        for j in J:
            for (a, b) in M.PAIRS:
                m += x[k, j, a] - x[k, j, b] == dp[k, j, (a, b)] - dm[k, j, (a, b)]
    if 'k5_min' in kw:
        m += n['k5_전문기술'] >= kw['k5_min'] * pulp.lpSum(n.values())
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    TC = sum(W[p] * M.psi[k] * M.phi[j] * (dp[k, j, p].value() + dm[k, j, p].value()) for k in K for j in J for p in M.PAIRS)
    return {f'{j}|{s}': round(u[j, s].value() / 1000, 2) for j in J for s in range(4) if u[j, s].value() > 1}, TC / 1000

th_min = json.load(open(HERE / ('results.json' if TAG == 'base' else f'results_{TAG}.json')))['theta_min']['theta_min']
R['gap_below_theta_min'] = solve_u(N0, th_min - 0.01)[0]
first = {}
for th in (2.0, 8.0):
    for N in range(275000, 239999, -1000):
        g, _ = solve_u(N, th)
        if g:
            first[th] = dict(N=N // 1000, gaps=g); break
R['first_gap_1k'] = first

# (c) Table 5 at theta = 8
R['T5_theta8'] = {g: [r1(v / 1000) for v in S(N0, 8.0, g)['X']['j3_기동']] for g in (0, 2)}

# (d) finer theta sweep at N3 and N4
R['theta_fine_N3'] = [(round(t, 2), r1(S(N3, float(t))['k5sh'], 2), r1(S(N3, float(t))['TC'] / 1000))
                      for t in np.arange(4.0, 8.01, 0.25)]
R['TC_N2'] = {th: r1(S(N2, th)['TC'] / 1000) for th in (2.0, 4.0, 5.0, 6.0, 8.0)}

# (e) forced 2 percent technical share at N3, theta = 8
R['forced2_N3_theta8'] = round(solve_u(N3, 8.0, k5_min=0.02)[1], 1)
R['free_N3_theta8_TC'] = r1(S(N3, 8.0)['TC'] / 1000)

# (f) Appendix A: risk effect averaged over the other two factors, by cross-training level (upper level 8)
mphi = np.mean(list(M.phi.values()))
phid = lambda l: {j: max(0.1, mphi + l * (M.phi[j] - mphi)) for j in J}
psik5 = lambda v: {**M.psi, 'k5_전문기술': v}
hk5 = lambda v: {**M.h, 'k5_전문기술': v}
runs = {}
for sg in itertools.product([-1, 1], repeat=4):
    runs[sg] = M.solve(N3, 2.0, Fmat=FM, rho_scale=(2.0 if sg[0] < 0 else 8.0), psivec=psik5(0.6 if sg[1] < 0 else 0.2),
                       phivec=phid(0.0 if sg[2] < 0 else 2.0), hvec=hk5(2.2 if sg[3] < 0 else 1.4))['k5sh']
eff = {}
for ct in (-1, 1):
    hi = np.mean([v for s, v in runs.items() if s[0] > 0 and s[1] == ct])
    lo = np.mean([v for s, v in runs.items() if s[0] < 0 and s[1] == ct])
    eff['CT_high' if ct > 0 else 'CT_low'] = r1(hi - lo, 2)
eff['interaction_half_difference'] = r1((eff['CT_high'] - eff['CT_low']) / 2, 2)
R['A1_avg'] = eff

# (g) war-only weighting: N2 transition cost and the theta_war at which N3 transition cost exceeds it
def war_sig(tw): return {0: .3, 1: .5, 2: .8, 3: 1.0 * tw / 2}
R['war_only_N2_TC'] = {tw: r1(S(N2, 2.0, sigmamap=war_sig(tw))['TC'] / 1000) for tw in (2, 8)}
R['war_only_fine_N3'] = [(tw, r1(S(N3, 2.0, sigmamap=war_sig(tw))['k5sh'], 2), r1(S(N3, 2.0, sigmamap=war_sig(tw))['TC'] / 1000))
                         for tw in (5, 6, 6.5, 7, 7.5, 8, 10, 12, 16)]
R['war_only_deep_theta8'] = [(N / 1000, r1(S(N, 2.0, sigmamap=war_sig(8))['k5sh'], 2), r1(S(N, 2.0, sigmamap=war_sig(8))['SHORT_s4'] / 1000))
                             for N in (289000, 260000, 250000, 240000, 230000, 219000, 210000, 197100)]

# (h) fixed full-scale-war requirement: theta 2 vs 8 below the risk threshold
rF = copy.deepcopy(M.r); oF = copy.deepcopy(M.o)
for j in J:
    oF[j] = list(oF[j]); rF[j] = list(rF[j]); oF[j][3] += rF[j][3] * N0; rF[j][3] = 0.0
fx = []
for N in range(340000, 194999, -15000):
    a = S(N, 2.0, rmat=rF, ovec=oF); b = S(N, 8.0, rmat=rF, ovec=oF)
    fx.append((N // 1000, r1(a['k5sh'], 2), r1(b['k5sh'], 2), r1(a['TC'] / 1000), r1(b['TC'] / 1000), r1(b['TC'] / a['TC'], 2)))
for N in (N3, N4):
    a = S(N, 2.0, rmat=rF, ovec=oF); b = S(N, 8.0, rmat=rF, ovec=oF)
    fx.append((N / 1000, r1(a['k5sh'], 2), r1(b['k5sh'], 2), r1(a['TC'] / 1000), r1(b['TC'] / 1000), r1(b['TC'] / a['TC'], 2)))
R['fixedwar_path'] = fx
R['fixedwar_k5_N0'] = {th: r1(S(N0, th, rmat=rF, ovec=oF)['k5sh'], 2) for th in (2.0, 8.0)}

# (i) moved peaks: intelligence peaks in limited conflict and protection in tension (columns swapped)
alt_o = copy.deepcopy(M.o); alt_r = copy.deepcopy(M.r)
alt_o['j1_정보'] = [22000, 26000, 30000, 22000]
alt_r['j5_방호'] = [0.05, 0.13, 0.08, 0.11]
alt2_o = copy.deepcopy(M.o); alt2_r = copy.deepcopy(M.r)
alt2_o['j1_정보'] = [30000, 26000, 22000, 22000]          # intelligence peaks in peacetime
alt2_r['j6_지속'] = [0.12, 0.16, 0.13, 0.14]             # sustainment peaks in tension
pk = {}
for lab, rr, oo in (('swap_intel_protection', alt_r, alt_o), ('intel_peace_sust_tension', alt2_r, alt2_o)):
    pk[lab] = {th: [(N // 1000, r1(S(N, th, rmat=rr, ovec=oo)['k5sh'], 2)) for N in (365000, 289000, 219000, 197100)] for th in (2.0, 8.0)}
R['moved_peaks'] = pk


out = HERE / ('supplement.json' if TAG == 'base' else f'supplement_{TAG}.json')
json.dump(json.loads(json.dumps(R, default=str)), open(out, 'w'), ensure_ascii=False, indent=1)
for k, v in R.items():
    print(k, json.dumps(v, ensure_ascii=False, default=str))
