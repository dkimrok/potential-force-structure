"""Manuscript tables and most reported numbers from one script.

Usage:  python tables.py [base|alt|orig]
  base (default): transition matrix from the posture ledger (model.F); alt: model.F_ALT (1980 and 1995 as peacetime)
  orig          : the matrix of the original submission (model.F_ORIG), for comparison
Writes results.json (base) or results_<tag>.json next to this script and prints a readable summary.
"""
import sys, json, copy, itertools, pathlib
import numpy as np, pulp
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'model'))
import potential_structure_lp as M

TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
FM = {'base': M.F, 'alt': M.F_ALT, 'orig': M.F_ORIG}[TAG]
K, J = M.K, M.J
N0, N1, N2, N3, N4 = 365000, 348000, 289000, 219000, 197100
R = {}
r1 = lambda v, d=1: round(float(v), d)

def S(N, th, g=2.0, **kw):
    kw.setdefault('Fmat', FM)
    return M.solve(N, g, rho_scale=th, **kw)

def pi_of(F):
    w, v = np.linalg.eig(F.T); p = np.real(v[:, np.argmin(abs(w - 1))]); return p / p.sum()

# --- Table 1
p = pi_of(FM)
R['T1'] = {'F': np.round(FM, 3).tolist(), 'pi': np.round(p, 3).tolist(), 'p_war': float(p[3])}

# --- practice-implied minimum weight at N0 (smallest theta with no shortfall, gamma = 2)
lo = None
for th in np.arange(1.00, 3.001, 0.01):
    s = S(N0, th)
    if s['SHORT'] < 1e-6:
        lo = round(float(th), 2); break
last = S(N0, lo - 0.01)
R['theta_min'] = {'theta_min': lo, 'gap_just_below': r1(last['SHORT'] / 1000, 2),
                  'war_gap_just_below': r1(last['SHORT_s4'] / 1000, 2)}

# --- Table 4 (N0 frontier) and Table 5 (maneuver deployment)
R['T4'] = {}
for th in (2.0, 8.0):
    R['T4'][th] = [(g, r1(S(N0, th, g)['SS'] / 1000), r1(S(N0, th, g)['TC'] / 1000), round(S(N0, th, g)['CI'] / 1000))
                  for g in (0, 0.5, 1, 2, 5)]
R['T5'] = {g: [r1(v / 1000) for v in S(N0, 2.0, g)['X']['j3_기동']] for g in (0, 2)}

# --- Table 6 (path)
R['T6'] = {}
for th in (2.0, 8.0):
    rows = []
    for N in (N0, N1, N2, N3, N4):
        s = S(N, th)
        rows.append(dict(N=N // 1000 if N != N4 else 197.1, TC=r1(s['TC'] / 1000), SD=round(s['CI'] / 1000),
                         gap=r1(s['SHORT'] / 1000), war=r1(s['SHORT_s4'] / 1000), k5=r1(s['k5sh'], 2)))
    R['T6'][th] = rows

# --- thresholds on the 5,000 grid
def entry_risk(th, **kw):
    ent = risk = anygap = None
    for N in range(400000, 179999, -5000):
        s = S(N, th, **kw)
        if ent is None and s['k5sh'] > 0.05: ent = N // 1000
        if anygap is None and s['SHORT'] > 1: anygap = N // 1000
        if risk is None and s['SHORT_s4'] > 1: risk = N // 1000
    return ent, anygap, risk
R['thresholds'] = {th: entry_risk(th) for th in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0)}
R['k5_at_top'] = {th: [r1(S(N, th)['k5sh'], 2) for N in (400000, 365000, 348000, 320000, 300000, 289000)] for th in (2.0, 8.0)}

# --- zero-shortfall feasibility limit
def min_holdings(N, postures):
    m = pulp.LpProblem('f', pulp.LpMinimize)
    n = {k: pulp.LpVariable('n' + str(i), 0) for i, k in enumerate(K)}
    x = {(k, j, s): pulp.LpVariable(f'x{a}_{b}_{s}', 0) for a, k in enumerate(K) for b, j in enumerate(J) for s in postures}
    m += pulp.lpSum(n.values())
    for j in J:
        for s in postures:
            m += pulp.lpSum(M.A[k][j] * x[k, j, s] for k in K) >= M.r[j][s] * N + M.o[j][s]
    for k in K:
        for s in postures:
            m += pulp.lpSum(x[k, j, s] for j in J) <= n[k]
    m.solve(pulp.PULP_CBC_CMD(msg=0)); return pulp.value(m.objective)
feas = {}
for lab, ps in (('all', [0, 1, 2, 3]), ('war', [3])):
    lastN = None
    for N in range(400000, 150000, -1000):
        if min_holdings(N, ps) <= N + 1e-6: lastN = N
        else: break
    feas[lab] = lastN
R['feasibility_limit'] = feas

# --- decomposition: technical type allowed vs excluded (gamma = 2)
def decomp(N, th):
    a = S(N, th); h2 = dict(M.h); h2['k5_전문기술'] = 1e6
    b = S(N, th, hvec=h2)
    hc = lambda s: sum(M.h[k] * s['n'][k] for k in K)
    Z = lambda s, hh: s['HC'] + s['OP'] + 2.0 * s['TC']  # risk term added below
    risk = lambda s: sum(th * M.SIGMA[sx] * M.rho[j] * 0 for j in J for sx in range(4))
    return dict(k5=r1(a['k5sh'], 2), dOP=r1((a['OP'] - b['OP']) / 1000, 2), dHC=r1((hc(a) - hc(b)) / 1000, 2),
                dTC=r1((a['TC'] - b['TC']) / 1000, 2), dGap=r1((a['SHORT'] - b['SHORT']) / 1000, 2),
                net_noRisk=r1(((a['OP'] - b['OP']) + (hc(a) - hc(b)) + 2 * (a['TC'] - b['TC'])) / 1000, 2))
R['decomp'] = {f'{N // 1000}_{th}': decomp(N, th) for N in (N0, N2, N3) for th in (2.0, 8.0)}

# --- factorial at N3 (Table 7, Table 8, Appendix A)
mphi = np.mean(list(M.phi.values()))
phid = lambda l: {j: max(0.1, mphi + l * (M.phi[j] - mphi)) for j in J}
psik5 = lambda v: {**M.psi, 'k5_전문기술': v}
hk5 = lambda v: {**M.h, 'k5_전문기술': v}
names = ['risk', 'cross-training', 'dispersion', 'holding premium']
def factorial(hi):
    rows = []
    for sg in itertools.product([-1, 1], repeat=4):
        kw = dict(rho_scale=(2.0 if sg[0] < 0 else hi), psivec=psik5(0.6 if sg[1] < 0 else 0.2),
                  phivec=phid(0.0 if sg[2] < 0 else 2.0), hvec=hk5(2.2 if sg[3] < 0 else 1.4))
        s = M.solve(N3, 2.0, Fmat=FM, **kw); rows.append((sg, s['k5sh'], s['TC'] / 1000))
    eff = {}
    for i, n in enumerate(names):
        eff[n] = (r1(np.mean([k for s, k, t in rows if s[i] > 0]) - np.mean([k for s, k, t in rows if s[i] < 0]), 2),
                  r1(np.mean([t for s, k, t in rows if s[i] > 0]) - np.mean([t for s, k, t in rows if s[i] < 0]), 2))
    inter = {}
    for i, j in itertools.combinations(range(4), 2):
        inter[f'{names[i]} x {names[j]}'] = (
            r1(np.mean([k for s, k, t in rows if s[i] * s[j] > 0]) - np.mean([k for s, k, t in rows if s[i] * s[j] < 0]), 2),
            r1(np.mean([t for s, k, t in rows if s[i] * s[j] > 0]) - np.mean([t for s, k, t in rows if s[i] * s[j] < 0]), 2))
    return dict(main=eff, interactions=inter, range_k5=(r1(min(r[1] for r in rows), 2), r1(max(r[1] for r in rows), 2)),
                runs=[(list(s), r1(k, 2), r1(t, 2)) for s, k, t in rows])
R['factorial'] = {hi: factorial(hi) for hi in (4.0, 6.0, 8.0, 12.0)}
R['A1'] = {th: {ps: r1(S(N3, th, psivec=psik5(ps))['k5sh'], 2) for ps in (0.6, 0.2)} for th in (2.0, 8.0)}

# --- Table 9 (theta sweep at N3) and the war-only weight experiment
R['T9'] = [(th, r1(S(N3, float(th))['k5sh'], 2), r1(S(N3, float(th))['SHORT'] / 1000), r1(S(N3, float(th))['SHORT_s4'] / 1000),
            r1(S(N3, float(th))['TC'] / 1000)) for th in range(2, 9)]
war_only = []
for tw in (2, 3, 4, 6, 8):
    sig = {0: .3, 1: .5, 2: .8, 3: 1.0 * tw / 2}
    a = S(N3, 2.0, sigmamap=sig); b = S(N4, 2.0, sigmamap=sig)
    ent, anyg, risk = entry_risk(2.0, sigmamap=sig)
    war_only.append(dict(theta_war=tw, entry=ent, risk=risk, N3_k5=r1(a['k5sh'], 2), N3_gap=r1(a['SHORT'] / 1000),
                         N3_war=r1(a['SHORT_s4'] / 1000), N3_TC=r1(a['TC'] / 1000), N4_k5=r1(b['k5sh'], 2),
                         N4_war=r1(b['SHORT_s4'] / 1000), N4_TC=r1(b['TC'] / 1000)))
R['war_only'] = war_only

# --- deep decline
R['deep'] = {th: [(N / 1000, r1(S(N, th)['k5sh'], 2), r1(S(N, th)['SHORT_s4'] / 1000)) for N in
                  (289000, 270000, 260000, 255000, 250000, 245000, 240000, 235000, 230000, 225000, 220000, 215000, 210000, 205000, 200000, 197100, 190000)]
             for th in (2.0, 8.0, 16.0)}

# --- robustness: war occupancy (escalation prior and war persistence)
pw = []
for lw, ww in ((0.02, 0.6667), (0.035, 0.6667), (0.06, 0.75), (0.08, 0.80)):
    F2 = FM.copy(); obs = FM[2, :3] / FM[2, :3].sum()
    F2[2, :3] = obs * (1 - lw); F2[2, 3] = lw; F2[3] = [0, 0, 1 - ww, ww]
    pwar = float(pi_of(F2)[3])
    pw.append(dict(alpha=lw, war_persist=ww, p_war=round(pwar * 100, 2),
                   entry={th: entry_risk(th, Fmat=F2)[0] for th in (2.0, 8.0)}))
R['pwar_range'] = pw

# --- fixed full-scale-war requirement
rF = copy.deepcopy(M.r); oF = copy.deepcopy(M.o)
for j in J:
    oF[j] = list(oF[j]); rF[j] = list(rF[j]); oF[j][3] += rF[j][3] * N0; rF[j][3] = 0.0
R['fixedwar'] = {th: entry_risk(th, rmat=rF, ovec=oF) for th in (2.0, 8.0)}

# --- monotone requirement profile
mono_r = copy.deepcopy(M.r); mono_o = copy.deepcopy(M.o)
for j in J:
    for s in range(1, 4):
        mono_r[j][s] = max(mono_r[j][s], mono_r[j][s - 1]); mono_o[j][s] = max(mono_o[j][s], mono_o[j][s - 1])
R['monotone_max_k5'] = {th: r1(max(S(N, th, rmat=mono_r, ovec=mono_o)['k5sh'] for N in range(365000, 194999, -10000)), 2) for th in (2.0, 8.0)}

# --- psi sweep at N3
R['psi_sweep'] = {th: {ps: (r1(S(N3, th, psivec=psik5(ps))['k5sh'], 2), r1(S(N3, th, psivec=psik5(ps))['TC'] / 1000)) for ps in (0.2, 0.3, 0.4, 0.5, 0.6)} for th in (2.0, 8.0)}

# --- multiplier V = (rho_maneuver * theta * sigma_war) / p_war
R['multiplier'] = {lab: round(2 * th * 1.0 / p[3]) for lab, th in (('theta_min', R['theta_min']['theta_min']), ('theta_2', 2.0), ('theta_8', 8.0))}
R['multiplier']['pwar_range_theta2'] = [round(4 / (x['p_war'] / 100)) for x in pw]

# --- transition-weight sensitivity: long-run flow weights instead of conditional sums
W, PI = M._WPI(FM)
flow = {pq: p[pq[0]] * FM[pq[0], pq[1]] + p[pq[1]] * FM[pq[1], pq[0]] for pq in M.PAIRS}
scale = sum(W.values()) / sum(flow.values())
def solveW(N, th, Wd):
    orig = M._WPI
    M._WPI = lambda F: (Wd, {s: PI[s] for s in range(4)})
    try: return M.solve(N, 2.0, rho_scale=th, Fmat=FM)
    finally: M._WPI = orig
wsens = {'W_conditional': {str(k): round(v, 4) for k, v in W.items()}, 'W_flow': {str(k): round(v, 4) for k, v in flow.items()}, 'scale': round(scale, 3)}
for lab, Wd in (('flow_raw', flow), ('flow_scaled', {k: v * scale for k, v in flow.items()})):
    for th in (2.0, 8.0):
        ent = risk = None
        for N in range(400000, 179999, -5000):
            s = solveW(N, th, Wd)
            if ent is None and s['k5sh'] > 0.05: ent = N // 1000
            if risk is None and s['SHORT_s4'] > 1: risk = N // 1000
        a = solveW(N3, th, Wd); top = solveW(N0, th, Wd)
        wsens[f'{lab}_{th}'] = dict(entry=ent, risk=risk, N0_k5=r1(top['k5sh'], 2), N3_k5=r1(a['k5sh'], 2), N3_gap=r1(a['SHORT'] / 1000), N3_war=r1(a['SHORT_s4'] / 1000))
R['W_sensitivity'] = wsens

# --- Figure 1 data (theta = 8 and 2) and Figure 2 data
G = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5, 8]
R['fig1'] = {th: {N // 1000: [(g, r1(S(N, th, g)['TC'] / 1000), r1(S(N, th, g)['SS'] / 1000), r1(S(N, th, g)['SHORT'] / 1000)) for g in G] for N in (N0, N2)} for th in (2.0, 8.0)}
grid = sorted(set(list(range(365000, 224999, -5000)) + [289000, 219000, 215000, 210000, 205000, 200000, 197100]), reverse=True)
R['fig2'] = {th: [(N / 1000, S(N, th)['k5sh'], S(N, th)['SHORT'] / 1000, S(N, th)['TC'] / 1000, S(N, th)['SHORT_s4'] / 1000) for N in grid] for th in (2.0, 8.0)}

out = HERE / ('results.json' if TAG == 'base' else f'results_{TAG}.json')
json.dump(json.loads(json.dumps(R, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))), open(out, 'w'), ensure_ascii=False, indent=1)
for k, v in R.items():
    if k in ('fig1', 'fig2', 'factorial'): continue
    print(k, json.dumps(v, ensure_ascii=False, default=str)[:900])
for hi, f in R['factorial'].items():
    print('factorial upper', hi, 'main', f['main'], 'range', f['range_k5'])
print('interactions (upper 8):', R['factorial'][8.0]['interactions'])
