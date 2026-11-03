import sys, copy, json, itertools, pulp
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'model')); import potential_structure_lp as M
K, J, PAIRS = M.K, M.J, M.PAIRS
R6 = 'k6_reserve'
def solve(N, theta, cap=0.0, e=0.8, h6=0.1, psi6=1.0, rmat=None, ovec=None, gamma=2.0):
    rmat = rmat or M.r; ovec = ovec or M.o
    W, PI = M._WPI(M.F)
    KK = K + [R6]
    A = {**M.A, R6: {j: e*M.A['k1_전투전력'][j] for j in J}}
    h = {**M.h, R6: h6}; psi = {**M.psi, R6: psi6}
    m = pulp.LpProblem('PSR', pulp.LpMinimize)
    n = {k: pulp.LpVariable(f'n_{k}', lowBound=0) for k in KK}
    x = {(k,j,s): pulp.LpVariable(f'x_{k}_{j}_{s}', 0) for k in KK for j in J for s in range(4)}
    dp = {(k,j,p): pulp.LpVariable(f'dp_{k}_{j}_{p}', 0) for k in KK for j in J for p in PAIRS}
    dm = {(k,j,p): pulp.LpVariable(f'dm_{k}_{j}_{p}', 0) for k in KK for j in J for p in PAIRS}
    u = {(j,s): pulp.LpVariable(f'u_{j}_{s}', 0) for j in J for s in range(4)}
    HC = pulp.lpSum(h[k]*n[k] for k in KK)
    OP = pulp.lpSum(M.LAMBDA*PI[s]*x[k,j,s] for k in KK for j in J for s in range(4))
    RK = pulp.lpSum(theta*M.SIGMA[s]*M.rho[j]*u[j,s] for j in J for s in range(4))
    TC = pulp.lpSum(W[p]*psi[k]*M.phi[j]*(dp[k,j,p]+dm[k,j,p]) for k in KK for j in J for p in PAIRS)
    m += HC + OP + RK + gamma*TC
    for j in J:
        for s in range(4):
            m += pulp.lpSum(A[k][j]*x[k,j,s] for k in KK) + u[j,s] >= rmat[j][s]*N + ovec[j][s]
    for k in KK:
        for s in range(4):
            m += pulp.lpSum(x[k,j,s] for j in J) <= n[k]
    for j in J:
        for s in (0,1):
            m += x[R6,j,s] == 0          # reserves are mobilized only in limited conflict and full-scale war
    m += pulp.lpSum(n[k] for k in K) <= N   # standing-force ceiling excludes reservists
    m += n[R6] <= cap
    for k in KK:
        for j in J:
            for (a,b) in PAIRS:
                m += x[k,j,a]-x[k,j,b] == dp[k,j,(a,b)]-dm[k,j,(a,b)]
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[m.status] == 'Optimal'
    nv = {k: n[k].value() for k in KK}; act = sum(nv[k] for k in K)
    used_war = sum(x[R6,j,3].value() for j in J)
    return dict(k5=100*nv['k5_전문기술']/act, gap=sum(u[j,s].value() for j in J for s in range(4))/1000,
                war=sum(u[j,3].value() for j in J)/1000, TC=pulp.value(TC)/1000, res=nv[R6]/1000, res_war=used_war/1000)

def thresholds(**kw):
    ent = risk = None
    for N in range(370000, 179999, -5000):
        s = solve(N, **kw)
        if ent is None and s['k5'] > 0.05: ent = N//1000
        if risk is None and s['war'] > 0.05: risk = N//1000
        if ent and risk: break
    return ent, risk

# fixed full-scale-war requirement (B-1 test in §5.4)
N0 = 365000; rF = copy.deepcopy(M.r); oF = copy.deepcopy(M.o)
for j in J:
    oF[j] = list(oF[j]); rF[j] = list(rF[j]); oF[j][3] += rF[j][3]*N0; rF[j][3] = 0.0

out = {}
mode = sys.argv[1]
if mode == 'check':   # cap 0 must reproduce the baseline model
    for th in (2.0, 8.0):
        a = solve(219000, th, cap=0); b = M.solve(219000, 2.0, rho_scale=th)
        print(th, round(a['k5'],2), round(b['k5sh'],2), round(a['TC'],2), round(b['TC']/1000,2), round(a['gap'],2), round(b['SHORT']/1000,2))
if mode == 'base':
    for th, cap, e, h6 in itertools.product((2.0, 8.0), (0, 3700, 16000, 30000, 50000), (0.5, 0.8), (0.1, 0.5)):
        if cap == 0 and (e, h6) != (0.5, 0.1): continue
        ent, risk = thresholds(theta=th, cap=cap, e=e, h6=h6)
        n3 = solve(219000, th, cap=cap, e=e, h6=h6); n4 = solve(197000, th, cap=cap, e=e, h6=h6)
        key = f'{th}|{cap}|{e}|{h6}'
        out[key] = dict(entry=ent, risk=risk, N3={k: round(v,2) for k,v in n3.items()}, N197={k: round(v,2) for k,v in n4.items()})
        print(key, out[key], flush=True)
    json.dump(out, open(str(__import__('pathlib').Path(__file__).with_name('reserve_base.json')),'w'), indent=1)
if mode == 'fixed':
    for th, cap, e, h6 in itertools.product((2.0, 8.0), (0, 16000, 30000, 50000), (0.5, 0.8), (0.1,)):
        if cap == 0 and e != 0.5: continue
        ent, risk = thresholds(theta=th, cap=cap, e=e, h6=h6, rmat=rF, ovec=oF)
        n3 = solve(219000, th, cap=cap, e=e, h6=h6, rmat=rF, ovec=oF)
        key = f'{th}|{cap}|{e}|{h6}'
        out[key] = dict(entry=ent, risk=risk, N3={k: round(v,2) for k,v in n3.items()})
        print(key, out[key], flush=True)
    json.dump(out, open(str(__import__('pathlib').Path(__file__).with_name('reserve_fixed_grid.json')),'w'), indent=1)
