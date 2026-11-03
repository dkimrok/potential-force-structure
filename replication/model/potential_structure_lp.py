"""
Potential-Structure Force-Design LP  (unified Ch3+Ch4 model)
"""
import pulp, itertools, numpy as np

K = ['k1_전투전력', 'k2_지휘정보', 'k3_전투지원', 'k4_근무지원', 'k5_전문기술']
J = ['j1_정보', 'j2_C2', 'j3_기동', 'j4_화력', 'j5_방호', 'j6_지속']
S = ['s1_평시', 's2_긴장', 's3_제한충돌', 's4_전면전']
PAIRS = list(itertools.combinations(range(4), 2))

r = {
 'j1_정보':[0,0,0,0], 'j2_C2':[0,0,0,0],
 'j3_기동':[0.12,0.16,0.24,0.32], 'j4_화력':[0.06,0.09,0.15,0.21],
 'j5_방호':[0.05,0.08,0.13,0.11], 'j6_지속':[0.14,0.12,0.13,0.16]}
o = {'j1_정보':[22000,30000,26000,22000], 'j2_C2':[24000,26000,27000,27000],
     'j3_기동':[0,0,0,0],'j4_화력':[0,0,0,0],'j5_방호':[0,0,0,0],'j6_지속':[0,0,0,0]}

A = {
 'k1_전투전력':{'j1_정보':0.2,'j2_C2':0.3,'j3_기동':1.0,'j4_화력':1.0,'j5_방호':0.7,'j6_지속':0.2},
 'k2_지휘정보':{'j1_정보':1.0,'j2_C2':1.0,'j3_기동':0.3,'j4_화력':0.3,'j5_방호':0.2,'j6_지속':0.2},
 'k3_전투지원':{'j1_정보':0.2,'j2_C2':0.2,'j3_기동':0.6,'j4_화력':0.3,'j5_방호':1.0,'j6_지속':0.4},
 'k4_근무지원':{'j1_정보':0.1,'j2_C2':0.3,'j3_기동':0.2,'j4_화력':0.2,'j5_방호':0.3,'j6_지속':1.0},
 'k5_전문기술':{'j1_정보':0.9,'j2_C2':0.8,'j3_기동':0.4,'j4_화력':0.5,'j5_방호':0.6,'j6_지속':0.5}}
phi = {'j1_정보':1.2,'j2_C2':1.0,'j3_기동':0.7,'j4_화력':1.1,'j5_방호':0.9,'j6_지속':0.6}
psi = {'k1_전투전력':1.2,'k2_지휘정보':1.0,'k3_전투지원':1.0,'k4_근무지원':1.1,'k5_전문기술':0.4}
h = {'k1_전투전력':1.0,'k2_지휘정보':1.3,'k3_전투지원':1.1,'k4_근무지원':0.9,'k5_전문기술':1.8}
rho = {'j1_정보':3.0,'j2_C2':3.0,'j3_기동':2.0,'j4_화력':2.5,'j5_방호':2.0,'j6_지속':1.5}
RHO_SCALE = 6.0
SIGMA = {0:0.3, 1:0.5, 2:0.8, 3:1.0}
# Posture-transition matrix. Rows/columns: peacetime, tension, limited conflict, full-scale war.
# Peacetime/tension/limited rows: relative frequencies of the 65 observed one-year transitions, 1958-2023
# (data/posture_ledger.csv, counts below). Limited-conflict row: observed shares multiplied by (1 - ALPHA)
# with an assumed escalation probability ALPHA to full-scale war. Full-scale-war row: doctrinal assumption.
# COUNTS_ALT codes 1980 and 1995 as peacetime (coding-sensitivity test, Section 5.4).
COUNTS = np.array([[11, 6, 7], [3, 8, 3], [9, 1, 17]], dtype=float)
COUNTS_ALT = np.array([[12, 6, 8], [3, 8, 3], [10, 1, 14]], dtype=float)
ALPHA = 0.035
def build_F(counts, alpha=ALPHA, war_row=(0.0, 0.0, 1/3, 2/3)):
    Fm = np.zeros((4, 4))
    Fm[:3, :3] = counts / counts.sum(axis=1, keepdims=True)
    Fm[2, :3] *= (1 - alpha); Fm[2, 3] = alpha
    Fm[3] = war_row
    return Fm
F = build_F(COUNTS)
F_ALT = build_F(COUNTS_ALT)
# matrix from the coding of the original submission, kept for comparison
F_ORIG = np.array([[0.6486,0.1622,0.1892,0.0000],
                  [0.3750,0.5000,0.1250,0.0000],
                  [0.4828,0.2414,0.2414,0.0345],
                  [0.0000,0.0000,0.3333,0.6667]])
W = {p: F[p[0],p[1]] + F[p[1],p[0]] for p in PAIRS}
_w,_v = np.linalg.eig(F.T); _pi = np.real(_v[:,np.argmin(abs(_w-1))]); PI = _pi/_pi.sum()
LAMBDA = 1.0
N_PATH = {'N0':365000,'N1':348000,'N2':289000,'N3':219000}

def _WPI(Fmat):
    Wd = {p: Fmat[p[0],p[1]]+Fmat[p[1],p[0]] for p in PAIRS}
    w,v = np.linalg.eig(Fmat.T); pi = np.real(v[:,np.argmin(abs(w-1))]); pi=pi/pi.sum()
    return Wd, {s:pi[s] for s in range(4)}

def solve(N, gamma, Amat=A, phivec=phi, psivec=psi, rmat=r, hvec=h, ovec=o,
          rhovec=rho, sigmamap=SIGMA, lam=LAMBDA, rho_scale=RHO_SCALE, Fmat=F):
    Wdict, PId = _WPI(Fmat)
    m = pulp.LpProblem('PS', pulp.LpMinimize)
    n = {k: pulp.LpVariable(f'n_{k}', lowBound=0) for k in K}
    x = {(k,j,s): pulp.LpVariable(f'x_{k}_{j}_{s}',0) for k in K for j in J for s in range(4)}
    dp= {(k,j,p): pulp.LpVariable(f'dp_{k}_{j}_{p}',0) for k in K for j in J for p in PAIRS}
    dm= {(k,j,p): pulp.LpVariable(f'dm_{k}_{j}_{p}',0) for k in K for j in J for p in PAIRS}
    u = {(j,s): pulp.LpVariable(f'u_{j}_{s}',0) for j in J for s in range(4)}
    HC = pulp.lpSum(hvec[k]*n[k] for k in K)
    OPterm = pulp.lpSum(lam*PId[s]*x[k,j,s] for k in K for j in J for s in range(4))
    RKterm = pulp.lpSum(rho_scale*sigmamap[s]*rhovec[j]*u[j,s] for j in J for s in range(4))
    TCterm = pulp.lpSum(Wdict[p]*psivec[k]*phivec[j]*(dp[k,j,p]+dm[k,j,p])
                        for k in K for j in J for p in PAIRS)
    m += HC + OPterm + RKterm + gamma*TCterm
    for j in J:
        for s in range(4):
            m += pulp.lpSum(Amat[k][j]*x[k,j,s] for k in K) + u[j,s] >= rmat[j][s]*N + ovec[j][s]
    for k in K:
        for s in range(4):
            m += pulp.lpSum(x[k,j,s] for j in J) <= n[k]
    m += pulp.lpSum(n[k] for k in K) <= N
    for k in K:
        for j in J:
            for (a,b) in PAIRS:
                m += x[k,j,a]-x[k,j,b] == dp[k,j,(a,b)] - dm[k,j,(a,b)]
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[m.status]
    nval = {k: n[k].value() for k in K}
    HCv = sum(hvec[k]*nval[k] for k in K)
    OPv = sum(lam*PId[s]*x[k,j,s].value() for k in K for j in J for s in range(4))
    TCv = sum(Wdict[p]*psivec[k]*phivec[j]*(dp[k,j,p].value()+dm[k,j,p].value())
              for k in K for j in J for p in PAIRS)
    CI  = sum(dp[k,j,p].value()+dm[k,j,p].value() for k in K for j in J for p in PAIRS)
    SHORT = sum(u[j,s].value() for j in J for s in range(4))
    SHORT_s4 = sum(u[j,3].value() for j in J)
    k5sh = 100*nval['k5_전문기술']/sum(nval.values())
    Xjs = {j:[sum(x[k,j,s].value() for k in K) for s in range(4)] for j in J}
    return dict(status=status, n=nval, HC=HCv, OP=OPv, SS=HCv+OPv, TC=TCv, CI=CI,
                SHORT=SHORT, SHORT_s4=SHORT_s4, k5sh=k5sh, X=Xjs,
                Ntot=sum(nval.values()), slack=N-sum(nval.values()))
