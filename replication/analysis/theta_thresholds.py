"""Thresholds in the weights at N3 (gamma = 2).
Finding 2: the smallest common weight theta at which transition cost at N3 exceeds its N2 level, and the smallest
theta at which it exceeds the N2 level by more than half.
Finding 4: raising only the full-scale-war weight (sub-war postures at theta = 2), the smallest full-scale-war weight
from which no multifunctional personnel are held at N3.
Scans on a 0.01 grid, then refines each crossing on a 0.001 grid. Writes theta_thresholds.json next to this script."""
import sys, json, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'model'))
import potential_structure_lp as M

N2, N3 = 289000, 219000
tc = lambda N, th: M.solve(N, 2.0, rho_scale=float(th))['TC']
tc_n2 = tc(N2, 2.0)  # the N2 force meets every requirement; its transition cost does not depend on theta

def first_above(level, lo=2.0, hi=8.0):
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 2)
    prev = lo
    for th in grid:
        if tc(N3, th) > level:
            fine = np.round(np.arange(prev, th + 1e-9, 0.001), 3)
            return float(next(t for t in fine if tc(N3, t) > level)), round(tc(N3, th))
        prev = th
    return None, None

def war_sig(tw):  # sub-war consequence weights at the common weight 2; the war weight scaled to tw
    return {0: .3, 1: .5, 2: .8, 3: 1.0 * tw / 2}
k5_war_only = lambda tw: M.solve(N3, 2.0, rho_scale=2.0, sigmamap=war_sig(float(tw)))['k5sh']

def first_zero(lo=2.0, hi=16.0):
    prev = lo
    for tw in np.round(np.arange(lo, hi + 1e-9, 0.01), 2):
        if k5_war_only(tw) < 1e-6:
            return float(next(t for t in np.round(np.arange(prev, tw + 1e-9, 0.001), 3) if k5_war_only(t) < 1e-6))
        prev = tw
    return None

out = {'TC_N2': round(tc_n2)}
out['exceeds_N2'] = first_above(tc_n2)
out['exceeds_N2_by_half'] = first_above(1.5 * tc_n2)
out['war_only_zero_share_from'] = first_zero()
json.dump(out, open(HERE / 'theta_thresholds.json', 'w'), indent=1)
print(out)
