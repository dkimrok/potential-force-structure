"""Entry and risk-acceptance thresholds under moved requirement peaks (baseline coding)."""
import sys, copy, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'model'))
import potential_structure_lp as M
alt_o = copy.deepcopy(M.o); alt_r = copy.deepcopy(M.r)
alt_o['j1_정보'] = [22000, 26000, 30000, 22000]; alt_r['j5_방호'] = [0.05, 0.13, 0.08, 0.11]
alt2_o = copy.deepcopy(M.o); alt2_r = copy.deepcopy(M.r)
alt2_o['j1_정보'] = [30000, 26000, 22000, 22000]; alt2_r['j6_지속'] = [0.12, 0.16, 0.13, 0.14]
out = {}
for lab, rr, oo in (('swap_intel_protection', alt_r, alt_o), ('intel_peace_sust_tension', alt2_r, alt2_o)):
    for th in (2.0, 8.0):
        ent = gap = war = None; ent_gap = None
        for N in range(400000, 179999, -5000):
            s = M.solve(N, 2.0, rho_scale=th, rmat=rr, ovec=oo)
            if ent is None and s['k5sh'] > 0.05: ent = N // 1000; ent_gap = round(s['SHORT'] / 1000, 2)
            if gap is None and s['SHORT'] > 1: gap = N // 1000
            if war is None and s['SHORT_s4'] > 1: war = N // 1000
        out[f'{lab}_{th}'] = dict(entry=ent, gap_at_entry=ent_gap, first_gap=gap, first_war_gap=war)
json.dump(out, open(pathlib.Path(__file__).with_name('moved_peaks.json'), 'w'), indent=1)
print(out)
