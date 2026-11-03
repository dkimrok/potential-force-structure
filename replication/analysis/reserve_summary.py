import sys, json, itertools; sys.argv=['reserve_test','none']
import reserve_test as res, potential_structure_lp as M
out={}
for th, cap, e, h6 in itertools.product((2.0,8.0),(0,3700,16000,30000,50000),(0.5,0.8),(0.1,0.5)):
    if cap==0 and (e,h6)!=(0.5,0.5): continue
    kw=dict(theta=th,cap=cap,e=e,h6=h6,rmat=res.rF,ovec=res.oF)
    ent,risk=res.thresholds(**kw)
    a=res.solve(255500,**kw); b=res.solve(197100,**kw)
    out[f'{th}|{cap}|{e}|{h6}']=dict(entry=ent,risk=risk,P2040={k:round(v,2) for k,v in a.items()},SQ={k:round(v,2) for k,v in b.items()})
    print(f'{th}|{cap}|{e}|{h6}', out[f'{th}|{cap}|{e}|{h6}'], flush=True)
json.dump(out,open(str(__import__('pathlib').Path(__file__).with_name('reserve_fixed.json')),'w'),indent=1)
# suppression along deep decline, baseline requirements, no reserves
for N in range(255000,189999,-5000):
    a=M.solve(N,2.0,rho_scale=2.0); b=M.solve(N,2.0,rho_scale=8.0)
    print('path',N//1000,round(a['k5sh'],2),round(b['k5sh'],2))
for N in (219000,197100):
    for th in (2.0,8.0):
        s=M.solve(N,2.0,rho_scale=th); print('base',N,th,round(s['k5sh'],2),round(s['SHORT_s4']/1000,2))
