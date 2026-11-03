"""Recounts year-to-year posture transitions in posture_coding.json (annual coding,
1958-2023; 1 = peacetime, 2 = tension, 3 = limited conflict) and prints the
maximum-likelihood transition rows next to the matrix F used in
model/potential_structure_lp.py.
"""
import json, pathlib, sys
import numpy as np
here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(here.parents[1] / 'model'))
import potential_structure_lp as M

c = json.load(open(here.with_name('posture_coding.json')))
ys = sorted(map(int, c))
C = np.zeros((4, 4))
for a, b in zip(ys, ys[1:]):
    if b == a + 1:
        C[c[str(a)] - 1, c[str(b)] - 1] += 1
print('transition counts (rows/cols: peace, tension, limited, war)')
print(C)
print('MLE rows for peace, tension, limited')
print((C[:3] / C[:3].sum(1, keepdims=True)).round(4))
print('F in the model')
print(M.F)
