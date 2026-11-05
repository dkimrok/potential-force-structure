"""Figures 1 and 2 from results.json (run tables.py first)."""
import json, pathlib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
R = json.load(open(HERE / 'results.json'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

# Figure 1: frontier at theta = 8, N0 and N2
f1 = R['fig1']['8.0']
fig, ax = plt.subplots(figsize=(8.06, 5.5), dpi=150)
for N, sty, lab in (('365', dict(color='black', ls='-', marker='o', mfc='white'), 'N0 = 365,000'),
                    ('289', dict(color='black', ls='--', marker='s', mfc='white'), 'N2 = 289,000')):
    pts = f1[N]
    ax.plot([p[1] for p in pts], [p[2] for p in pts], lw=1.6, ms=7, mew=1.0, label=lab, **sty)
for g, dx, dy in ((0, 4, 4), (0.5, 3, 4), (1, 3, 4), (2, 3, 4), (5, 3, 4)):
    p = [q for q in f1['365'] if q[0] == g][0]
    ax.annotate(f'γ={g:g}', (p[1], p[2]), xytext=(p[1] + dx, p[2] + dy), fontsize=10)
ax.set_xlabel('Transition cost  (thousand person-equivalents)', fontsize=11)
ax.set_ylabel('Steady-state cost  (holding + operating, thousand)', fontsize=11)
ax.set_title('Figure 1. Cost trade-off frontier of the potential structure', fontsize=12)
ax.grid(True, ls=':', lw=0.6, color='#BBBBBB'); ax.legend(loc='upper right', fontsize=10)
fig.tight_layout(); fig.savefig(HERE / 'figure1.png', dpi=150)

# Figure 2: downsizing path at theta = 2 and 8
P = R['fig2']
fig, axs = plt.subplots(3, 1, figsize=(6.4, 7.6), sharex=True, dpi=150)
spec = [(1, '(a) Multifunctional share (%)'), (2, '(b) Capability gap, all postures (thousand)'), (3, '(c) Transition cost (thousand)')]
sty = {'2.0': dict(color='black', ls='-', marker='o', mfc='white', label='θ = 2 (reference weight)'),
       '8.0': dict(color='#555555', ls='--', marker='s', mfc='white', label='θ = 8')}
for ax, (col, title) in zip(axs, spec):
    for th in ('2.0', '8.0'):
        ax.plot([r[0] for r in P[th]], [r[col] for r in P[th]], lw=1.6, ms=4.5, mew=1.0, **sty[th])
    ax.set_title(title, loc='left', fontsize=10)
    ax.grid(True, ls=':', lw=0.6, color='#BBBBBB')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[2].legend(loc='upper left', frameon=True, fontsize=8.5)
axs[2].set_xlabel('Army size N (thousand)')
axs[2].set_xlim(368, 194)
axs[2].set_xticks([360, 340, 320, 300, 289, 260, 240, 219, 197.1])
axs[2].set_xticklabels(['360', '340', '320', '300', '289\n(N₂)', '260', '240', '219\n(N₃)', '197\n(N₄)'])
fig.suptitle('Figure 2. Downsizing path at θ = 2 and θ = 8 (γ = 2)', fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(HERE / 'figure2.png', dpi=150)
print('saved')
