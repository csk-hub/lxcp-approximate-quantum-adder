import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import lxcp_adder as lx
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results'); FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'

# ---------------- Figure 1: block diagram ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis('off')
def box(x, y, w, h, text, fc='#EAF1FB', ec='#1F4E79', fs=8, bold=False, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=lw); ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color='#111111', linespacing=1.25)
def arrow(x1, y1, x2, y2, color='#333333', style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, lw=1.2, color=color, linestyle=ls, shrinkA=0, shrinkB=0))
# row 1 (classical front end)
ax.text(1, 61.5, "Classical front end", fontsize=8.5, fontweight='bold', color='#1F4E79')
xs = [1, 21, 41, 61, 81]; w = 17; y = 49; h = 9
labels = ["MIT-BIH ECG\n(360 Hz, ch. 0)", "Band-pass\n5–15 Hz", "Derivative\n(gradient)", "Squaring", "8-bit\nquantisation"]
for x, t in zip(xs, labels): box(x, y, w, h, t)
for i in range(4): arrow(xs[i] + w + 0.4, y + h/2, xs[i+1] - 0.4, y + h/2)
# row 2 (approximate adder tree)
ax.add_patch(FancyBboxPatch((1, 21), 60, 21, boxstyle="round,pad=0.3,rounding_size=1.5", fc='#FFF4E5', ec='#C55A11', lw=1.6))
ax.text(2.5, 39.2, "Approximate reversible adder tree (MWI)", fontsize=8.5, fontweight='bold', color='#C55A11')
box(3, 24, 17, 12, "54 squared\nsamples\n(150 ms window)", fc='white', ec='#C55A11')
box(24, 24, 17, 12, "53 adders,\nwidth 8→13 bits\nbalanced tree", fc='white', ec='#C55A11')
box(44, 24, 15, 12, "LXCP / LXCP-G\nlow k bits: CNOT\ncarry: 1 Toffoli", fc='#FFE3BF', ec='#C55A11', bold=False, fs=6.9)
arrow(20.4, 30, 23.6, 30, color='#C55A11'); arrow(41.4, 30, 43.6, 30, color='#C55A11')
# gate-cost side box
box(66, 24, 33, 18, "Gate-cost model\nToffoli count, T-count\n(7 T gates per Toffoli)\nClifford+T total, depth, qubits\n(Qiskit circuit, OpenQASM)", fc='#EEF7EC', ec='#2E7D32', fs=7.4)
arrow(59.4, 33, 65.6, 33, color='#2E7D32', ls='--')
# row 3 (decision + evaluation)
ax.text(1, 17.3, "Decision and evaluation (identical for every adder design)", fontsize=8.5, fontweight='bold', color='#1F4E79')
xs3 = [1, 27, 53, 79]; w3 = 20; y3 = 3; h3 = 11
labels3 = ["Median offset\nremoval,\nthreshold", "Peak detection,\nR-peak localisation\n(≥ 250 ms apart)", "Matching to MIT-BIH\nannotations\n(±150 ms)", "Se, PPV, DER,\nMWI NRMSE and\ncorrelation"]
for x, t in zip(xs3, labels3): box(x, y3, w3 if x != 27 else 22, h3, t, fs=7.8)
arrow(xs3[0] + w3 + 0.4, y3 + h3/2, xs3[1] - 0.4, y3 + h3/2)
arrow(xs3[1] + 22 + 0.4, y3 + h3/2, xs3[2] - 0.4, y3 + h3/2)
arrow(xs3[2] + w3 + 0.4, y3 + h3/2, xs3[3] - 0.4, y3 + h3/2)
# connectors between rows
ax.plot([89.5, 89.5, 31, 31], [48.6, 45.6, 45.6, 43.0], color='#333333', lw=1.2)
arrow(31, 43.2, 31, 42.4, color='#333333')
ax.plot([59.4, 63, 63, 11, 11], [28, 28, 19.6, 19.6, 15.2], color='#333333', lw=1.2)
arrow(11, 15.6, 11, 14.4, color='#333333')
plt.savefig(os.path.join(FIG,'fig1_block_diagram.png'), dpi=300, bbox_inches='tight', facecolor='white'); plt.close()

# ---------------- Figure 2: circuits ----------------
def draw_circuit(ax, qc, labels, title):
    nq = len(labels); gates = []
    for inst in qc.data:
        q = [qc.find_bit(x).index for x in inst.qubits]; gates.append((inst.operation.name, q))
    free = [0]*nq; col_busy = {}
    placed = []
    for name, q in gates:
        lo, hi = min(q), max(q)
        c = max(free[lo:hi+1])
        # avoid vertical overlap with gates already in that column
        while any(not (hi < l or lo > h) for (l, h) in col_busy.get(c, [])): c += 1
        col_busy.setdefault(c, []).append((lo, hi))
        for i in range(lo, hi+1): free[i] = c + 1
        placed.append((name, q, c))
    ncols = max(c for _, _, c in placed) + 1
    for i in range(nq):
        ax.plot([-0.6, ncols - 0.4], [-i, -i], color='#555555', lw=0.8, zorder=1)
        ax.text(-0.9, -i, labels[i], ha='right', va='center', fontsize=7.5)
    for name, q, c in placed:
        ys = [-i for i in q]; col = '#C55A11' if name == 'ccx' else '#1F4E79'
        ax.plot([c, c], [min(ys), max(ys)], color=col, lw=1.4, zorder=2)
        for i in q[:-1]: ax.add_patch(Circle((c, -i), 0.14, color=col, zorder=3))
        t = q[-1]; ax.add_patch(Circle((c, -t), 0.26, fc='white', ec=col, lw=1.4, zorder=3))
        ax.plot([c-0.26, c+0.26], [-t, -t], color=col, lw=1.2, zorder=4); ax.plot([c, c], [-t-0.26, -t+0.26], color=col, lw=1.2, zorder=4)
    ax.set_xlim(-3.2, ncols - 0.2); ax.set_ylim(-nq + 0.4, 0.8); ax.axis('off'); ax.set_title(title, fontsize=8.5, loc='left', fontweight='bold')
n, k = 4, 2
qa = lx.build_adder(n, k, True, False, True)
labels_a = ["c"] + [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)] + ["z"]
qb = lx.build_adder(n, k, True, True, False)
labels_b = labels_a + ["g1"]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 7.6), gridspec_kw={'height_ratios': [10, 11]})
draw_circuit(axs[0], qa, labels_a, f"(a) LXCP, n = {n}, k = {k}: {qa.count_ops()['ccx']} Toffoli, {qa.count_ops()['cx']} CNOT")
draw_circuit(axs[1], qb, labels_b, f"(b) LXCP-G, n = {n}, k = {k}: {qb.count_ops()['ccx']} Toffoli, {qb.count_ops()['cx']} CNOT")
axs[1].text(-3.1, -len(labels_b) - 0.1, "Orange: Toffoli gate;  blue: CNOT gate.  c = prediction/carry qubit, z = carry-out, g1 = extra carry qubit (left dirty).", fontsize=7, va='top')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig2_circuits.png'), dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
print(qa.count_ops(), qb.count_ops())
