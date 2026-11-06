# ============ QRS detection: your five cells + the proposed LXCP / LXCP-G adders ============
# Run once, then restart the kernel:   %pip install wfdb scipy matplotlib numpy
import os, csv, time
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------- configuration -----------------------------
FS, W = 360, 54                                  # MIT-BIH rate; 150 ms window = 54 samples
RECORDS = ['100', '101', '103', '105', '119', '200']   # add more (or wfdb.get_record_list('mitdb'))
SECONDS = 300                                    # per record; None = full 30 min (much slower)
APPROX_CELLS = ['AFA1', 'AFA2', 'AFA3', 'AA2', 'AA3', 'LXCP', 'LXCP-G']   # LXCP-G = same accuracy as LXCP, fewer gates
KS = [2, 4, 6, 8]                                # approximate low bits per adder (k = 0 is exact)
OFFSET_REMOVE = True                             # subtract the median of the MWI before thresholding
DEMO_K = 4                                       # k used for the waveform figure
BEATS = set("NLRBAaJSVrFejnE/fQ?")               # beat annotation symbols
OUT = 'qrs_results'
os.makedirs(OUT, exist_ok=True)

# ------------- 1. your Verilog cells as 1-bit functions -> (Sum, Cout) -------------
CELLS = {
 'EXACT': lambda a, b, c: (a ^ b ^ c, (a & b) | (b & c) | (a & c)),
 'AFA1' : lambda a, b, c: ((a ^ b) & (1 - c), (a | b) & c),
 'AFA2' : lambda a, b, c: ((1 - a) | (b & c), a | (b & c)),
 'AFA3' : lambda a, b, c: ((a & b) | (1 - c), a),
 'AA2'  : lambda a, b, c: (1, (a & b) | (b & c) | (a & c)),
 'AA3'  : lambda a, b, c: (a ^ b ^ c, 1),
}
def _tt(f):                                      # index x = A | B<<1 | Cin<<2
    s = np.zeros(8, np.int64); c = np.zeros(8, np.int64)
    for x in range(8):
        s[x], c[x] = f(x & 1, (x >> 1) & 1, (x >> 2) & 1)
    return s, c
TT = {n: _tt(f) for n, f in CELLS.items()}

def _anf(f):                                     # algebraic normal form -> monomial bitmasks
    g = [int(v) for v in f]
    for i in range(3):
        for x in range(8):
            if (x >> i) & 1: g[x] ^= g[x ^ (1 << i)]
    return [m for m in range(8) if g[m]]
ANF = {n: (_anf(s), _anf(c)) for n, (s, c) in TT.items()}

def cell_ccx(cell):                              # Toffolis per bit in the reversible circuit
    cost = {0: 0, 1: 0, 2: 1, 3: 3}              # degree-3 monomial = 3 CCX via a work qubit
    s_m, c_m = ANF[cell]
    return sum(cost[bin(m).count('1')] for m in s_m + c_m)

# ------------- proposed adder: Lower-XOR, Carry-Predicted (see lxcp_adder.py for the circuits) -------------
def lxcp_add(x, y, width, k):
    x = x.astype(np.int64); y = y.astype(np.int64); k = min(k, width)
    if k == 0: return x + y
    low = (x ^ y) & ((1 << k) - 1)                       # bits < k : XOR only, no carries
    pred = (x >> (k - 1)) & (y >> (k - 1)) & 1           # predicted carry into bit k
    return low | (((x >> k) + (y >> k) + pred) << k)     # exact upper part

def lxcp_ref(x, y, width, k):                            # slow bit-by-bit reference
    c = np.zeros_like(x); res = np.zeros_like(x)
    for i in range(width):
        a_, b_ = (x >> i) & 1, (y >> i) & 1
        if i < k:
            res |= (a_ ^ b_) << i
            c = a_ & b_ if i == k - 1 else np.zeros_like(x)
        else:
            res |= (a_ ^ b_ ^ c) << i
            c = (a_ & b_) | (b_ & c) | (a_ & c)
    return res | (c << width)

# ------------- 2. vectorised hybrid adder: bits < k use `cell`, the rest are exact -------------
def cell_add(x, y, width, cell, k):
    x = x.astype(np.int64); y = y.astype(np.int64)
    k = min(k, width)
    if cell in ('LXCP', 'LXCP-G'): return lxcp_add(x, y, width, k)
    if cell == 'EXACT' or k == 0:
        return x + y
    ts, tc = TT[cell]
    c = np.zeros_like(x); res = np.zeros_like(x)
    for i in range(k):
        idx = ((x >> i) & 1) | (((y >> i) & 1) << 1) | (c << 2)
        res |= ts[idx] << i
        c = tc[idx]
    return res | (((x >> k) + (y >> k) + c) << k)    # exact upper part with the carry from the cells

def add_ref(x, y, width, cell, k):               # slow bit-by-bit reference (for the self-test)
    c = np.zeros_like(x); res = np.zeros_like(x)
    for i in range(width):
        ts, tc = TT[cell] if i < k else TT['EXACT']
        idx = ((x >> i) & 1) | (((y >> i) & 1) << 1) | (c << 2)
        res |= ts[idx] << i
        c = tc[idx]
    return res | (c << width)

# self-test: fast adder == reference for all 8-bit inputs, every cell and every k
A_, B_ = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
A_, B_ = A_.ravel(), B_.ravel()
for cell in [c for c in APPROX_CELLS if c in TT]:          # the five Verilog cells
    for k in range(9):
        assert (cell_add(A_, B_, 8, cell, k) == add_ref(A_, B_, 8, cell, k)).all(), (cell, k)
assert (cell_add(A_, B_, 8, 'AFA1', 0) == A_ + B_).all()
for k in range(9):
    assert (lxcp_add(A_, B_, 8, k) == lxcp_ref(A_, B_, 8, k)).all(), k
assert (lxcp_add(A_, B_, 8, 1) == A_ + B_).all()          # k = 1 is exact
print("adder self-test passed")

# ------------- 3. moving-window integration built from the approximate adders -------------
def tree_sum(x, cell, k, width=8):
    while x.shape[1] > 1:
        n = x.shape[1]; pairs = n // 2
        s = cell_add(x[:, 0:2*pairs:2], x[:, 1:2*pairs:2], width, cell, k)
        x = np.concatenate([s, x[:, -1:]], axis=1) if n % 2 else s
        width += 1
    return x[:, 0]

def mwi(q, cell, k):
    win = sliding_window_view(q, W)
    out = np.empty(win.shape[0], dtype=np.int64)
    for s in range(0, len(out), 20000):
        out[s:s+20000] = tree_sum(win[s:s+20000].astype(np.int64), cell, k)
    return np.concatenate([np.zeros(W - 1, dtype=np.int64), out])

def adder_ccx(cell, k, w):
    """Toffolis in ONE w-bit adder. Common accounting: every design's exact upper part is an in-place
    Cuccaro adder (2 per bit); the five cells add their own low-part cost (ANF mapping);
    LXCP = 0-Toffoli low part + prediction + uncompute; LXCP-G = 1-Toffoli-per-bit upper part, carries left dirty."""
    k = min(k, w)
    if cell == 'EXACT' or k == 0: return 2 * w
    if cell == 'LXCP':   return 1 if k == w else 2 * (w - k) + 2
    if cell == 'LXCP-G': return 1 if k == w else (w - k) + 1
    return k * cell_ccx(cell) + 2 * (w - k)

def tree_toffolis(cell, k, n=W, w=8):
    total = 0
    while n > 1:
        pairs = n // 2
        total += pairs * adder_ccx(cell, k, w)
        n = pairs + n % 2; w += 1
    return total
print("exact adder-tree Toffolis (Cuccaro):", tree_toffolis('EXACT', 0), "(expected 946)")

# ------------- 4. Pan-Tompkins style front end, detector, scoring -------------
def preprocess(sig):
    b, a = butter(2, [5/(FS/2), 15/(FS/2)], btype='band')
    f = filtfilt(b, a, sig)
    s = np.gradient(f) ** 2
    scale = np.percentile(s, 99.5)
    q = np.clip(np.round(s / max(scale, 1e-12) * 255), 0, 255).astype(np.int64)   # 8-bit input to the adders
    return f, q

def detect(f, m):
    m = m.astype(float)
    if OFFSET_REMOVE: m = m - np.median(m)
    ref = np.percentile(m, 99.5)
    if ref <= 0: return np.array([], dtype=int)
    pk, _ = find_peaks(m, distance=int(0.25*FS), height=0.25*ref)
    return np.unique([max(0, p - W) + int(np.argmax(np.abs(f[max(0, p - W):p + 1]))) for p in pk])

def score(ref, det, tol=int(0.15*FS)):
    det = np.sort(det); used = np.zeros(len(det), bool); tp = 0
    for r in ref:
        i = np.searchsorted(det, r)
        c = [j for j in (i-1, i) if 0 <= j < len(det) and not used[j] and abs(det[j] - r) <= tol]
        if c:
            j = min(c, key=lambda j: abs(det[j] - r)); used[j] = True; tp += 1
    return tp, len(ref) - tp, len(det) - tp

# ------------- 5. data -------------
def load_record(r):
    sampto = SECONDS * FS if SECONDS else None
    rec = wfdb.rdrecord(f'mitdb/{r}', channels=[0], sampto=sampto)
    ann = wfdb.rdann(f'mitdb/{r}', 'atr', sampto=sampto)
    ref = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEATS])
    return rec.p_signal[:, 0], ref

missing = [r for r in RECORDS if not os.path.exists(f'mitdb/{r}.dat')]
if missing:
    wfdb.dl_database('mitdb', 'mitdb', records=missing)      # needs internet access

# ------------- 6. experiment -------------
combos = [('EXACT', 0)] + [(c, k) for c in APPROX_CELLS for k in KS]
tot = {cb: np.zeros(3) for cb in combos}
nrmse = {cb: [] for cb in combos}; corr = {cb: [] for cb in combos}
per_record = []; demo = None
t0 = time.time()
for r in RECORDS:
    sig, ref = load_record(r)
    f, q = preprocess(sig)
    m_ex = mwi(q, 'EXACT', 0)
    store = {('EXACT', 0): m_ex}; cache = {}
    for cb in combos:
        cell, k = cb; canon = 'LXCP' if cell == 'LXCP-G' else cell      # LXCP-G computes the same function
        if (canon, k) not in cache:
            m = m_ex if canon == 'EXACT' else mwi(q, canon, k)
            cache[(canon, k)] = (m, score(ref, detect(f, m)))
        m, (tp, fn, fp) = cache[(canon, k)]
        if k == DEMO_K or cell == 'EXACT': store[cb] = m
        tot[cb] += (tp, fn, fp)
        per_record.append(dict(record=r, cell=cell, k=k, TP=tp, FN=fn, FP=fp))
        nrmse[cb].append(np.sqrt(np.mean((m - m_ex) ** 2)) / max(m_ex.max(), 1))
        sd = m.std()
        corr[cb].append(np.corrcoef(m_ex, m)[0, 1] if sd > 0 else 0.0)
    if demo is None: demo = (f, ref, store)
    print(f"record {r} done  ({time.time() - t0:.0f} s)")

# ------------- 7. results table -------------
base_t = tree_toffolis('EXACT', 0)
rows = []
for cb in combos:
    cell, k = cb; tp, fn, fp = tot[cb]; tc = tree_toffolis(cell, k)
    rows.append(dict(cell=cell, k=k,
        Se=tp/(tp+fn), PPV=tp/(tp+fp) if tp+fp else 0.0, DER=(fp+fn)/(tp+fn),
        MWI_NRMSE=float(np.mean(nrmse[cb])), MWI_corr=float(np.mean(corr[cb])),
        Toffoli=tc, T_count=7*tc, Toffoli_saving_pct=100*(1 - tc/base_t)))
with open(f'{OUT}/qrs_summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
with open(f'{OUT}/qrs_per_record.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=per_record[0].keys()); w.writeheader(); w.writerows(per_record)

print(f"\n{'cell':7}{'k':>3}{'Se':>8}{'PPV':>8}{'DER':>8}{'NRMSE':>8}{'corr':>7}{'CCX':>7}{'save%':>7}")
for r in rows:
    print(f"{r['cell']:7}{r['k']:>3}{r['Se']:8.4f}{r['PPV']:8.4f}{r['DER']:8.4f}"
          f"{r['MWI_NRMSE']:8.3f}{r['MWI_corr']:7.3f}{r['Toffoli']:7d}{r['Toffoli_saving_pct']:7.1f}")

# ------------- 8. figures -------------
ex = rows[0]; cols = {c: f'C{i}' for i, c in enumerate(APPROX_CELLS)}
SHOW = [c for c in APPROX_CELLS if c != 'LXCP-G']
plt.figure(figsize=(6, 4))
for cell in SHOW:
    rr = [r for r in rows if r['cell'] == cell]
    plt.plot([r['k'] for r in rr], [r['DER'] for r in rr], 'o-', color=cols[cell], label=cell)
plt.axhline(ex['DER'], color='k', ls='--', label='Exact')
plt.xlabel('Approximate low bits k'); plt.ylabel('Detection error rate'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/der_vs_k.png', dpi=300); plt.show()

plt.figure(figsize=(6, 4))
for cell in APPROX_CELLS:
    rr = [r for r in rows if r['cell'] == cell]
    plt.plot([r['Toffoli'] for r in rr], [r['Se'] for r in rr], 'o-', color=cols[cell], label=cell)
plt.plot([ex['Toffoli']], [ex['Se']], 'k*', ms=12, label='Exact')
plt.xlabel('Toffoli count of the adder tree'); plt.ylabel('Sensitivity'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/sensitivity_vs_toffoli.png', dpi=300); plt.show()

f, ref, store = demo
a0 = 10 * FS if len(f) > 20 * FS else 0; a1 = min(len(f), a0 + 10 * FS); t = np.arange(a0, a1) / FS
def norm(m):
    m = m.astype(float); m = m - np.median(m); return m / max(np.percentile(m, 99.5), 1e-12)
fig, ax = plt.subplots(len(SHOW) + 1, 1, figsize=(8, 2*(len(SHOW) + 1)), sharex=True)
ax[0].plot(t, f[a0:a1], 'k', lw=.8); ax[0].set_ylabel('ECG (filtered)')
rb = ref[(ref >= a0) & (ref < a1)]; ax[0].plot(rb / FS, f[rb], 'rv', ms=5)
for i, cell in enumerate(SHOW, 1):
    ax[i].plot(t, norm(store[('EXACT', 0)])[a0:a1], color='0.6', label='Exact')
    if (cell, DEMO_K) in store:
        ax[i].plot(t, norm(store[(cell, DEMO_K)])[a0:a1], color=cols[cell], label=f'{cell}, k={DEMO_K}')
    ax[i].legend(loc='upper right', fontsize=7); ax[i].set_ylabel('MWI (norm.)')
ax[-1].set_xlabel('Time (s)'); plt.tight_layout(); plt.savefig(f'{OUT}/mwi_waveforms.png', dpi=300); plt.show()
print("saved to", OUT)
