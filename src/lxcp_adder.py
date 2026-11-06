# ======================================================================================
#  LXCP: Lower-XOR, Carry-Predicted approximate adder  (reversible, Qniverse/Qiskit-ready)
#
#  b <- approx(a + b)   (in place on the b register, carry-out on qubit z)
#    bits 0..k-1 : CNOT only        b_i <- a_i XOR b_i          (no carry logic, 0 Toffoli)
#    carry into bit k is PREDICTED  c_k = a_{k-1} AND b_{k-1}   (1 Toffoli)
#    bits k..n-1 : exact adder  (Cuccaro ripple-carry, or a 1-Toffoli-per-bit variant)
#  k = 0 -> exact adder.  k = 1 -> still exact (half-adder on bit 0).  k = n -> fully approximate.
#
#  Works with Qiskit if installed, otherwise with a tiny built-in circuit class.
# ======================================================================================
import csv
import numpy as np
from collections import Counter
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results'); FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

try:
    from qiskit import QuantumCircuit
    BACKEND = "qiskit"
except ImportError:
    BACKEND = "builtin"
    class _Op:
        def __init__(s, name): s.name = name
    class _Bit:
        def __init__(s, i): s.index = i
    class _Inst:
        def __init__(s, name, qubits): s.operation = _Op(name); s.qubits = qubits
    class QuantumCircuit:
        def __init__(s, nq, nc=0):
            s.num_qubits = nq; s.num_clbits = nc; s.data = []; s.measures = []
        def find_bit(s, q): return _Bit(q)
        def x(s, a): s.data.append(_Inst('x', [a]))
        def cx(s, a, b): s.data.append(_Inst('cx', [a, b]))
        def ccx(s, a, b, c): s.data.append(_Inst('ccx', [a, b, c]))
        def measure(s, qs, cs): s.measures += list(zip(qs, cs))
        def compose(s, other, inplace=True):
            s.data += other.data; s.measures += other.measures; return s
        def count_ops(s): return Counter(i.operation.name for i in s.data)
        def depth(s):
            lvl = [0] * s.num_qubits
            for i in s.data:
                d = max(lvl[q] for q in i.qubits) + 1
                for q in i.qubits: lvl[q] = d
            return max(lvl) if lvl else 0

def to_qasm(qc):
    if BACKEND == "qiskit":
        from qiskit import qasm2
        return qasm2.dumps(qc)
    L = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{qc.num_qubits}];',
         f'creg c[{max(qc.num_clbits, 1)}];']
    for i in qc.data:
        L.append(f"{i.operation.name} " + ",".join(f"q[{q}]" for q in i.qubits) + ";")
    for q, c in qc.measures: L.append(f"measure q[{q}] -> c[{c}];")
    return "\n".join(L)

# ---------------------------------- circuits ----------------------------------
# qubit layout: 0 = carry/prediction qubit | 1..n = a | n+1..2n = b | 2n+1 = carry-out z | 2n+2.. = extra
def maj(qc, x, y, z): qc.cx(z, y); qc.cx(z, x); qc.ccx(x, y, z)
def uma(qc, x, y, z): qc.ccx(x, y, z); qc.cx(z, x); qc.cx(x, y)

def build_adder(n, k, predict=True, garbage=False, clean=True):
    """predict=False      -> plain lower-XOR adder (no carry prediction)
       garbage=False      -> upper part = Cuccaro adder (2 Toffoli/bit, all ancillas restored)
       garbage=True       -> upper part = carry-ancilla ripple (1 Toffoli/bit), carry qubits left dirty:
                             use only when the result is measured right away
       clean=True         -> uncompute the prediction qubit (+1 Toffoli); clean=False leaves it set"""
    A = lambda i: 1 + i
    B = lambda i: n + 1 + i
    C0, Z = 0, 2*n + 1
    extra = max(n - k - 1, 0) if garbage else 0
    qc = QuantumCircuit(2*n + 2 + extra)
    if k >= 1 and predict:                                   # carry prediction (before b_{k-1} is changed)
        qc.ccx(A(k-1), B(k-1), Z if k == n else C0)
    for i in range(k):                                       # approximate low bits: CNOT only
        qc.cx(A(i), B(i))
    if k < n:
        if not garbage:                                      # exact upper part, Cuccaro
            carry = C0
            for i in range(k, n):
                maj(qc, carry, B(i), A(i)); carry = A(i)
            qc.cx(A(n-1), Z)
            for i in reversed(range(k, n)):
                uma(qc, C0 if i == k else A(i-1), B(i), A(i))
            if k >= 1 and predict and clean:                 # C0 = ab; with s = a^b: ab ^ a.s ^ a = 0
                qc.ccx(A(k-1), B(k-1), C0); qc.cx(A(k-1), C0)
        else:                                                # exact upper part, 1 Toffoli per bit
            carries = {k: C0, n: Z}
            for j in range(k + 1, n): carries[j] = 2*n + 2 + (j - k - 1)
            for i in range(k, n):
                c, cn = carries[i], carries[i+1]
                qc.cx(A(i), B(i)); qc.cx(A(i), c); qc.ccx(B(i), c, cn)
                qc.cx(A(i), cn); qc.cx(A(i), c); qc.cx(c, B(i))
    return qc

# ---------------------------- exact reversible simulation ----------------------------
U = np.uint64
def run(qc, n, a, b):
    """simulate on arrays a, b; returns (result, final_state)"""
    a = np.asarray(a, dtype=np.uint64); b = np.asarray(b, dtype=np.uint64)
    st = (a << U(1)) | (b << U(n + 1))
    for inst in qc.data:
        g = inst.operation.name; q = [qc.find_bit(x).index for x in inst.qubits]
        if   g == 'x':   st ^= U(1) << U(q[0])
        elif g == 'cx':  st ^= ((st >> U(q[0])) & U(1)) << U(q[1])
        elif g == 'ccx': st ^= (((st >> U(q[0])) & (st >> U(q[1]))) & U(1)) << U(q[2])
    m = U(2**n - 1)
    res = ((st >> U(n + 1)) & m) + (((st >> U(2*n + 1)) & U(1)) << U(n))
    return res.astype(np.int64), st

# ---------------------- fast behavioural models (used for the ECG study) ----------------------
def lxcp_add(x, y, width, k):
    x = np.asarray(x, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    k = min(k, width)
    if k == 0: return x + y
    low = (x ^ y) & ((1 << k) - 1)
    pred = (x >> (k - 1)) & (y >> (k - 1)) & 1
    return low | (((x >> k) + (y >> k) + pred) << k)

def lxa_add(x, y, width, k):                                 # lower-XOR without prediction
    x = np.asarray(x, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    k = min(k, width)
    return (x + y) if k == 0 else (((x ^ y) & ((1 << k) - 1)) | (((x >> k) + (y >> k)) << k))

# ------------------------------------ metrics ------------------------------------
def gate_stats(qc):
    o = qc.count_ops(); ccx, cx, x = o.get('ccx', 0), o.get('cx', 0), o.get('x', 0)
    return dict(CCX=ccx, CX=cx, T_count=7*ccx, Clifford_T_total=15*ccx + cx + x,   # CCX = 7 T + 6 CX + 2 H
                depth=qc.depth(), qubits=qc.num_qubits)

def error_metrics(res, n):
    r = np.arange(2**n); ex = (r[:, None] + r[None, :]).ravel(); e = res - ex; ae = np.abs(e); nz = ex > 0
    return dict(ER=(ae > 0).mean(), MED=ae.mean(), NMED=ae.mean()/(2**(n+1) - 1),
                MRED=(ae[nz]/ex[nz]).mean(), MaxED=int(ae.max()), Bias=e.mean())

if __name__ == "__main__":
    n = 8
    r = np.arange(2**n); AA, BB = np.meshgrid(r, r, indexing='ij'); AA, BB = AA.ravel(), BB.ravel()

    # ---- 1. correctness: every variant, every k, all 65,536 inputs ----
    for k in range(n + 1):
        ref = lxcp_add(AA, BB, n, k); ref_x = lxa_add(AA, BB, n, k)
        for garbage in (False, True):
            for clean in ((True, False) if not garbage else (False,)):
                qc = build_adder(n, k, True, garbage, clean)
                res, st = run(qc, n, AA, BB)
                assert (res == ref).all(), ("LXCP", k, garbage, clean)
                assert (((st >> U(1)) & U(2**n - 1)).astype(np.int64) == AA).all()      # a register restored
                if not garbage and clean and k < n: assert ((st & U(1)) == 0).all()      # prediction qubit cleaned
        assert (run(build_adder(n, k, predict=False), n, AA, BB)[0] == ref_x).all(), ("LXA", k)
    assert (run(build_adder(n, 0), n, AA, BB)[0] == AA + BB).all()
    assert (lxcp_add(AA, BB, n, 1) == AA + BB).all()                                     # k = 1 is exact
    rng = np.random.default_rng(1)                                                       # wider adders (used in the ECG tree)
    for w in (9, 10, 13):
        xa = rng.integers(0, 2**w, 20000); xb = rng.integers(0, 2**w, 20000)
        for k in (0, 3, 6, w):
            assert (run(build_adder(w, k), w, xa, xb)[0] == lxcp_add(xa, xb, w, k)).all(), (w, k)
    print(f"all circuits verified (backend: {BACKEND})")

    # ---- 2. comparison table (8-bit, exhaustive) ----
    designs = [("Exact Cuccaro", dict(predict=False), 0)]
    rows = []
    def add_row(name, k, **kw):
        qc = build_adder(n, k, **kw); res = run(qc, n, AA, BB)[0]
        rows.append(dict(design=name, k=k, **error_metrics(res, n), **gate_stats(qc)))
    add_row("Exact (Cuccaro)", 0, predict=False)
    for k in range(1, n + 1):
        add_row("LXA (no prediction)", k, predict=False)
        add_row("LXCP", k, predict=True, garbage=False, clean=True)
        add_row("LXCP-G (garbage carries)", k, predict=True, garbage=True, clean=False)
    with open(os.path.join(RES, "lxcp_8bit_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    base = rows[0]
    print(f"\n{'design':26}{'k':>3}{'ER':>8}{'MED':>8}{'NMED':>9}{'MaxED':>6}{'CCX':>5}{'CX':>5}{'T':>5}{'total':>7}{'depth':>6}{'qb':>4}{'CCX save':>9}")
    for r_ in rows:
        print(f"{r_['design']:26}{r_['k']:>3}{r_['ER']:8.4f}{r_['MED']:8.2f}{r_['NMED']:9.5f}{r_['MaxED']:>6}"
              f"{r_['CCX']:>5}{r_['CX']:>5}{r_['T_count']:>5}{r_['Clifford_T_total']:>7}{r_['depth']:>6}{r_['qubits']:>4}"
              f"{100*(1 - r_['CCX']/base['CCX']):8.1f}%")

    # ---- 3. circuit for the Qniverse Code Editor (4-bit example, 11 + 7, k = 2: exact 18, this adder gives 16) ----
    def platform_circuit(n, k, a, b, **kw):
        core = build_adder(n, k, **kw)
        qc = QuantumCircuit(core.num_qubits, n + 1)
        for i in range(n):
            if (a >> i) & 1: qc.x(1 + i)
            if (b >> i) & 1: qc.x(n + 1 + i)
        qc.compose(core, inplace=True)
        qc.measure([n + 1 + i for i in range(n)] + [2*n + 1], range(n + 1))   # b register + carry-out
        return qc
    print("\n" + to_qasm(platform_circuit(4, 2, 11, 7)))
