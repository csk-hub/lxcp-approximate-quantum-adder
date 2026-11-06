import numpy as np, json
import lxcp_adder as lx
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results'); FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
U=np.uint64
def noisy_run(qc, n, a, b, p, rng):
    """Pauli-error Monte Carlo on a classical reversible circuit. Only X-type components of Pauli errors change
    computational-basis outcomes, so each gate is followed by a random non-identity Pauli (prob. p_gate) and its X part is applied."""
    N=len(a); st=(a.astype(np.uint64)<<U(1))|(b.astype(np.uint64)<<U(n+1))
    for inst in qc.data:
        g=inst.operation.name; q=[qc.find_bit(x).index for x in inst.qubits]
        if g=='x': st^=U(1)<<U(q[0]); pg=p/10; npaul=4
        elif g=='cx': st^=((st>>U(q[0]))&U(1))<<U(q[1]); pg=p; npaul=16
        else: st^=(((st>>U(q[0]))&(st>>U(q[1])))&U(1))<<U(q[2]); pg=min(1.0,6*p); npaul=64   # Toffoli ~ 6 CNOTs
        hit=np.nonzero(rng.random(N)<pg)[0]
        if len(hit)==0: continue
        idx=rng.integers(1,npaul,size=len(hit))
        k=len(q)
        for j,qq in enumerate(q):
            pj=(idx//(4**(k-1-j)))%4                 # 0=I,1=X,2=Y,3=Z
            flip=((pj==1)|(pj==2)).astype(np.uint64)
            st[hit]^=flip<<U(qq)
    m=U(2**n-1)
    return (((st>>U(n+1))&m)+(((st>>U(2*n+1))&U(1))<<U(n))).astype(np.int64)

n=8; N=60000
rng=np.random.default_rng(2024)
a=rng.integers(0,2**n,N); b=rng.integers(0,2**n,N); ex=a+b
designs=[("Exact (Cuccaro-type)",dict(k=0,predict=False,garbage=False,clean=True)),
         ("Exact (1 Toffoli/bit)",dict(k=1,predict=True,garbage=True,clean=False)),
         ("LXCP k=4",dict(k=4,predict=True,garbage=False,clean=True)),
         ("LXCP-G k=4",dict(k=4,predict=True,garbage=True,clean=False)),
         ("LXCP k=6",dict(k=6,predict=True,garbage=False,clean=True)),
         ("LXCP-G k=6",dict(k=6,predict=True,garbage=True,clean=False))]
ps=[0,1e-4,3e-4,1e-3,3e-3,1e-2,2e-2,3e-2]
out=[]
for name,kw in designs:
    k=kw.pop('k'); qc=lx.build_adder(n,k,**kw)
    ideal=lx.run(qc,n,a,b)[0]
    for p in ps:
        res=noisy_run(qc,n,a,b,p,rng) if p>0 else ideal
        e=np.abs(res-ex)
        out.append(dict(design=name,p=p,NMED=float(e.mean()/(2**(n+1)-1)),MED=float(e.mean()),P_exact=float((res==ex).mean()),
                        P_within4=float((e<=4).mean()), noise_MED=float(np.abs(res-ideal).mean())))
json.dump(out,open(os.path.join(RES,'noise_results.json'),'w'))
import collections
print(f"{'design':24}{'p':>8}{'MED':>9}{'NMED':>10}{'P(exact)':>10}{'P(|e|<=4)':>11}")
for o in out: print(f"{o['design']:24}{o['p']:8.0e}{o['MED']:9.2f}{o['NMED']:10.5f}{o['P_exact']:10.3f}{o['P_within4']:11.3f}")
