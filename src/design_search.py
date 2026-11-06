import numpy as np, itertools
n=8
r=np.arange(2**n); A,B=np.meshgrid(r,r,indexing='ij'); A=A.ravel(); B=B.ravel(); EX=A+B
def bit(x,i): return (x>>i)&1
def build(k, cs, up):
    """cs[i] for i=1..k-1 in {'0','a','b'}: carry approximation into low bit i (CX only); up in {'and','a','0'}: carry into bit k"""
    res=np.zeros_like(A)
    for i in range(k):
        c=0 if i==0 else {'0':0,'a':bit(A,i-1),'b':bit(B,i-1)}[cs[i-1]]
        res|=(bit(A,i)^bit(B,i)^c)<<i
    cin={'and':bit(A,k-1)&bit(B,k-1),'a':bit(A,k-1),'0':0}[up]
    return res | (((A>>k)+(B>>k)+cin)<<k)
for k in (3,4,5,6):
    out=[]
    for cs in itertools.product('0ab',repeat=k-1):
        for up in ('and','a','0'):
            e=EX-build(k,cs,up); out.append((np.abs(e).mean(), np.abs(e).max(), e.mean(), ''.join(cs), up, (e!=0).mean()))
    base=[o for o in out if o[3]=='0'*(k-1) and o[4]=='and'][0]
    xor0=[o for o in out if o[3]=='0'*(k-1) and o[4]=='0'][0]
    print(f"k={k}  LXCP(and,XOR): MED={base[0]:.2f} Max={base[1]} bias={base[2]:.2f} ER={base[5]:.3f} | LXA: MED={xor0[0]:.2f}")
    z=sorted([o for o in out if o[4]!='and'], key=lambda o:o[0])[:3]
    print("   best 0-Toffoli:", [(round(o[0],2),o[1],round(o[2],2),o[3],o[4]) for o in z])
    z=sorted([o for o in out if o[4]=='and'], key=lambda o:o[0])[:3]
    print("   best with AND-predict:", [(round(o[0],2),o[1],round(o[2],2),o[3],o[4]) for o in z])
