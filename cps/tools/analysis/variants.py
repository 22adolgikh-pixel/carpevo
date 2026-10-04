import os as _o
import sys,inspect,pickle,re,numpy as np,cv2
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))
import autogrid as A
from bench import score
D=pickle.load(open('/tmp/cpsan_bench.pkl','rb'))
src=inspect.getsource(A.clean)
def make(kern,fill_gap=True,name='clean_v'):
    s=src.replace("def clean(M, t):","def %s(M, t):"%name)
    s=s.replace("n, lab, stats, _ = cv2.connectedComponentsWithStats((1 - dark).astype(np.uint8), connectivity=4)\n    outside",
        "dark_c = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, KERN) if KERN is not None else dark\n    n, lab, stats, _ = cv2.connectedComponentsWithStats((1 - dark_c).astype(np.uint8), connectivity=4)\n    outside")
    ns=dict(A.__dict__);ns['KERN']=kern
    exec(s,ns);f=ns[name]
    if not fill_gap: return f
    def g(M,t):
        out=f(M,t);gap=(cv2.morphologyEx((M==1).astype(np.uint8),cv2.MORPH_CLOSE,kern)>0)&(M!=1) if kern is not None else None
        if gap is None: return out
        gray=(out==2).astype(np.float32);nb=cv2.filter2D(gray,-1,np.ones((3,3),np.float32),borderType=cv2.BORDER_CONSTANT)
        out=out.copy();out[gap&(nb>=3)]=2;return out
    return g
print('база       :',score(D,A.clean))
for nm,k in(('closing 3x3 rect',np.ones((3,3),np.uint8)),('closing cross',cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))),('closing 2x2',np.ones((2,2),np.uint8))):
    print(nm.ljust(11),':',score(D,make(k)))
print('--- линейные разрывы')
def line_close(dark):
    h=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8));v=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8));return h|v
src2=src.replace("def clean(M, t):","def clean_l(M, t):").replace("n, lab, stats, _ = cv2.connectedComponentsWithStats((1 - dark).astype(np.uint8), connectivity=4)\n    outside",
 "dark_c = LC(dark)\n    n, lab, stats, _ = cv2.connectedComponentsWithStats((1 - dark_c).astype(np.uint8), connectivity=4)\n    outside")
ns=dict(A.__dict__);ns['LC']=line_close;exec(src2,ns);cl=ns['clean_l']
def cl_fill(M,t):
    out=cl(M,t);gap=(line_close((M==1).astype(np.uint8))>0)&(M!=1);gray=(out==2).astype(np.float32);nb=cv2.filter2D(gray,-1,np.ones((3,3),np.float32),borderType=cv2.BORDER_CONSTANT);out=out.copy();out[gap&(nb>=3)]=2;return out
print('линейное закрытие :',score(D,cl_fill))
print('линейное закрытие без дозаливки:',score(D,cl))
# где вылезает лишнее: по фигурам
import numpy as np
bad=[]
for d in D:
    T=d['T'];ok=T>=0
    if not (T==2).any(): continue
    a=A.clean(d['M0'].copy(),d['t']);b=cl_fill(d['M0'].copy(),d['t'])
    ea=((a==2)&(T!=2)&ok).sum();eb=((b==2)&(T!=2)&ok).sum()
    if eb-ea>30: bad.append((d['id'],int(ea),int(eb),int((T==2).sum())))
print(len(bad),sorted(bad,key=lambda x:x[1]-x[2])[:8])
