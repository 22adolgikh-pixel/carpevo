import os as _o
import sys,pickle,numpy as np,cv2
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))
import autogrid as A
from regds import regions
from sklearn.metrics import roc_auc_score
D=pickle.load(open('/tmp/cpsan_bench.pkl','rb'))
lost=set(__import__('pandas').read_csv('/tmp/cpsan_lost_gray.csv').id)
rows=[]
for d in D:
    if d['id'] in lost: continue
    T=d['T'];ok=T>=0;M0=d['M0'];t=d['t']
    M=A.clean(M0.copy(),t);err=((M!=T)&ok).mean()
    regs,ref,rsd,tc,dark=regions(M0,t)
    closed=np.zeros(M0.shape,bool)
    for r in regs: closed|=r['comp']
    near=cv2.dilate((M0==1).astype(np.uint8),np.ones((3,3),np.uint8))>0
    outc=(~closed)&(~near)&(M0!=1)
    dd=ref-tc
    leak=float((outc&(dd>0.10)).sum())/max(1,(M0!=1).sum())        # доля клеток «снаружи», заметно темнее бумаги
    amb=sum(r['area'] for r in regs if 0.07<ref-r['tm']<0.22)/max(1,(M0!=1).sum())   # доля площади областей у границы решения
    gfrac=float((M==2).mean())
    rows.append((d['id'],err,leak,amb,gfrac,(T==2).any()))
import pandas as pd
r=pd.DataFrame(rows,columns=['id','err','leak','amb','gfrac','hasg']);r['bad']=r.err>.1
print(len(r),'плохих (>10%%):',r.bad.sum())
for c in('leak','amb','gfrac'): print(c,'AUC',round(roc_auc_score(r.bad,r[c]),3))
r['risk']=r.leak+r.amb
print('risk AUC',round(roc_auc_score(r.bad,r.risk),3))
for q in(.7,.8,.9):
    th=r.risk.quantile(q);m=r.risk>=th;print('топ %d%% по риску: %d рисунков, ловит %d из %d плохих (%.0f%%), среди них плохих %.0f%%'%(100-100*q,m.sum(),(m&r.bad).sum(),r.bad.sum(),100*(m&r.bad).sum()/r.bad.sum(),100*(m&r.bad).sum()/m.sum()))
r.to_csv('risk.csv',index=False)
