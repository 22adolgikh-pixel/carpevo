import os as _o
import sys,os,json,numpy as np,cv2
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))
import autogrid as A,learn
from regds import mapU,regions
import os as _o; B=_o.environ.get('CPS_ANALYSIS_DATA','/tmp/cd/data');tot=dict(miss_closed=0,miss_outside=0,miss_other=0,fp_closed=0,fp_outside=0,gray_true=0,cells=0,ink_err=0)
per=[]
for fn in sorted(os.listdir(B+'/work')):
    try:w=json.load(open(B+'/work/'+fn))
    except Exception: continue
    fid=fn[:-5]
    if not(w.get('done') and (w.get('auto') or {}).get('reviewed') and not w.get('color_mode') and w.get('matrix') and w.get('quad') and w.get('grid')): continue
    g=cv2.imread(B+'/crops/'+fid+'.png',0)
    if g is None: continue
    try:
        U=learn.matrix_labels(w);Cu,_=learn.final_centers(w)
        G=A.detect_grid(g,w.get('frame') or [0,0,g.shape[1],g.shape[0]]);C=A.cell_centers(G);v=A.cell_means(g,C,min(G['px'],G['py']));M0,info=A.classify(v);t=info['t']
        regs,ref,rsd,tc,dark=regions(M0,t);closed=np.zeros(M0.shape,bool)
        for r in regs: closed|=r['comp']
        M=A.clean(M0.copy(),t);T=mapU(G,Cu,U);ok=T>=0
        tg=(T==2)&ok;pg=(M==2)&ok
        tot['gray_true']+=tg.sum();tot['cells']+=ok.sum()
        tot['miss_closed']+=(tg&~pg&closed).sum();tot['miss_outside']+=(tg&~pg&~closed&(M0!=1)).sum();tot['miss_other']+=(tg&~pg&(M0==1)).sum()
        tot['fp_closed']+=(pg&~tg&closed).sum();tot['fp_outside']+=(pg&~tg&~closed).sum()
        tot['ink_err']+=(((M==1)!=(T==1))&ok).sum()
    except Exception as e: print('fail',fid,e)
print({k:int(v) for k,v in tot.items()})
c=tot['cells'];print({k:round(100*v/c,2) for k,v in tot.items() if k not in('cells',)}, '(% от всех клеток)')
