import os as _o
import sys,os,json,numpy as np,cv2,csv,time
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))
import autogrid as A,learn
import os as _o; B=_o.environ.get('CPS_ANALYSIS_DATA','/tmp/cd/data');rows=[];t0=time.time()
def mapped(M,G,Cu):
    Hi=cv2.getPerspectiveTransform(np.array(G['corners'],np.float32),np.array([[0,0],[G['cols'],0],[G['cols'],G['rows']],[0,G['rows']]],np.float32))
    uv=cv2.perspectiveTransform(Cu.reshape(-1,1,2).astype(np.float32),Hi).reshape(Cu.shape[:2]+(2,))
    ii=np.floor(uv[...,0]).astype(int);jj=np.floor(uv[...,1]).astype(int)
    ok=(ii>=0)&(ii<M.shape[1])&(jj>=0)&(jj<M.shape[0]);P=np.zeros(Cu.shape[:2],np.uint8);P[ok]=M[jj[ok],ii[ok]];return P,ok.mean()
for fn in sorted(os.listdir(B+'/work')):
    try:w=json.load(open(B+'/work/'+fn))
    except Exception: continue
    fid=fn[:-5]
    if not(w.get('done') and (w.get('auto') or {}).get('reviewed') and not w.get('color_mode') and w.get('matrix') and w.get('quad') and w.get('grid')): continue
    g=cv2.imread(B+'/crops/'+fid+'.png',0)
    if g is None: continue
    try:
        U=learn.matrix_labels(w);Cu,_=learn.final_centers(w)
        fr=w.get('frame') or [0,0,g.shape[1],g.shape[0]]
        r={'id':fid,'sheet':w.get('sheet'),'h':U.shape[0],'w':U.shape[1],'gray_true':float((U==2).mean()),'ink_true':float((U==1).mean()),'edited':bool((w.get('auto') or {}).get('edited'))}
        for name,kw in(('nm',dict(use_model=False,dark_only=False)),('md',dict(use_model=True,dark_only=False))):
            G,C,M,info=A.auto_figure(g,fr,**kw);P,cov=mapped(M,G,Cu)
            r[name+'_err3']=float((P!=U).mean());r[name+'_errink']=float(((P>0)!=(U>0)).mean());r[name+'_cov']=float(cov)
            r[name+'_gray_miss']=float(((P!=2)&(U==2)).sum()/max(1,(U==2).sum()));r[name+'_gray_extra']=float(((P==2)&(U!=2)).sum()/max(1,(U!=2).sum()))
            if name=='md': r['conf']=info['confidence'];r['nflags']=len(info['flags']);r['flags']='|'.join(f[:25] for f in info['flags']);r['dist']=(info.get('distortion') or {}).get('dist');r['pitch']=G['px'];r['gridw']=G['cols'];r['gridh']=G['rows']
        rows.append(r)
    except Exception as e: print('fail',fid,e)
    if len(rows)%100==0: print(len(rows),round(time.time()-t0),flush=True)
with open('/tmp/cpsan_ev.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)
print('done',len(rows))
import pandas as _pd
_d=_pd.read_csv('/tmp/cpsan_ev.csv');_d[(_d.gray_true==0)&(_d.md_gray_extra>.10)][['id']].to_csv('/tmp/cpsan_lost_gray.csv',index=False)
