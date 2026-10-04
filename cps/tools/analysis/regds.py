import os as _o
import sys,os,json,numpy as np,cv2,pickle,time
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))
import autogrid as A,learn
import os as _o; B=_o.environ.get('CPS_ANALYSIS_DATA','/tmp/cd/data')
def mapU(G,Cu,U):
    Hi=cv2.getPerspectiveTransform(np.array(G['corners'],np.float32),np.array([[0,0],[G['cols'],0],[G['cols'],G['rows']],[0,G['rows']]],np.float32))
    uv=cv2.perspectiveTransform(Cu.reshape(-1,1,2).astype(np.float32),Hi).reshape(Cu.shape[:2]+(2,))
    ii=np.floor(uv[...,0]).astype(int);jj=np.floor(uv[...,1]).astype(int)
    T=np.full((G['rows'],G['cols']),-1,np.int8);ok=(ii>=0)&(ii<G['cols'])&(jj>=0)&(jj<G['rows'])
    T[jj[ok],ii[ok]]=U[ok]; return T   # -1 = вне эталона
def regions(M,t):
    dark=(M==1).astype(np.uint8);H,W=M.shape
    nd4=cv2.filter2D(dark.astype(np.float32),-1,np.array([[0,1,0],[1,0,1],[0,1,0]],np.float32),borderType=cv2.BORDER_CONSTANT)
    tc=t+A.BLEED*nd4;near=cv2.dilate(dark,np.ones((3,3),np.uint8))>0
    n,lab,stats,_=cv2.connectedComponentsWithStats((1-dark).astype(np.uint8),connectivity=4)
    outside=np.zeros_like(dark,bool);regs=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i];comp=lab==i
        if x==0 or y==0 or x+w==W or y+h==H: outside|=comp;continue
        inner=comp&~near;src=inner if inner.sum()>=3 else comp
        regs.append(dict(comp=comp,tm=float(np.median(tc[src])),area=int(area),std=float(np.std(tc[src])),q10=float(np.percentile(tc[src],10)),q90=float(np.percentile(tc[src],90)),
            raw=float(np.median(t[src])),bb=(x,y,w,h),nin=int(inner.sum())))
    o=outside&~near;ref=float(np.median(tc[o])) if o.sum()>=15 else 1.0
    refsd=float(np.std(tc[o])) if o.sum()>=15 else 0.0
    return regs,ref,refsd,tc,dark
def feats(regs,ref,refsd,tc,dark):
    F=[];tms=np.array([r['tm'] for r in regs]);ars=np.array([r['area'] for r in regs],float);H,W=dark.shape
    for r in regs:
        x,y,w,h=r['bb'];d=ref-r['tm']
        others=[ref-q for q in tms]
        F.append([d,r['tm'],ref,refsd,r['std'],ref-r['q10'],ref-r['q90'],np.log1p(r['area']),r['area']/(H*W),r['nin']/max(1,r['area']),
            w/W,h/H,(x+w/2)/W,(y+h/2)/H,len(regs),d-min(others),d-max(others),float(np.sum(ars)>0 and r['area']/ars.sum()),
            float(dark.mean()),float(np.average(ref-tms,weights=ars)),float(np.median(ref-tms)),sum(1 for q in others if q>0.1)/len(regs)])
    return np.array(F,np.float32)
if __name__=='__main__':
    X=[];Y=[];GRP=[];FIG=[];AREA=[];t0=time.time()
    for fn in sorted(os.listdir(B+'/work')):
        try:w=json.load(open(B+'/work/'+fn))
        except Exception: continue
        fid=fn[:-5]
        if not(w.get('done') and (w.get('auto') or {}).get('reviewed') and not w.get('color_mode') and w.get('matrix') and w.get('quad') and w.get('grid')): continue
        g=cv2.imread(B+'/crops/'+fid+'.png',0)
        if g is None: continue
        try:
            U=learn.matrix_labels(w);Cu,_=learn.final_centers(w)
            G=A.detect_grid(g,w.get('frame') or [0,0,g.shape[1],g.shape[0]]);C=A.cell_centers(G);v=A.cell_means(g,C,min(G['px'],G['py']));M,info=A.classify(v);t=info['t']
            regs,ref,refsd,tc,dark=regions(M,t)
            if not regs: continue
            T=mapU(G,Cu,U);F=feats(regs,ref,refsd,tc,dark)
            for r,f in zip(regs,F):
                tt=T[r['comp']];tt=tt[tt>=0]
                if len(tt)<max(2,0.3*r['area']): continue
                X.append(f);Y.append(float((tt==2).mean()));GRP.append(w.get('sheet') or fid.split('_')[0]);FIG.append(fid);AREA.append(r['area'])
        except Exception as e: print('fail',fid,e)
    pickle.dump(dict(X=np.array(X),Y=np.array(Y),G=GRP,F=FIG,A=np.array(AREA)),open('/tmp/cpsan_regds.pkl','wb'));print(len(X),round(time.time()-t0))
