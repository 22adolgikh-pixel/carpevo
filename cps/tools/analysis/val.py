import os as _o
import sys,os,json,numpy as np,cv2
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)));sys.path.insert(0,'/tmp')
import autogrid as A,distortion as D,learn
from drift import SP
dv=json.load(open('/tmp/cpsan_dist_vals.json'));X=[];Y=[]
for fid,a in dv.items():
    w=json.load(open(SP+'/work/'+fid+'.json'));g=cv2.imread(SP+'/crops/'+fid+'.png',0)
    fr=w.get('frame') or [0,0,g.shape[1],g.shape[0]]
    try:G,C,M,info=A.auto_figure(g,fr,dark_only=True,use_model=False)
    except Exception: continue
    U=(learn.matrix_labels(w)>0);Cu,_=learn.final_centers(w)
    nodes=G['nodes'];Hi=cv2.getPerspectiveTransform(np.array(G['corners'],np.float32),np.array([[0,0],[G['cols'],0],[G['cols'],G['rows']],[0,G['rows']]],np.float32))
    uv=cv2.perspectiveTransform(Cu.reshape(-1,1,2).astype(np.float32),Hi).reshape(Cu.shape[:2]+(2,))
    ii=np.floor(uv[...,0]).astype(int);jj=np.floor(uv[...,1]).astype(int)
    ok=(ii>=0)&(ii<M.shape[1])&(jj>=0)&(jj<M.shape[0])
    if ok.mean()<.9: continue
    P=np.zeros(U.shape,bool);P[ok]=M[jj[ok],ii[ok]]>0
    X.append(a['dist']);Y.append(float((P!=U).mean()))
X=np.array(X);Y=np.array(Y);from scipy.stats import spearmanr
print(len(X),'spearman',spearmanr(X,Y))
q=np.quantile(X,[.5,.8]);
for lo,hi,name in((0,q[0],'низкая'),(q[0],q[1],'средняя'),(q[1],9,'высокая')):
    m=(X>=lo)&(X<=hi);print(name,m.sum(),'ошибка медиана %.3f среднее %.3f'%(np.median(Y[m]),Y[m].mean()))
