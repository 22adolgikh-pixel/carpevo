import os as _o
import sys,json,numpy as np,cv2
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))
import autogrid as A,learn
import os as _o; B=_o.environ.get('CPS_ANALYSIS_DATA','/tmp/cd/data')
def render(fid):
    w=json.load(open(B+'/work/'+fid+'.json'));g=cv2.imread(B+'/crops/'+fid+'.png',0)
    U=learn.matrix_labels(w);Cu,_=learn.final_centers(w)
    G,C,M,info=A.auto_figure(g,w.get('frame') or [0,0,g.shape[1],g.shape[0]],use_model=False,dark_only=False)
    pal=np.array([[255,255,255],[30,30,30],[170,170,170]],np.uint8)
    s=8;a=cv2.resize(pal[M],None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST);b=cv2.resize(pal[U],None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
    h=420;im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR);im=cv2.resize(im,(int(im.shape[1]*h/im.shape[0]),h))
    f=lambda x:cv2.resize(x,(int(x.shape[1]*h/x.shape[0]),h),interpolation=cv2.INTER_NEAREST)
    out=np.hstack([im,np.full((h,6,3),128,np.uint8),f(a),np.full((h,6,3),128,np.uint8),f(b)])
    cv2.imwrite('/tmp/cpsan_v_%s.png'%fid,out)
for f in sys.argv[1:]: render(f)
