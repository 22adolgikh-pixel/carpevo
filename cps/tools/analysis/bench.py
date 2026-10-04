import os as _o
import sys,os,json,numpy as np,cv2,pickle
sys.path.insert(0,_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),'..','..')); sys.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))
import autogrid as A,learn
from regds import mapU
import os as _o; B=_o.environ.get('CPS_ANALYSIS_DATA','/tmp/cd/data');lost=set(__import__('pandas').read_csv('/tmp/cpsan_lost_gray.csv').id)
def build():
    D=[]
    for fn in sorted(os.listdir(B+'/work')):
        try:w=json.load(open(B+'/work/'+fn))
        except Exception: continue
        fid=fn[:-5]
        if not(w.get('done') and (w.get('auto') or {}).get('reviewed') and not w.get('color_mode') and w.get('matrix') and w.get('quad') and w.get('grid')): continue
        g=cv2.imread(B+'/crops/'+fid+'.png',0)
        if g is None: continue
        try:
            U=learn.matrix_labels(w);Cu,_=learn.final_centers(w)
            G=A.detect_grid(g,w.get('frame') or [0,0,g.shape[1],g.shape[0]]);C=A.cell_centers(G);v=A.cell_means(g,C,min(G['px'],G['py']));M0,info=A.classify(v)
            D.append(dict(id=fid,M0=M0,t=info['t'],T=mapU(G,Cu,U)))
        except Exception as e: print('fail',fid,e)
    pickle.dump(D,open('/tmp/cpsan_bench.pkl','wb'));return D
def score(D,clean,only_gray=True):
    miss=extra=tg=ink=n=0;badfig=0;nf=0
    for d in D:
        T=d['T'];ok=T>=0
        has=(T==2).any()
        if only_gray and (not has): continue
        if d['id'] in lost: continue
        M=clean(d['M0'].copy(),d['t']);tgc=(T==2)&ok;pg=(M==2)&ok
        miss+=(tgc&~pg).sum();extra+=(pg&~tgc).sum();tg+=tgc.sum();ink+=(((M==1)!=(T==1))&ok).sum();n+=ok.sum()
        e=((M!=T)&ok).mean();nf+=1;badfig+=e>.1
    return 'пропущено серого %.1f%% | лишнего серого %.1f%% от объёма серого | ошибка контура %.2f%% клеток | рисунков с ошибкой>10%%: %d из %d | всего ошибок %.2f%%'%(100*miss/tg,100*extra/tg,100*ink/n,badfig,nf,100*(miss+extra+ink)/n)
if __name__=='__main__':
    if not os.path.exists('/tmp/cpsan_bench.pkl'): D=build()
    else: D=pickle.load(open('/tmp/cpsan_bench.pkl','rb'))
    print(len(D),'рисунков в стенде')
    print('текущий clean:',score(D,A.clean))
