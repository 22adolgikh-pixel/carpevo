import sys, json, cv2, numpy as np, glob
sys.path.insert(0,'/home/claude/cps_wt/cps')
import rug_rect, rug_mesh
exec(open('run.py').read().split('rows=[]')[0].split("os.makedirs('out',exist_ok=True)")[1])
Q=json.load(open('quads.json'))
for i,q in Q.items():
    img=cv2.imread(glob.glob(f'img/*{q["hash"]}.jpg')[0])
    a=rug_rect.rectify(img,q['quad'],1200); b,_=rug_mesh.mesh_rectify(img,q['quad'],1200); c,info=rug_mesh.mesh_rectify2(img,q['quad'],1200)
    s=[sym(x) for x in (a,b,c)]
    print(i,'field',None if not info['field'] else {k:round(v,1) for k,v in info['field']['range_pct'].items()},'symLR %.1f/%.1f/%.1f symUD %.1f/%.1f/%.1f'%(s[0][0],s[1][0],s[2][0],s[0][1],s[1][1],s[2][1]))
    H=800; r=lambda x: cv2.resize(x,(int(x.shape[1]*H/x.shape[0]),H))
    cv2.imwrite(f'out/{i}_3.jpg',np.hstack([r(a),np.full((H,8,3),255,np.uint8),r(b),np.full((H,8,3),255,np.uint8),r(c)]))
