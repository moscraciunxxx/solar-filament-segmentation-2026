import json,numpy as np,cv2,random
from pycocotools import mask as mu
from pathlib import Path
C=Path('/workspace/lt/comp/MAGFiLO_1.0_Kaggle_2026/train'); coco=json.load(open(C/'MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
views={}; id2={im['id']:Path(im['file_name']).stem for im in coco['images']}
for im in coco['images']: views.setdefault(id2[im['id']],{})[im['id']]=[]
for a in coco['annotations']: views[id2[a['image_id']]][a['image_id']].append(a['segmentation'])
def soft_target(stem):
    acc=np.zeros((2048,2048),np.float32); vs=views[stem]
    for vid,segs in vs.items():
        u=np.zeros((2048,2048),np.uint8)
        for seg in segs:
            if isinstance(seg,str): seg=json.loads(seg)
            polys=seg if (seg and isinstance(seg[0],list)) else [seg]; polys=[list(map(float,p)) for p in polys if len(p)>=6]
            if polys: u|=mu.decode(mu.merge(mu.frPyObjects(polys,2048,2048))).astype(np.uint8)
        acc+=u
    return np.round(acc/max(1,len(vs))*255).astype(np.uint8)
def rot_shift(geo,y0,x0,n):
    cx,cy,R=geo; yy_,xx_=np.mgrid[y0:y0+n,x0:x0+n].astype(np.float64)
    X=(xx_-cx)/R; Y=-(yy_-cy)/R; ins=X*X+Y*Y<0.999
    B=np.arcsin(np.clip(Y,-1,1)); cB=np.cos(B); L=np.arcsin(np.clip(X/np.maximum(cB,1e-6),-1,1))
    w=14.713-2.396*np.sin(B)**2-1.787*np.sin(B)**4-0.9856
    return np.where(ins,R*cB*np.cos(L)*np.deg2rad(w/1440.0),0.0).astype(np.float32)
idx=json.load(open('ds/nb_index.json'))
cand=[(s,max(e['nb'],key=lambda d:abs(d['dt_min']))) for s,e in sorted(idx.items()) if e['nb']]
random.seed(11); sites={}
for s,d in cand: sites.setdefault(s[14:],[]).append((s,d))
pick=[]
for site in sorted(sites):
    pick+=random.sample(sites[site],min(2,len(sites[site])))
pick=pick[:10]
assert len(pick)==10, len(pick)
def contrast(img,tg):
    m=tg>=128
    if m.sum()<50: return np.nan
    ring=(cv2.dilate(m.astype(np.uint8),np.ones((13,13)))>0)&~(cv2.dilate(m.astype(np.uint8),np.ones((5,5)))>0)
    f=img.astype(np.float32); return float(f[ring].mean()-f[m].mean())
rows=[]; tiles=[]
yy,xx=np.mgrid[:2048,:2048].astype(np.float32)
for s,d in pick:
    tr=cv2.imread(str(C/'train_images'/(s+'.jpeg')),0); nb=cv2.imread(f"ds/nb/{d['stem']}.jpeg",0); tg=soft_target(s)
    D=rot_shift(idx[s]['geo'],0,0,2048); dt=d['dt_min']; exag=12  # control: 12x the shift
    tg_d=cv2.remap(tg,xx-dt*D,yy,cv2.INTER_LINEAR); tg_w=cv2.remap(tg,xx+exag*dt*D,yy,cv2.INTER_LINEAR)
    r=dict(train=s,nb=d['stem'],dt=dt,c_train=contrast(tr,tg),c_nb_derot=contrast(nb,tg_d),c_nb_raw=contrast(nb,tg),c_nb_wrong12x=contrast(nb,tg_w),maxshift=float(np.abs(D).max()*abs(dt)))
    r['ratio']=r['c_nb_derot']/r['c_train']; rows.append(r); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
    # tile around biggest filament
    ys,xs=np.nonzero(tg>=128); i=len(ys)//2; cy,cx=int(np.clip(ys[i],128,1920)),int(np.clip(xs[i],128,1920))
    def ov(img,t):
        c=cv2.cvtColor(img[cy-128:cy+128,cx-128:cx+128],cv2.COLOR_GRAY2BGR); cs,_=cv2.findContours((t[cy-128:cy+128,cx-128:cx+128]>=128).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(c,cs,-1,(0,0,255),1); return c
    tiles.append(np.hstack([ov(tr,tg),ov(nb,tg_d)]))
cv2.imwrite('overlay_check.png',np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,10,2)]))
json.dump(rows,open('overlay_check.json','w'),indent=1)
ok=sum(r['ratio']>=0.8 for r in rows); print('PASS' if ok>=9 else 'FAIL', ok,'/',len(rows))
