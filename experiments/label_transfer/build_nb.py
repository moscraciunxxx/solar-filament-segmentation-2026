# Build the LT4 neighbour set: archive JPEG -> gray, competition zero-mask of the paired train frame, q95 JPEG.
# QC per neighbour vs its train frame: global phase shift (inner disk, high-pass) and hp correlation; geometry by limb fit.
import json, os, sys, glob, numpy as np, cv2, datetime as dt
from PIL import Image
C='/workspace/lt/comp/MAGFiLO_1.0_Kaggle_2026/train/train_images'
st=json.load(open('state.json')); os.makedirs('ds/nb',exist_ok=True)
P=lambda s: dt.datetime.strptime(s[:14],'%Y%m%d%H%M%S')
def limbfit(img):
    g=cv2.GaussianBlur(img.astype(np.float32),(0,0),3); m=(g>0.5*np.percentile(g[g>20],50)).astype(np.uint8)
    n,lab,sts,_=cv2.connectedComponentsWithStats(m); k=1+np.argmax(sts[1:,4]); m=(lab==k).astype(np.uint8)
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); c=max(cs,key=len)[:,0,:].astype(float)
    x,y=c[:,0],c[:,1]; s=np.linalg.lstsq(np.c_[2*x,2*y,np.ones_like(x)],x*x+y*y,rcond=None)[0]
    return [float(s[0]),float(s[1]),float(np.sqrt(s[2]+s[0]**2+s[1]**2))]
def rot_shift(geo):
    cx,cy,R=geo; X=(xx-cx)/R; Y=-(yy-cy)/R; ins=X*X+Y*Y<0.999
    B=np.arcsin(np.clip(Y,-1,1)); cB=np.cos(B); L=np.arcsin(np.clip(X/np.maximum(cB,1e-6),-1,1))
    w=14.713-2.396*np.sin(B)**2-1.787*np.sin(B)**4-0.9856
    return np.where(ins,R*cB*np.cos(L)*np.deg2rad(w/1440.0),0.0).astype(np.float32)
yy,xx=np.mgrid[:2048,:2048]; WIN=cv2.createHanningWindow((2048,2048),cv2.CV_32F)
idx=json.load(open('ds/nb_index.json')) if os.path.exists('ds/nb_index.json') else {}
qc=json.load(open('qc.json')) if os.path.exists('qc.json') else {}
for s,v in sorted(st.items()):
    if s in idx: continue
    comp=cv2.imread(f'{C}/{s}.jpeg',cv2.IMREAD_GRAYSCALE).astype(np.float32)
    geo=limbfit(comp); cx,cy,R=geo; r=np.hypot(xx-cx,yy-cy)
    zero=(comp==0)&(r>0.9*R); inner=(r<0.85*R).astype(np.float32)
    hp=lambda a:(a-cv2.GaussianBlur(a,(0,0),15))*inner
    ha=hp(comp); ent={'geo':geo,'nb':[]}
    D=rot_shift(geo); EXP1=float(np.median(D[inner>0]))
    for o in v['got']:
        nb=(P(s)+dt.timedelta(minutes=o)).strftime('%Y%m%d%H%M%S')+s[14:]
        a=np.array(Image.open(f'raw/{nb}.jpg').convert('L'))
        def meas(arr):
            af=arr.astype(np.float32); af[zero]=0; hb=hp(af)
            (dx,dy),resp=cv2.phaseCorrelate(ha*WIN,hb*WIN); return dx,dy,resp,float(np.corrcoef(ha[inner>0],hb[inner>0])[0,1])
        dx,dy,resp,cc=meas(a); ex=EXP1*o            # expected global x-shift from solar rotation
        rx,ry=dx-ex,dy; ix=int(round(rx)) if abs(rx)>=1 else 0; iy=int(round(ry)) if abs(ry)>=1 else 0      # integer-pixel re-registration (no interpolation), labels keep the rotation model
        if ix or iy: a=np.roll(a,(-iy,-ix),axis=(0,1))
        dx2,dy2,resp2,cc2=meas(a)
        ok=(abs(ix)<=6 and abs(iy)<=6 and abs(dx2-ex)<1.0 and abs(dy2)<1.0 and cc2>0.3)
        a[zero]=0
        qc[nb]={'train':s,'dt':o,'dx':dx,'dy':dy,'exp_dx':ex,'roll':[iy,ix],'dx2':dx2,'dy2':dy2,'hpcorr':cc2,'ok':bool(ok)}
        if ok:
            cv2.imwrite(f'ds/nb/{nb}.jpeg',a,[cv2.IMWRITE_JPEG_QUALITY,95]); ent['nb'].append({'stem':nb,'dt_min':o,'roll':[iy,ix]})
    idx[s]=ent
    if len(idx)%25==0:
        json.dump(idx,open('ds/nb_index.json','w')); json.dump(qc,open('qc.json','w')); print(len(idx),flush=True)
json.dump(idx,open('ds/nb_index.json','w')); json.dump(qc,open('qc.json','w'))
q=list(qc.values()); print('stems',len(idx),'neighbours tested',len(q),'kept',sum(x['ok'] for x in q))
