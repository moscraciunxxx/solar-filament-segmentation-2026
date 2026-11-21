# Polite neighbour downloader: 1 connection, ~2.5 s cycle, pause 30 min on 403, resumable.
import csv, time, os, json, bisect, urllib.request, urllib.error, datetime as dt
P=lambda s: dt.datetime.strptime(s[:14],'%Y%m%d%H%M%S')
folds={r['stem']:r['fold'] for r in csv.DictReader(open('/workspace/unet/ukit/folds_ours.csv'))}
test=[l.strip() for l in open('/workspace/magfilo_check/test_stems.txt') if l.strip()]
tt=sorted(P(s) for s in test)
trs=sorted(folds,key=P); trt=[(P(s),s) for s in trs]
def near_test(t):
    i=bisect.bisect(tt,t)
    return any(abs((tt[j]-t).total_seconds())<=3600 for j in (i-1,i) if 0<=j<len(tt))
def near_otherfold(t,stem):
    f=folds[stem]
    return any(folds[s]!=f and abs((u-t).total_seconds())<=3600 for u,s in trt)
offs=[]
for k in range(1,6): offs+= [-k,k]
K=4
state_f='state.json'
state=json.load(open(state_f)) if os.path.exists(state_f) else {}
log=open('dl.log','a')
def L(*a):
    s=time.strftime('%H:%M:%S ')+' '.join(map(str,a)); print(s,flush=True); log.write(s+'\n'); log.flush()
excl=open('excluded.txt','a')
UA='Mozilla/5.0 (research; Kaggle filament-segmentation-2026 label transfer)'
fails403=0
for stem in trs:
    got=state.get(stem,{}).get('got',[])
    if state.get(stem,{}).get('done'): continue
    t0=P(stem); site=stem[14:]
    tried=state.get(stem,{}).get('tried',[])
    dead=set()
    for o in offs:
        if len(got)>=K: break
        if o in tried: continue
        if (o>0) in dead: continue
        t=t0+dt.timedelta(minutes=o); nb=t.strftime('%Y%m%d%H%M%S')+site
        if near_test(t) or near_otherfold(t,stem):
            excl.write(f'{stem} {nb} {"test" if near_test(t) else "otherfold"}\n'); excl.flush(); tried.append(o); continue
        url=f'https://gong2.nso.edu/HA/hag/{nb[:6]}/{nb[:8]}/{nb}.jpg'
        t1=time.time()
        while True:
            try:
                req=urllib.request.Request(url,headers={'User-Agent':UA})
                data=urllib.request.urlopen(req,timeout=60).read()
                open(f'raw/{nb}.jpg','wb').write(data); got.append(o); code=200; break
            except urllib.error.HTTPError as e:
                code=e.code
                if code==403:
                    fails403+=1; L('403 on',nb,'-> pausing 30 min (count',fails403,')'); time.sleep(1800); continue
                break
            except Exception as e:
                L('ERR',nb,repr(e)); time.sleep(30); continue
        tried.append(o)
        if code==404: dead.add(o>0)   # first gap on a side = edge of the observing run; stop probing that side
        L(stem,o,code,round(time.time()-t1,2))
        time.sleep(max(0,2.5-(time.time()-t1)))
    state[stem]={'got':got,'tried':tried,'done':True}
    json.dump(state,open(state_f,'w'))
L('ALLDONE', sum(len(v['got']) for v in state.values()))
