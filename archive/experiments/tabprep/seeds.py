import json, numpy as np
R={}
for l in open('results.jsonl'):
    r=json.loads(l); R[r['variant']]=r
def get(v,s): return R.get(v+(f"@s{s}" if s else ""))
print("base par seed:", [get('base',s)['rmse'] for s in range(3)], [get('base',s)['plis'] for s in range(3)])
for v in ['gb','oafe','gbT','gb+oafe']:
    D=[];rm=[]
    for s in range(3):
        a,b=get(v,s),get('base',s)
        if a is None: continue
        D.append(np.array(a['plis'])-np.array(b['plis'])); rm.append(a['rmse']-b['rmse'])
    D=np.array(D)
    print(f"{v:10s} delta rmse par seed {np.round(rm,3)} moy {np.mean(rm):+.3f} | delta par pli (moy 3 seeds) {np.round(D.mean(0),3)} | plis*seeds mieux {(D<0).sum()}/{D.size}")
# bruit de seed : écart base s1/s2 vs s0
B=np.array([get('base',s)['plis'] for s in range(3)]); print("écart-type seed par pli base", np.round(B.std(0,ddof=1),3), "rmse", np.round([get('base',s)['rmse'] for s in range(3)],3))
