import json, sys
ref=[76.656,77.305,76.804,77.583,77.659]
if len(sys.argv)>1: ref=json.loads(sys.argv[1])
for l in open('results.jsonl'):
    r=json.loads(l)
    d=[round(r['plis'][i]-ref[k],3) for i,k in enumerate(r['folds'])]
    st=r['stats']
    print(f"{r['variant']:28s} {r['rmse']:.3f} {d} mieux {sum(x<0 for x in d)}/{len(d)} {r['minutes']}min new={st[0]['n_new']} gain={sum(x['gain_share_new'] for x in st)/len(st):.3f} iter={[x['best_iter'] for x in st]} rss={max(x['maxrss_gb'] for x in st)} tfeat={st[0]['t_feat']}")
