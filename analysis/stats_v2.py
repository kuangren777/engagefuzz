import csv,collections,statistics as st,sys
from scipy.stats import spearmanr
def run(fn, minfail=1):
    R=list(csv.DictReader(open(fn)))
    cells=collections.defaultdict(list)
    for r in R: cells[(r['model'],r['suite'],r['ut'],r['it'])].append(r)
    p={k:st.mean(int(x['security']) for x in v) for k,v in cells.items()}
    lv=collections.defaultdict(list)
    for k,v in cells.items():
        for x in v: lv[int(x['level2'])].append(p[k])
    nv=[r for r in R if r['security']=='0']
    print(f"== {fn} runs={len(R)} cells={len(cells)} mixed={sum(0<v<1 for v in p.values())} ASR={st.mean(int(r['security']) for r in R):.3f}")
    print("  level2 -> (n, mean p):",{l:(len(v),round(st.mean(v),3)) for l,v in sorted(lv.items())})
    print(f"  nonviol={len(nv)} atk_hit_side={st.mean(int(r['atk_hit_side']) for r in nv):.3f} atk_hit_any={st.mean(int(r['atk_hit_any']) for r in nv):.3f}")
    # per model, model-centred fail-only
    X=[];C=[];Y=[];M=[]
    for k,v in cells.items():
        f=[x for x in v if x['security']=='0']
        if len(f)<minfail: continue
        X.append(st.mean(int(x['level2']) for x in f)); C.append(st.mean(int(x['n_ro']) for x in f)); Y.append(p[k]); M.append(k[0])
    print(f"  fail-only (minfail={minfail}) n={len(X)} prox2 rho={spearmanr(X,Y)[0]:.3f} p={spearmanr(X,Y)[1]:.3g} | cov rho={spearmanr(C,Y)[0]:.3f} p={spearmanr(C,Y)[1]:.3g}")
    for m in sorted(set(M)):
        a=[r for r in nv if r['model']==m]; b=[r for r in R if r['model']==m]
        idx=[i for i,mm in enumerate(M) if mm==m]
        xs=[X[i] for i in idx]; ys=[Y[i] for i in idx]
        rho = spearmanr(xs,ys)[0] if len(set(xs))>1 and len(set(ys))>1 else float('nan')
        print(f"   {m:28s} runs={len(b)} ASR={st.mean(int(r['security']) for r in b):.3f} atk_hit_side|nonviol={st.mean(int(r['atk_hit_side']) for r in a) if a else 0:.3f} mixedcells={sum(1 for i in idx if 0<Y[i]<1)} rho={rho:.3f}")
for fn in sys.argv[1:]:
    run(fn); 
run(sys.argv[1],3)
