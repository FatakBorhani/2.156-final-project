import numpy as np, pandas as pd
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
d=pd.read_parquet('ds.parquet')
ours=load_ours(); ours.to_pickle('ours.pkl')
f3=ours[ours.feasible]
print('ours feasible', len(f3), f3.groupby('run').size().to_dict())
print('HV ours check', hypervolume(np.c_[-f3.LgB, f3.AR]))
d['others_ok']=(d[['v_iota','v_mirror','v_flux','v_well']]<=0.01).all(axis=1)
d['worst_noqi']=d[['v_iota','v_mirror','v_flux','v_well']].clip(lower=0).max(axis=1)
print('counts worst<=0.3/0.4/0.5', [(d.worst<=t).sum() for t in (0.3,0.4,0.5)], 'others_ok', d.others_ok.sum(), 'others_ok&qi<=0.3', (d.others_ok&(d.v_qi<=0.3)).sum())
print(pd.crosstab(d.nfp,[d.worst<=0.3]).T, pd.crosstab(d.nfp,d.worst<=0.4).T, pd.crosstab(d.nfp,d.worst<=0.5).T, pd.crosstab(d.nfp,d.others_ok).T, pd.crosstab(d.nfp, d.worst_noqi<=0.1).T)
bins=np.arange(3,12.6,1.0)
d['ARbin']=pd.cut(d.AR,bins)
def env(g):
    return pd.Series({'n':len(g),'p95':g.LgB.quantile(.95) if len(g)>=20 else np.nan,'p99':g.LgB.quantile(.99) if len(g)>=100 else np.nan,'max':g.LgB.max()})
rows=[]
for nfp in range(1,6):
    g=d[d.nfp==nfp]
    for lab,sub in [('all',g),('w<=0.5',g[g.worst<=0.5]),('w<=0.4',g[g.worst<=0.4]),('w<=0.3',g[g.worst<=0.3]),('others_ok',g[g.others_ok]),('noqi<=0.1',g[g.worst_noqi<=0.1])]:
        e=sub.groupby('ARbin',observed=False).apply(env)
        for b,r in e.iterrows(): rows.append(dict(nfp=nfp,set=lab,ARbin=str(b),**r))
E=pd.DataFrame(rows); E.to_csv('q1_envelope.csv',index=False)
for lab in ['all','w<=0.5','w<=0.4','others_ok','noqi<=0.1']:
    col='p99' if lab=='all' else 'max'
    t=E[E.set==lab].pivot(index='ARbin',columns='nfp',values=col).round(1)
    n=E[E.set==lab].pivot(index='ARbin',columns='nfp',values='n').astype(int)
    print('\n==',lab,col); print(t.astype(str)+' ('+n.astype(str)+')')
print('\n== all p95'); print(E[E.set=='all'].pivot(index='ARbin',columns='nfp',values='p95').round(1))
# HV per nfp
print('\nHV as if feasible')
res=[]
for nfp in range(1,6):
    g=d[d.nfp==nfp]; r={'nfp':nfp}
    for lab,sub in [('all',g),('w<=0.5',g[g.worst<=0.5]),('w<=0.4',g[g.worst<=0.4]),('w<=0.3',g[g.worst<=0.3]),('others_ok',g[g.others_ok]),('noqi<=0.1',g[g.worst_noqi<=0.1])]:
        r[lab]=round(hypervolume(np.c_[-sub.LgB,sub.AR]),1); r['n_'+lab]=len(sub)
    res.append(r)
print(pd.DataFrame(res))
# front points for w<=0.4 per nfp
for nfp in range(1,6):
    sub=d[(d.nfp==nfp)&(d.worst<=0.4)].sort_values('AR')
    fr=[]; best=-1
    for _,r in sub.iterrows():
        if r.LgB>best: fr.append((round(r.AR,2),round(r.LgB,2),round(r.worst,3))); best=r.LgB
    print(nfp,'w<=0.4 front (AR,LgB,worst):',fr)
