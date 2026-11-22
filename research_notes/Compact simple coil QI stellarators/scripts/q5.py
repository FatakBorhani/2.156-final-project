import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
d=pd.read_parquet('ds.parquet').merge(pd.read_parquet('gain.parquet'),on='plasma_config_id')
V=['v_'+c for c in CNAMES]
d['iota_tot']=d.iota_edge.abs()*d.nfp
for name,m in [('L>=12 & AR<=10',(d.LgB>=12)&(d.AR<=10)),('L>=12 & AR<=9',(d.LgB>=12)&(d.AR<=9)),('L>=15 & AR<=11',(d.LgB>=15)&(d.AR<=11)),('L>=20 & AR<=8',(d.LgB>=20)&(d.AR<=8)),('L>=10 & AR<=8',(d.LgB>=10)&(d.AR<=8)),('L>=8 & AR<=7',(d.LgB>=8)&(d.AR<=7))]:
    S=d[m]
    print(f'\n### {name}: n={len(S)}', S.groupby('nfp').size().to_dict(), 'method', S.method.value_counts().to_dict())
    if len(S)==0: continue
    print(' worst q[0,.1,.5]', S.worst.quantile([0,.1,.5]).round(2).tolist(), ' n worst<=0.5', (S.worst<=0.5).sum(), '<=1', (S.worst<=1).sum())
    print(' pass% ', ((S[V]<=0.01).mean()*100).round(1).to_dict())
    print(' median viol', S[V].median().round(2).to_dict())
    print(' iota_edge/nfp q', S.iota_edge.abs().quantile([.1,.5,.9]).round(3).tolist(), ' mirror q', S.mirror.quantile([.1,.5,.9]).round(3).tolist(), ' elong q', S.elong.quantile([.1,.5,.9]).round(2).tolist())
    print(' tgt AR q', S['tgt.aspect_ratio'].quantile([.1,.5,.9]).round(2).tolist(), ' tgt iota q', S['tgt.rotational_transform'].quantile([.1,.5,.9]).round(2).tolist(),
          ' tgt elong', S['tgt.max_elongation'].quantile([.1,.5,.9]).round(2).tolist(), ' tgt mirror', S['tgt_mirror'].quantile([.1,.5,.9]).round(3).tolist())
    print(' vs all-dataset tgt iota median', d['tgt.rotational_transform'].median().round(2), 'tgt mirror median', d.tgt_mirror.median().round(3))
    best=S.sort_values('worst').head(6)
    print(best[['nfp','AR','LgB','worst']+V+['elong','method','tgt.aspect_ratio','tgt.rotational_transform','tgt.max_elongation','tgt_mirror','gain']].round(3).to_string())
# max L in AR bins for each nfp regardless of constraints, + their iota
print('\nmax L by AR bin (any violation), with iota_edge/nfp of argmax')
d['ARbin']=pd.cut(d.AR,[3,5,6,7,8,9,10,11,12.2])
for nfp in range(1,6):
    g=d[d.nfp==nfp]; r=g.loc[g.groupby('ARbin',observed=True).LgB.idxmax()]
    print(nfp, [(str(b), round(L,1), round(abs(i),2), round(w,2)) for b,L,i,w in zip(r.ARbin,r.LgB,r.iota_edge,r.worst)])
# relationship L vs iota: among AR 8-10, nfp3, L quantiles by iota bins
print('\nnfp3 & nfp4 & nfp5, AR in (8,10]: L quantiles by |iota_edge|/nfp bin')
for nfp in [3,4,5]:
    g=d[(d.nfp==nfp)&(d.AR>8)&(d.AR<=10)]
    print(nfp); print(g.groupby(pd.cut(g.iota_edge.abs(),[0,.05,.1,.15,.2,.25,.3,.4,.6])).LgB.describe()[['count','50%','max']].round(2).T)
