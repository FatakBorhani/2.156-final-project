import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
d=pd.read_parquet('ds.parquet'); ours=pd.read_pickle('ours.pkl'); f3=ours[ours.feasible]
fr=front_points(f3.LgB.values,f3.AR.values)
d['gain']=hv_gain(d.LgB.values,d.AR.values,fr)
d[['plasma_config_id','gain']].to_parquet('gain.parquet')
V=['v_'+c for c in CNAMES]
print('gain quantiles', d.gain.quantile([.5,.9,.99,.999,1]).round(1).tolist())
for thr in [1,5,20,50]:
    A=d[d.gain>=thr]
    print(f'\n### gain>={thr}: n={len(A)} by nfp', A.groupby('nfp').size().to_dict())
    print('fail % per constraint'); t=(A.groupby('nfp')[V].apply(lambda g:(g>0.01).mean()*100)).round(1); t.loc['all']=((A[V]>0.01).mean()*100).round(1); print(t)
    print('median violation'); t=A.groupby('nfp')[V].median().round(2); t.loc['all']=A[V].median().round(2); print(t)
    aw=A[V].idxmax(axis=1).str[2:]; print('argmax-violation share %'); print((pd.crosstab(A.nfp,aw,normalize='index')*100).round(1)); print('all', (aw.value_counts(normalize=True)*100).round(1).to_dict())
    print('min worst',A.worst.min().round(3),'n worst<=0.5',(A.worst<=0.5).sum(),'n worst<=1',(A.worst<=1).sum())
    # if we could fix qi (ignore it), which is next blocker
    W=A[['v_iota','v_mirror','v_flux','v_well']]; aw2=W.idxmax(axis=1).str[2:]
    print('ignoring qi, argmax share', (aw2.value_counts(normalize=True)*100).round(1).to_dict(), 'worst_noqi min', W.clip(lower=0).max(axis=1).min().round(3))
