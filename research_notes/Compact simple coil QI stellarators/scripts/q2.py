import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
d=pd.read_parquet('ds.parquet'); ours=pd.read_pickle('ours.pkl'); f3=ours[ours.feasible]
V=['v_'+c for c in CNAMES]
P=d[V]<=0.01
for c in CNAMES: d['p_'+c]=P['v_'+c]
d['ARbin']=pd.cut(d.AR,[0,6,7,8,9,10,11,13]); d['Lbin']=pd.cut(d.LgB,[-1,4,6,8,10,12,14,30])
print('overall pass %'); print((d.groupby('nfp')[['p_'+c for c in CNAMES]].mean()*100).round(1))
for key in ['ARbin','Lbin']:
    print('\n=== pass % by',key,'(all nfp)'); t=d.groupby(key)[['p_'+c for c in CNAMES]].mean()*100; t['n']=d.groupby(key).size(); print(t.round(1))
    for c in CNAMES:
        t=(d.pivot_table(index=key,columns='nfp',values='p_'+c,aggfunc='mean')*100).round(1)
        print('--',c); print(t)
# median violation by bin (continuous view)
print('\n=== median violation by ARbin x constraint (nfp=3)')
print(d[d.nfp==3].groupby('ARbin')[V].median().round(3))
print('\n=== median violation by Lbin x constraint (nfp=3)')
print(d[d.nfp==3].groupby('Lbin')[V].median().round(3))
print('\n=== median violation by Lbin x constraint (all nfp)')
print(d.groupby('Lbin')[V].median().round(3))
# above our front
fa=f3.sort_values('AR')
def best_L_at(ar):
    m=fa.AR<=ar
    return fa.LgB[m].max() if m.any() else -np.inf
grid=np.sort(fa.AR.values); cummax=np.maximum.accumulate(fa.LgB.values)
idx=np.searchsorted(grid,d.AR.values,side='right')-1
bestL=np.where(idx>=0,cummax[np.clip(idx,0,None)],-np.inf)
d['above']=d.LgB.values>bestL
d['margin']=d.LgB.values-bestL
A=d[d.above]
print('\nabove our front: n=',len(A),'by nfp',A.groupby('nfp').size().to_dict())
print('fail rate % among above:'); print((1-A.groupby('nfp')[['p_'+c for c in CNAMES]].mean())*100)
A['argworst']=A[V].idxmax(axis=1).str[2:]
print('argworst counts:'); print(pd.crosstab(A.nfp,A.argworst))
nf=(~A[['p_'+c for c in CNAMES]]).sum(axis=1)
print('n failing constraints dist', nf.value_counts().sort_index().to_dict())
sole=A[nf==1]; print('sole blocker', sole[V].idxmax(axis=1).value_counts().to_dict())
print('median violation among above by nfp'); print(A.groupby('nfp')[V].median().round(3))
print('worst quantiles among above', A.worst.quantile([0,.01,.1,.5]).round(3).tolist())
# "far above": margin >=2 and AR<=11
B=A[(A.margin>=2)]
print('\nfar above (margin>=2): n',len(B),B.groupby('nfp').size().to_dict(),'worst min',B.worst.min())
print((1-B[['p_'+c for c in CNAMES]].mean())*100)
print(B[V].median().round(3))
d[['plasma_config_id','above','margin']].to_parquet('above.parquet')
