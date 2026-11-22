import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
d=pd.read_parquet('ds.parquet')
ok=lambda c,t=0.01: d['v_'+c]<=t
subsets={'none':np.ones(len(d),bool),'iota':ok('iota'),'iota+mirror':ok('iota')&ok('mirror'),'iota+flux':ok('iota')&ok('flux'),'iota+well':ok('iota')&ok('well'),
 'iota+mirror+flux':ok('iota')&ok('mirror')&ok('flux'),'all but qi':ok('iota')&ok('mirror')&ok('flux')&ok('well'),
 'iota,mirror,flux<=.01; well<=.3':ok('iota')&ok('mirror')&ok('flux')&ok('well',0.3),
 'all but qi <=0.1':ok('iota',.1)&ok('mirror',.1)&ok('flux',.1)&ok('well',.1),
 'all but qi <=0.2':ok('iota',.2)&ok('mirror',.2)&ok('flux',.2)&ok('well',.2)}
rows=[]
for k,m in subsets.items():
    r={'subset':k}
    for nfp in range(1,6):
        s=d[m&(d.nfp==nfp)]; r[f'HV{nfp}']=round(hypervolume(np.c_[-s.LgB,s.AR]),1); r[f'n{nfp}']=len(s)
    rows.append(r)
T=pd.DataFrame(rows); print(T.to_string())
# envelope (max L) per AR bin for iota-ok and all-but-qi subsets
d['ARbin']=pd.cut(d.AR,[3,6,7,8,9,10,11,12.2])
for k in ['iota','iota+mirror','iota+mirror+flux','all but qi <=0.1','all but qi <=0.2']:
    s=d[subsets[k]]
    t=s.pivot_table(index='ARbin',columns='nfp',values='LgB',aggfunc='max').round(1)
    p=s.pivot_table(index='ARbin',columns='nfp',values='LgB',aggfunc=lambda x: np.quantile(x,.99) if len(x)>=100 else np.nan).round(1)
    print('\n== max L (p99 where n>=100) |',k); print(t.astype(str)+' / '+p.astype(str))
