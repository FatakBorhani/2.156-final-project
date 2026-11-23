import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings('ignore')
from common import *
from scipy.stats import spearmanr
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
o=pd.read_pickle('ours.pkl')
F=[]
for rc,zs,nfp in zip(o.rc,o.zs,o.nfp):
    rc=np.asarray(rc,float); zs=np.asarray(zs,float)
    f=spectral_features(rc,zs,nfp)
    M,N2=rc.shape; N=(N2-1)//2
    E=rc**2+zs**2; E[0,N]=0; tot=E.sum()
    f['Efrac_m5']=E[5].sum()/tot if M>5 else 0.0
    f['Efrac_n5']=(E[:,0].sum()+E[:,-1].sum())/tot if N>=5 else 0.0
    f['Efrac_m5_or_n5']=(E[5].sum() + (E[:5,0].sum()+E[:5,-1].sum() if N>=5 else 0))/tot if M>5 else (E[:,0].sum()+E[:,-1].sum())/tot if N>=5 else 0.0
    f['Efrac_n4p']=E[:,np.abs(np.arange(-N,N+1))>=4].sum()/tot
    f['Efrac_m4p']=E[4:].sum()/tot
    f['elong_proxy']=boundary_xsection_elong(rc,zs,nfp)
    F.append(f)
o=pd.concat([o.drop(columns=['rc','zs']).reset_index(drop=True),pd.DataFrame(F)],axis=1)
o['elong_cal']=1.15*o.elong_proxy-0.05
o.to_pickle('ours_feat.pkl')
V=['v_'+c for c in CNAMES]
f3=o[o.feasible]
print('feasible n',len(f3),'runs',f3.run.value_counts().to_dict())
fr=front_points(f3.LgB.values,f3.AR.values)
# front members: nondominated
def nondom(df):
    keep=[]
    for i,r in df.iterrows():
        dom=((df.AR<=r.AR)&(df.LgB>=r.LgB)&((df.AR<r.AR)|(df.LgB>r.LgB))).any()
        if not dom: keep.append(i)
    return df.loc[keep].sort_values('AR')
FR=nondom(f3)
print('nondominated front n',len(FR), FR.run.value_counts().to_dict())
f3['ARbin']=pd.cut(f3.AR,[9.5,10,10.5,11,12,13,14,15,16,17.5])
print('\n== feasible designs by AR bin: active-constraint rate (% with violation >= -0.02, i.e. within 0.03 of the 0.01 limit) and median violation')
act=(f3[V]>=-0.02)
t=act.groupby(f3.ARbin).mean().mul(100).round(0); t['n']=f3.groupby('ARbin').size(); t['maxL']=f3.groupby('ARbin').LgB.max().round(2); print(t)
print(f3.groupby('ARbin')[V].median().round(3))
print('\n== same for nondominated front points')
FR['ARbin']=pd.cut(FR.AR,[9.5,10,10.5,11,12,13,14,15,16,17.5])
t=(FR[V]>=-0.02).groupby(FR.ARbin).mean().mul(100).round(0); t['n']=FR.groupby('ARbin').size(); print(t)
print(FR.groupby('ARbin')[V].median().round(3))
cols=['elong_cal','axis_exc','axis_R_exc','axis_Z_exc','m1_helical','m1_over_m0','b_over_a10','Efrac_m1','Efrac_m2','Efrac_m3','Efrac_m4p','Efrac_n_ge2','Efrac_n_ge3','Efrac_n4p','rms_hi_over_a','spec_width']
print('\n== geometry along feasible set (median by AR bin)'); print(f3.groupby('ARbin')[cols].median().round(4).T)
print('Spearman with AR among feasible:', {c:round(spearmanr(f3[c],f3.AR)[0],2) for c in cols})
m5=f3[f3.run=='modes5'].copy(); m5['ARbin']=pd.cut(m5.AR,[9.5,10,10.5,11,12,13,14,15,16,17.5])
print('\n== modes5 feasible (n=%d): m=5 / n=5 energy by AR bin'%len(m5))
print(m5.groupby('ARbin',observed=True)[['Efrac_m5','Efrac_n5','Efrac_m5_or_n5','Efrac_m4p','Efrac_n4p','rms_hi_over_a','elong_cal']].agg(['median','max']).round(5).T)
print('Spearman AR vs m5/n5 within modes5 feasible:', {c:round(spearmanr(m5[c],m5.AR)[0],2) for c in ['Efrac_m5','Efrac_n5','Efrac_m5_or_n5','Efrac_m4p','Efrac_n4p','rms_hi_over_a']})
print('modes5 front share: AR<10.5 feasible designs by run', f3[f3.AR<10.5].run.value_counts().to_dict())
# nfp4 vs nfp3 at similar AR
n4=o[o.run=='nfp4']
print('\n== nfp4 hi rows', len(n4), 'worst quantiles', n4.worst.quantile([0,.1,.5]).round(3).tolist())
best4=n4.sort_values('worst').head(15)
print(best4[['AR','LgB','worst']+V+['elong_cal','axis_exc','m1_over_m0','Efrac_n_ge3','rms_hi_over_a']].round(3).to_string())
n4n=n4[n4.worst<=0.1]
print('nfp4 worst<=0.1: n',len(n4n),'AR range',n4n.AR.min().round(2),n4n.AR.max().round(2),'L range',n4n.LgB.min().round(2),n4n.LgB.max().round(2))
print('argmax constraint among nfp4 worst<=0.1:', n4n[V].idxmax(axis=1).value_counts().to_dict())
cmp3=f3[(f3.AR>=10.2)&(f3.AR<=11.2)]
print('\nnfp3 feasible at AR 10.2-11.2: n',len(cmp3),'L median/max',cmp3.LgB.median().round(2),cmp3.LgB.max().round(2))
cmp4=n4[(n4.AR>=10.2)&(n4.AR<=11.2)&(n4.worst<=0.1)]
print('nfp4 worst<=0.1 at AR 10.2-11.2: n',len(cmp4),'L median/max',cmp4.LgB.median().round(2) if len(cmp4) else None,cmp4.LgB.max().round(2) if len(cmp4) else None)
T=pd.DataFrame({'nfp3 feasible AR10.2-11.2':cmp3[cols+V+['LgB','AR']].median(),'nfp4 worst<=0.1 AR10.2-11.2':cmp4[cols+V+['LgB','AR']].median()})
print(T.round(4))
# all ours, L vs AR fit and residual correlates among feasible
c=np.polyfit(f3.AR,f3.LgB,1); f3['Lres']=f3.LgB-np.polyval(c,f3.AR)
print('\nfeasible L ~ AR linear fit', c.round(3), 'resid std', f3.Lres.std().round(3))
print('Spearman of residual L with features (feasible ours):', {k:round(spearmanr(f3[k],f3.Lres)[0],2) for k in cols})
