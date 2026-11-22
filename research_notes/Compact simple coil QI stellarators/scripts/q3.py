import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from common import *
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import spearmanr
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',100)
d=pd.read_parquet('ds.parquet')
d['abs_iota']=d.iota_edge.abs(); d['abs_iota_axis']=d.iota_axis.abs(); d['log10qi']=np.log10(d.qi)
d['tgt_iota_per_nfp']=d['tgt.rotational_transform']/d.nfp
for mth in ['qp','nae','desc_ig','other']: d['meth_'+mth]=(d.method==mth).astype(int)
spec=['axis_exc','axis_R_exc','axis_Z_exc','m1_helical','m1_over_m0','b_over_a10','Efrac_m0','Efrac_m1','Efrac_m2','Efrac_m3','Efrac_m4','Efrac_n_ge2','Efrac_n_ge3','rms_hi_over_a','spec_width']
gen=['tgt.aspect_ratio','tgt.max_elongation','tgt.rotational_transform','tgt_iota_per_nfp','tgt_mirror','tgt_xlmn_norm','meth_qp','meth_nae','meth_desc_ig']
phys=['elong','mirror','abs_iota','abs_iota_axis','mirror_axis','tri','flux','well','log10qi']
def residualize(s):
    s=s.copy(); s['Lres']=np.nan
    for nfp,g in s.groupby('nfp'):
        c=np.polyfit(g.AR,g.LgB,2 if len(g)>30 else 1); s.loc[g.index,'Lres']=g.LgB-np.polyval(c,g.AR)
    return s
sets={
 'A: near-feasible worst<=0.5 (all nfp)': d[d.worst<=0.5],
 'B: iota&mirror pass (all nfp)': d[(d.v_iota<=0.01)&(d.v_mirror<=0.01)],
 'C: nfp3, iota pass': d[(d.nfp==3)&(d.v_iota<=0.01)],
}
out={}
for name,s in sets.items():
    s=residualize(s)
    print(f'\n######## {name}  n={len(s)}  nfp counts {s.nfp.value_counts().sort_index().to_dict()}')
    # spearman with residual
    feats=['nfp']+phys+gen+spec
    rho={f:spearmanr(s[f],s.Lres,nan_policy='omit')[0] for f in feats if s[f].nunique()>1}
    R=pd.Series(rho).sort_values(key=abs,ascending=False)
    print('Spearman rho with residual L (after per-nfp quadratic AR fit):'); print(R.round(3).head(20).to_string())
    for lab,fs in [('geometry+targets (no outcome metrics)',['nfp','AR']+gen+spec),('all incl. physics metrics',['nfp','AR']+phys+gen+spec)]:
        fs=[f for f in fs if s[f].nunique()>1]
        X=s[fs].values; y=s.Lres.values
        m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=20,random_state=0)
        pred=cross_val_predict(m,X,y,cv=KFold(5,shuffle=True,random_state=0))
        r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
        m.fit(X,y)
        pi=permutation_importance(m,X,y,n_repeats=5,random_state=0,n_jobs=-1)
        imp=pd.Series(pi.importances_mean,index=fs).sort_values(ascending=False)
        print(f'-- GBT [{lab}] CV R2 on residual = {r2:.3f}; resid std {y.std():.2f}; top permutation importances (drop in R2):')
        print(imp.head(10).round(3).to_string())
        out[(name,lab)]=(r2,imp)
    # binned effect of top physical features
    for f in ['elong','mirror','abs_iota','axis_exc','m1_over_m0','Efrac_m_ge2' if 'Efrac_m_ge2' in s else 'Efrac_m2','tgt_mirror','tgt.max_elongation']:
        q=pd.qcut(s[f],5,duplicates='drop'); t=s.groupby(q,observed=True).Lres.agg(['count','mean']).round(2)
        print(f'  {f}: quintile mean resid L', list(zip([f'{i.left:.3g}-{i.right:.3g}' for i in t.index],t['mean'])))
pd.to_pickle(out,'q3_out.pkl')
