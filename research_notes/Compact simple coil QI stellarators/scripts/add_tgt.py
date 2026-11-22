import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq
fs = sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--proxima-fusion--constellaration/snapshots/*/data/train-*.parquet')))
p='omnigenous_field_and_targets.omnigenous_field.'
out=[]
for f in fs:
    t=pq.read_table(f,columns=['plasma_config_id',p+'modB_spline_knot_coefficients',p+'x_lmn']).to_pandas()
    for pid,k,x in zip(t.plasma_config_id,t[p+'modB_spline_knot_coefficients'],t[p+'x_lmn']):
        try:
            k0=np.asarray(k[0],float); mr=(k0.max()-k0.min())/(k0.max()+k0.min())
            k1=np.asarray(k[1],float) if len(k)>1 else np.zeros(1); krad=np.abs(k1).max()
        except Exception: mr=np.nan; krad=np.nan
        try:
            X=np.array([[np.asarray(r,float) for r in s] for s in x]); xn=np.sqrt((X**2).sum())
        except Exception: xn=np.nan
        out.append((pid,mr,krad,xn))
o=pd.DataFrame(out,columns=['plasma_config_id','tgt_mirror','tgt_knot_radial','tgt_xlmn_norm'])
d=pd.read_parquet('ds.parquet')
d=d.merge(o.drop_duplicates('plasma_config_id'),on='plasma_config_id',how='left')
d.to_parquet('ds.parquet'); print(d[['tgt_mirror','tgt_knot_radial','tgt_xlmn_norm']].describe().round(3))
