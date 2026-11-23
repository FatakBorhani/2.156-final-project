import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq
from common import *
fs = sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--proxima-fusion--constellaration/snapshots/*/data/train-*.parquet')))
assert all('bench-results' not in f for f in fs)
cols = None
frames = []
for f in fs:
    names = pq.read_schema(f).names
    use = [c for c in names if not c.endswith('.json') and not c.endswith('.id') and 'x_lmn' not in c and 'knot' not in c
           and not c.startswith('vmec_omnigenous') and 'desc_omnigenous_field_optimization_settings.objective' not in c]
    frames.append(pq.read_table(f, columns=use).to_pandas())
df = pd.concat(frames, ignore_index=True)
print(len(df))
ren = {'metrics.aspect_ratio':'AR','metrics.minimum_normalized_magnetic_gradient_scale_length':'LgB','boundary.n_field_periods':'nfp',
       'metrics.max_elongation':'elong','metrics.edge_rotational_transform_over_n_field_periods':'iota_edge',
       'metrics.qi':'qi','metrics.edge_magnetic_mirror_ratio':'mirror','metrics.flux_compression_in_regions_of_bad_curvature':'flux',
       'metrics.vacuum_well':'well','metrics.axis_magnetic_mirror_ratio':'mirror_axis','metrics.axis_rotational_transform_over_n_field_periods':'iota_axis',
       'metrics.average_triangularity':'tri'}
df = df.rename(columns=ren)
df['has_metrics'] = df[['AR','LgB','iota_edge','qi','mirror','flux','well']].notna().all(axis=1) & (df['qi'] > 0)
d = df[df.has_metrics].copy()
V = violations_from_metrics(d.iota_edge.values, d.qi.values, d.mirror.values, d.flux.values, d.well.values)
for i, c in enumerate(CNAMES): d['v_' + c] = V[:, i]
d['worst'] = np.clip(V, 0, None).max(axis=1)
# generation method
q = 'qp_init_omnigenous_field_optimization_settings.'; n = 'nae_init_omnigenous_field_optimization_settings.'; g = 'desc_omnigenous_field_optimization_settings.initial_guess_settings.'
d['method'] = np.where(d[q + 'aspect_ratio'].notna(), 'qp', np.where(d[n + 'aspect_ratio'].notna(), 'nae', np.where(d[g + 'aspect_ratio'].notna(), 'desc_ig', 'other')))
# spectral features
feats = [spectral_features(np.array([np.asarray(x) for x in rc]), np.array([np.asarray(x) for x in zs]), nf)
         for rc, zs, nf in zip(d['boundary.r_cos'], d['boundary.z_sin'], d['nfp'])]
F = pd.DataFrame(feats, index=d.index)
d = pd.concat([d.drop(columns=['boundary.r_cos','boundary.z_sin']), F], axis=1)
d.columns = [c.replace('omnigenous_field_and_targets.', 'tgt.').replace('qp_init_omnigenous_field_optimization_settings.', 'qp.')
             .replace('nae_init_omnigenous_field_optimization_settings.', 'nae.').replace('desc_omnigenous_field_optimization_settings.initial_guess_settings.', 'ig.')
             .replace('desc_omnigenous_field_optimization_settings.equilibrium_settings.', 'eq.') for c in d.columns]
d.to_parquet(SCR + '/ds.parquet')
print(d.shape, d.method.value_counts().to_dict(), d.nfp.value_counts().sort_index().to_dict())
print(list(d.columns))
