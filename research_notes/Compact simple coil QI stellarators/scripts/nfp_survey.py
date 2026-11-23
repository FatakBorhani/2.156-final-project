import glob, numpy as np, pandas as pd, pyarrow.parquet as pq, os
P = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--proxima-fusion--constellaration/snapshots/*/data/train-*.parquet"))
C = ["metrics.aspect_ratio","metrics.edge_rotational_transform_over_n_field_periods","metrics.qi",
     "metrics.edge_magnetic_mirror_ratio","metrics.flux_compression_in_regions_of_bad_curvature","metrics.vacuum_well",
     "metrics.minimum_normalized_magnetic_gradient_scale_length","boundary.n_field_periods"]
df = pd.concat([pq.read_table(p, columns=C).to_pandas() for p in P], ignore_index=True)
m = df.dropna(subset=C); m = m[m["metrics.qi"] > 0]
iota = np.abs(m[C[1]])
v = pd.DataFrame({"iota": (0.25-iota)/0.25, "qi": (np.log10(m[C[2]])+3.5)/3.5, "mirror": (m[C[3]]-0.25)/0.25,
                  "flux": (m[C[4]]-0.9)/0.9, "well": (0-m[C[5]])/0.1})
m = m.assign(worst=v.clip(lower=0).max(axis=1), **{f"v_{k}": v[k] for k in v})
m["score"] = (m[C[6]]-1)*(20-m[C[0]])
pd.set_option("display.width", 200)
for n, g in m.groupby(C[7]):
    print(f"\n=== nfp {n}: {len(g)} designs; worst quantiles", g.worst.quantile([0, .001, .01]).round(3).tolist())
    print("  constraint pass rates %:", {k: round(100*(g[f'v_{k}']<=0.01).mean(), 1) for k in v})
    t = g.nsmallest(5, "worst")[["worst", C[0], C[6], "v_iota","v_qi","v_mirror","v_flux","v_well"]].round(3)
    t.columns = ["worst","AR","LgB","iota","qi","mirror","flux","well"]; print(t.to_string(index=False))
    # designs with large rectangle area: what does the dataset reach per nfp (ignoring feasibility)?
    print("  among worst<=0.5: AR min", round(g[g.worst<=0.5][C[0]].min(),2), "LgB max", round(g[g.worst<=0.5][C[6]].max(),2),
          "best (LgB-1)(20-AR)", round(g[g.worst<=0.5].score.max(),1))
