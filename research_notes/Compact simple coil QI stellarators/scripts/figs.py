import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
OUT='/Users/leifakerley/Documents/Repo/2.156-final-project/research_notes/Compact simple coil QI stellarators/'
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.6,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
d=pd.read_parquet('ds.parquet'); o=pd.read_pickle('ours_feat.pkl'); f3=o[o.feasible]; n4=o[o.run=='nfp4']
fr=front_points(f3.LgB.values,f3.AR.values)
def stair(ax):
    xs=np.r_[fr[:,0],20]; ys=np.r_[fr[:,1],fr[-1,1]]
    ax.step(xs,ys,where='post',color='#0b0b0b',lw=1.6,label='our feasible front (HV 144.4)')
# Fig 1: per nfp L vs AR
fig,axs=plt.subplots(1,5,figsize=(16,3.6),sharey=True)
for k,ax in enumerate(axs):
    nfp=k+1; g=d[d.nfp==nfp]
    ax.scatter(g.AR,g.LgB,s=1,color='#c9c8c2',alpha=0.4,rasterized=True,label='all dataset')
    gi=g[g.v_iota<=0.01]; ax.scatter(gi.AR,gi.LgB,s=1,color=C[0],alpha=0.25,rasterized=True,label='iota ≥ 0.25 passes')
    gn=g[g.worst<=0.5]; ax.scatter(gn.AR,gn.LgB,s=6,color=C[1],label='worst violation ≤ 0.5')
    gq=g[(g.worst<=0.4)]; ax.scatter(gq.AR,gq.LgB,s=10,color=C[7],marker='D',label='worst ≤ 0.4')
    stair(ax)
    if nfp==4: ax.scatter(n4[n4.worst<=0.1].AR,n4[n4.worst<=0.1].LgB,s=14,marker='*',color=C[6],label='our nfp-4 (worst 0.027-0.1)')
    ax.set_title(f'nfp = {nfp}  (n = {len(g):,})'); ax.set_xlabel('aspect ratio'); ax.set_xlim(3,18); ax.set_ylim(0,24)
axs[0].set_ylabel('L∇B (min. normalized gradient scale length)')
h,l=[],[]
for ax in axs:
    for hh,ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
fig.legend(h,l,loc='lower center',ncol=6,frameon=False,markerscale=3,bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(); fig.savefig(OUT+'fig1_dataset_L_vs_AR_by_nfp.png',dpi=150,bbox_inches='tight'); plt.close()
# Fig 2: iota-pass max L envelope per AR bin, per nfp, vs our front
bins=np.arange(3,12.6,0.5); mid=(bins[:-1]+bins[1:])/2
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,(lab,mask) in zip(axs,[('iota passes (|ι|/nfp ≥ 0.25)',d.v_iota<=0.01),('iota and mirror pass',(d.v_iota<=0.01)&(d.v_mirror<=0.01))]):
    for k in range(5):
        g=d[mask&(d.nfp==k+1)]; mx=g.groupby(pd.cut(g.AR,bins),observed=False).LgB.max().values
        ax.plot(mid,mx,'-o',ms=3,lw=2,color=C[k],label=f'nfp {k+1}')
    stair(ax); ax.set_title('Dataset max L∇B per 0.5-AR bin, '+lab); ax.set_xlabel('aspect ratio'); ax.set_xlim(3,18)
axs[0].set_ylabel('L∇B'); axs[1].legend(frameon=False,loc='upper left')
fig.tight_layout(); fig.savefig(OUT+'fig2_envelopes_iota_mirror_by_nfp.png',dpi=150,bbox_inches='tight'); plt.close()
# Fig 3: |iota| vs L at AR 8-10 (explains high-L outliers)
fig,ax=plt.subplots(figsize=(6,4))
for k in [2,3,4]:
    g=d[(d.nfp==k+1)&(d.AR>8)&(d.AR<=10)]; ax.scatter(g.iota_edge.abs(),g.LgB,s=1,alpha=0.3,color=C[k-2],rasterized=True,label=f'nfp {k+1}')
ax.axvline(0.25,color='#52514e',ls='--',lw=1); ax.text(0.255,21,'iota limit 0.25',color='#52514e')
ax.set_xlabel('|edge iota| per field period'); ax.set_ylabel('L∇B'); ax.set_title('Dataset designs at AR 8-10'); ax.set_xlim(0,0.7)
ax.legend(frameon=False,markerscale=8); fig.tight_layout(); fig.savefig(OUT+'fig3_iota_vs_L_AR8to10.png',dpi=150); plt.close()
# Fig 4: spectral simplicity vs residual L (nfp3 iota pass)
s=d[(d.nfp==3)&(d.v_iota<=0.01)].copy(); c=np.polyfit(s.AR,s.LgB,2); s['Lres']=s.LgB-np.polyval(c,s.AR)
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,f,lab in [(axs[0],'rms_hi_over_a','RMS of (m ≥ 2 or |n| ≥ 3) coefficients / minor radius'),(axs[1],'Efrac_n_ge3','energy fraction in |n| ≥ 3 modes')]:
    for j,(m,g) in enumerate(s.groupby('method')):
        ax.scatter(g[f],g.Lres,s=1,alpha=0.3,color=C[j],rasterized=True,label=m)
    ax.set_xlabel(lab); ax.axhline(0,color='#52514e',lw=0.8)
axs[1].set_xscale('symlog',linthresh=1e-4); axs[0].set_ylabel('L∇B residual after AR fit'); axs[0].legend(frameon=False,markerscale=8,title='generator')
fig.suptitle('nfp = 3, iota-passing dataset designs (n = 36,136)'); fig.tight_layout(); fig.savefig(OUT+'fig4_spectral_vs_residual_L.png',dpi=150); plt.close()
# Fig 5: our front: violations vs AR for nondominated front + nfp4
FR=f3.copy(); keep=[]
for i,r in FR.iterrows():
    if not ((FR.AR<=r.AR)&(FR.LgB>=r.LgB)&((FR.AR<r.AR)|(FR.LgB>r.LgB))).any(): keep.append(i)
FR=FR.loc[keep].sort_values('AR')
fig,axs=plt.subplots(1,2,figsize=(12,4))
names={'iota':'iota','qi':'QI','mirror':'mirror','flux':'flux compression','well':'vacuum well'}
for k,cn in enumerate(CNAMES):
    axs[0].plot(FR.AR,FR['v_'+cn],'-o',ms=2.5,lw=1.5,color=C[k],label=names[cn])
axs[0].axhline(0.01,color='#0b0b0b',ls='--',lw=1); axs[0].text(17.3,0.015,'0.01',fontsize=8)
axs[0].set_xlabel('aspect ratio'); axs[0].set_ylabel('normalized violation'); axs[0].set_title('Our nondominated feasible front (n = %d): constraint violations'%len(FR)); axs[0].legend(frameon=False,ncol=2)
axs[1].plot(f3.AR,f3.elong_cal,'o',ms=2.5,color=C[0],label='nfp 3 feasible: elongation (proxy, calibrated)')
ax2=axs[1]
ax2.plot(n4.AR,n4.elong_cal,'*',ms=5,color=C[6],label='nfp 4 near-feasible')
ax2.set_xlabel('aspect ratio'); ax2.set_ylabel('max elongation (boundary proxy ×1.15)'); ax2.set_title('Elongation along our designs'); ax2.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT+'fig5_our_front_active_constraints.png',dpi=150); plt.close()
print('done')
# modes5 vs 4x4 at AR 9.5-10.5
z=f3[(f3.AR<10.5)]
print(z.groupby('run')[['AR','LgB','Efrac_m3','Efrac_m4p','Efrac_n4p','rms_hi_over_a','elong_cal']].agg(['count','median','min','max']).round(4).T.to_string())
