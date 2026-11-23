import json, glob, os
FT='/Users/leifakerley/Documents/Repo/2.156-final-project/data/feasibility_tests'
pts=set(); n_nfp={}
for f in sorted(glob.glob(FT+'/*_ledger.jsonl')):
    if 'dry' in f: continue
    for line in open(f):
        try: r=json.loads(line)
        except: continue
        if r.get('kind')=='hi' and r.get('feasible'):
            pts.add((round(r['aspect_ratio'],6), round(r['L_gradB'],6)))
            n_nfp[r.get('nfp',r.get('n_field_periods','?'))]=n_nfp.get(r.get('nfp',r.get('n_field_periods','?')),0)+1
pts=sorted(pts)
print('n distinct feasible',len(pts), n_nfp)
def front(ps):
    ps=sorted(ps); fr=[]; best=-1e9
    for a,l in ps:
        if l>best: fr.append((a,l)); best=l
    return fr
def hv(ps):
    fr=front([p for p in ps if p[0]<20 and p[1]>-1])
    tot=0
    for i,(a,l) in enumerate(fr):
        a2=fr[i+1][0] if i+1<len(fr) else 20
        tot+=(l+1)*(a2-a)
    return tot
fr=front(pts); print('front n',len(fr),'HV',hv(pts))
print('ARmin',fr[0],'ARmax',fr[-1])
# HV by AR band
bands=[(9.5,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17.2),(17.2,20)]
for lo,hi in bands:
    s=0
    for i,(a,l) in enumerate(fr):
        a2=fr[i+1][0] if i+1<len(fr) else 20
        x0=max(a,lo); x1=min(a2,hi)
        if x1>x0: s+=(l+1)*(x1-x0)
    print('band',lo,hi,'HV',round(s,2))
# what-ifs
def show(name,extra): print(name, round(hv(pts+extra),2), '+',round(hv(pts+extra)-hv(pts),2))
show('baseline-like pts (7.93,5.60),(9.98,8.45),(11.9,11.1)',[(7.93,5.60),(9.98,8.45),(11.9,11.1)])
show('nfp4 (10.70,9.07)',[(10.70,9.07)])
show('(9.0,6.5)',[(9.0,6.5)])
show('(8.0,5.6)',[(8.0,5.6)])
show('(7.0,4.5)',[(7.0,4.5)])
# front shifted up by +1 and +2 L
show('front +1 L',[(a,l+1) for a,l in fr])
show('front +2 L',[(a,l+2) for a,l in fr])
show('front x1.18 L (nfp4-like)',[(a,l*1.18) for a,l in fr])
show('front shifted -1 AR',[(a-1,l) for a,l in fr])
show('line L=1.215A-4.79 from 7 to 9.5',[(a/10,1.215*a/10-4.79) for a in range(70,95,2)])
show('high-AR extension (18,18),(19,19.5)',[(18,18),(19,19.5)])
# print front sample
for a,l in fr[::max(1,len(fr)//25)]: print(round(a,2),round(l,2))
print('---multipliers')
for m in [1.2,1.4,1.6,1.8,2.0,2.2]:
    print(m, round(hv([(a,l*m) for a,l in fr]),1))
print('---shift AR + mult')
for s in [0,1,2,3]:
    for m in [1.0,1.3,1.6]:
        print('shift',s,'mult',m, round(hv([(a-s,l*m) for a,l in fr]),1))
# baseline-only HV and ratio to line
print('baseline HV', hv([(7.93,5.60),(9.98,8.45),(11.9,11.1)]))
# our front L/A ratio
for a,l in [(9.53,7.13),(10.61,8.11),(12.02,9.78),(14.91,13.24),(17.16,16.37)]: print(a, round(l/a,3), 'axisym bound N(A-1)=',round(3*(a-1),1),'ratio',round(3*(a-1)/l,2))
