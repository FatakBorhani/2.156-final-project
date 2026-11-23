import json, glob, os
import numpy as np, pandas as pd

SCR = os.path.dirname(os.path.abspath(__file__))
REPO = '/Users/leifakerley/Documents/Repo/2.156-final-project'
FT = REPO + '/data/feasibility_tests'
CNAMES = ['iota', 'qi', 'mirror', 'flux', 'well']

def violations_from_metrics(iota, qi, mirror, flux, well):
    iota = np.abs(iota)
    return np.stack([(0.25 - iota) / 0.25,
                     (np.log10(qi) + 3.5) / 3.5,
                     (mirror - 0.25) / 0.25,
                     (flux - 0.9) / 0.9,
                     (0 - well) / 0.1], axis=-1)

def spectral_features(rc, zs, nfp):
    rc = np.asarray(rc, float); zs = np.asarray(zs, float)
    M, N2 = rc.shape; N = (N2 - 1) // 2
    R0 = rc[0, N]
    a = 0.5 * (abs(rc[1, N]) + abs(zs[1, N])) if M > 1 else np.nan
    E = rc**2 + zs**2
    E[0, N] = 0.0
    mm, nn = np.meshgrid(np.arange(M), np.arange(-N, N + 1), indexing='ij')
    tot = E.sum()
    f = {}
    f['R00'] = R0
    f['a_proxy'] = a
    f['b_over_a10'] = abs(zs[1, N]) / abs(rc[1, N]) if M > 1 and rc[1, N] != 0 else np.nan
    f['axis_exc'] = np.sqrt(E[0].sum()) / R0  # m=0, n!=0
    f['axis_R_exc'] = np.sqrt((rc[0]**2).sum() - R0**2) / R0
    f['axis_Z_exc'] = np.sqrt((zs[0]**2).sum()) / R0
    m1 = E[1].copy(); m1n0 = m1[N]; m1[N] = 0
    f['m1_helical'] = np.sqrt(m1.sum()) / a if a else np.nan
    f['m1_over_m0'] = np.sqrt(m1.sum()) / np.sqrt(E[0].sum()) if E[0].sum() > 0 else np.nan
    for k in range(M):
        f[f'Efrac_m{k}'] = E[k].sum() / tot
    f['Efrac_m_ge2'] = E[mm >= 2].sum() / tot
    f['Efrac_m_ge3'] = E[mm >= 3].sum() / tot
    f['Efrac_n_ge2'] = E[np.abs(nn) >= 2].sum() / tot
    f['Efrac_n_ge3'] = E[np.abs(nn) >= 3].sum() / tot
    f['rms_hi_over_a'] = np.sqrt(E[(mm >= 2) | (np.abs(nn) >= 3)].sum()) / a if a else np.nan
    # spectral width: energy-weighted mean of sqrt(m^2 + n^2)
    f['spec_width'] = (E * np.sqrt(mm**2 + nn**2)).sum() / tot
    f['Mmax'] = M - 1; f['Nmax'] = N
    return f

def boundary_xsection_elong(rc, zs, nfp, nphi=16, ntheta=128):
    """Max elongation of the cylindrical-phi cross-sections via polygon second moments."""
    rc = np.asarray(rc, float); zs = np.asarray(zs, float)
    M, N2 = rc.shape; N = (N2 - 1) // 2
    th = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    phis = np.linspace(0, np.pi/nfp, nphi)
    best = 0.0
    for ph in phis:
        R = np.zeros_like(th); Z = np.zeros_like(th)
        for m in range(M):
            for j, n in enumerate(range(-N, N+1)):
                ang = m*th - n*nfp*ph
                if rc[m, j]: R += rc[m, j]*np.cos(ang)
                if zs[m, j]: Z += zs[m, j]*np.sin(ang)
        # polygon second moments
        x, y = R, Z; x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x*y1 - x1*y
        A = cr.sum()/2
        if A == 0: continue
        cx = ((x + x1)*cr).sum()/(6*A); cy = ((y + y1)*cr).sum()/(6*A)
        Ixx = ((y**2 + y*y1 + y1**2)*cr).sum()/12 - A*cy**2
        Iyy = ((x**2 + x*x1 + x1**2)*cr).sum()/12 - A*cx**2
        Ixy = ((x*y1 + 2*x*y + 2*x1*y1 + x1*y)*cr).sum()/24 - A*cx*cy
        ev = np.linalg.eigvalsh(np.array([[Iyy, Ixy], [Ixy, Ixx]]) * np.sign(A))
        if ev.min() > 0:
            best = max(best, np.sqrt(ev.max()/ev.min()))
    return best

def load_ours():
    rows = []
    for f in sorted(glob.glob(FT + '/*_ledger.jsonl')):
        name = os.path.basename(f).replace('_ledger.jsonl', '')
        if name.endswith('_dry') or 'dry' in name: continue
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get('kind') != 'hi' or not r.get('ok'): continue
            b = r.get('boundary')
            if b is None: continue
            b = json.loads(b) if isinstance(b, str) else b
            v = r['violations']
            d = dict(run=name, start=r.get('start'), feasible=bool(r.get('feasible')), worst=r.get('worst'),
                     AR=r['aspect_ratio'], LgB=r['L_gradB'], nfp=b['n_field_periods'])
            for c, x in zip(CNAMES, v): d['v_' + c] = x
            d['rc'] = b['r_cos']; d['zs'] = b['z_sin']
            rows.append(d)
    return pd.DataFrame(rows)

def hypervolume(points_negL_AR, ref=(1.0, 20.0)):
    """2D minimization HV; points are (-L, AR)."""
    P = np.array([p for p in points_negL_AR if p[0] < ref[0] and p[1] < ref[1]])
    if len(P) == 0: return 0.0
    P = P[np.argsort(P[:, 0])]
    hv = 0.0; best_y = ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y); best_y = y
    return hv

def front_points(L, AR):
    """Nondominated (max L, min AR) points sorted by AR ascending."""
    o = np.argsort(AR); fr = []; best = -np.inf
    for i in o:
        if L[i] > best: fr.append((AR[i], L[i])); best = L[i]
    return np.array(fr)

def hv_gain(L, AR, front, ref=(1.0, 20.0)):
    """Exclusive HV gained by adding each point (L, AR) to the given feasible front."""
    # staircase in x=-L: g(x) = min AR among front pts with -L_f <= x, else ref AR
    fx = -front[:, 1]; fy = front[:, 0]
    o = np.argsort(fx); fx = fx[o]; fy = np.minimum.accumulate(fy[o])
    xs = np.r_[fx, ref[0]]
    gain = np.zeros(len(L))
    x_p = -np.asarray(L); y_p = np.asarray(AR)
    # segments: (-inf,fx0): g=ref; [fx_k, fx_{k+1}): g=fy_k
    seg_lo = np.r_[-np.inf, fx]; seg_hi = np.r_[fx, ref[0]]; seg_g = np.r_[ref[1], fy]
    for lo, hi, g in zip(seg_lo, seg_hi, seg_g):
        a = np.maximum(lo, x_p); b = np.minimum(hi, ref[0])
        w = np.clip(b - a, 0, None); h = np.clip(np.minimum(g, ref[1]) - y_p, 0, None)
        gain += w * h
    return np.where((x_p < ref[0]) & (y_p < ref[1]), gain, 0.0)
