"""Exact small-design simulations for the residual-moment / grid selector.

All responses are synthetic. The design, incidence pattern, grid and residual
projection are fixed before responses. No experimental outcome is a proof.
"""
from __future__ import annotations
import itertools, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
SEED=20261026

def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(mid-half),float(mid+half)

def op(a):
    return float(np.max(np.abs(np.linalg.eigvalsh((a+a.T)/2))))

def root(a):
    v,u=eigh(a)
    return (u*np.sqrt(v))@u.T

def residual_matrices(x,kr,kc,reference=True):
    # Fixed working covariance: both ratios equal one, for every scenario.
    v=np.eye(len(x))+kr+kc if reference else np.eye(len(x))
    vi=cho_solve(cho_factor(v,lower=True),np.eye(len(x)))
    vx=vi@x
    q=vi-vx@np.linalg.solve(x.T@vx,vx.T)
    kernels=np.array([np.eye(len(x)),kr,kc])
    bs=np.array([q@k@q for k in kernels])
    f=np.einsum('iab,jba->ij',bs,kernels)
    aa=np.einsum('ij,jab->iab',np.linalg.inv(f),bs)
    return q,f,aa

def grid(delta,upper=4.,h=.25):
    vals=[0.,h/delta]
    while vals[-1]<upper: vals.append(vals[-1]*(1+h))
    return np.array(vals)

def moments(y,aa):
    return np.column_stack([np.sum((y@a)*y,axis=1) for a in aa])

def bss(y,x,g,pairs):
    wx=cho_solve(cho_factor(g,lower=True),x)
    gram=x.T@wx; u=y@wx
    i,j=pairs[:,0],pairs[:,1]
    a=gram[i,i]; b=gram[i,j]; c=gram[j,j]
    score=(c*u[:,i]**2-2*b*u[:,i]*u[:,j]+a*u[:,j]**2)/(a*c-b*b)
    return pairs[np.argmax(score,axis=1)]

def design(rng,imbalance=False,nr=20,nc=24,p=12):
    rr,cc=np.indices((nr,nc)); rr=rr.ravel();cc=cc.ravel()
    prob=(.15+.70*np.linspace(0,1,nr)[rr]) if imbalance else np.full(len(rr),.65)
    keep=rng.random(len(rr))<prob
    rr,cc=rr[keep],cc[keep]
    zr=np.eye(nr)[rr];zc=np.eye(nc)[cc]
    kr=zr@zr.T;kc=zc@zc.T
    n=len(rr)
    # Mixed row, column and cell predictors, plus paired correlations.
    raw=np.column_stack([zr@rng.normal(size=(nr,p//3)),
        zc@rng.normal(size=(nc,p//3)),rng.normal(size=(n,p//3))])
    for j in range(p//3):
        raw[:,2*p//3+j]=.6*raw[:,j]+.8*raw[:,2*p//3+j]
    raw-=raw.mean(axis=0)
    x=raw/np.linalg.norm(raw,axis=0)*np.sqrt(n)
    return x,kr,kc,rr,cc

def bound_constants(aa,g,kr,kc,a,b,delta=.1,blocks=1):
    sr=root(g); t=np.log(6/delta)
    ds=np.array([sr@v@sr for v in aa])
    # theta_0=1; repeated independent blocks reduce Frobenius/operator norms.
    z=np.array([2*np.linalg.norm(v,'fro')*np.sqrt(t/blocks)+2*op(v)*t/blocks for v in ds])
    wg=[float(eigh(k,g,eigvals_only=True)[-1]) for k in (kr,kc)]
    rho=((z[1]+a*z[0])*wg[0]+(z[2]+b*z[0])*wg[1])/(1-z[0]) if z[0]<1 else float('inf')
    return z,rho

def incomplete_experiment():
    streams=np.random.SeedSequence(SEED).spawn(6)
    records=[];diagnostics=[];paired=[]
    scenarios=[('Missing cells',False,1.,1.),('Row imbalance',True,1.,1.),
        ('Zero row variance',False,0.,1.),('Small row variance',False,.02,1.)]
    reps=3000; amplitudes=np.array([.05,.10,.15,.20,.30,.40,.60,.80])
    pairs=np.array(list(itertools.combinations(range(12),2))); active=np.array([0,9])
    for z,(name,imb,a,b) in enumerate(scenarios):
        rng=np.random.default_rng(streams[z])
        x,kr,kc,rr,cc=design(rng,imb)
        n=len(x); q,f,aa=residual_matrices(x,kr,kc)
        g=np.eye(n)+a*kr+b*kc
        noise=rng.normal(size=(reps,n))@np.linalg.cholesky(g).T
        est=moments(noise,aa)
        # QX=0, so these are exactly the estimates for every signal amplitude.
        ratios=np.zeros((reps,2)); good=est[:,0]>0
        ratios[good]=np.clip(est[good,1:]/est[good,0,None],0.,4.)
        grids=[grid(np.bincount(rr).max()),grid(np.bincount(cc).max())]
        ids=np.column_stack([np.searchsorted(grids[j],ratios[:,j],side='left') for j in range(2)])
        unique,inverse=np.unique(ids,axis=0,return_inverse=True)
        selected_g=[]; kappas=np.empty(reps)
        for v,ij in enumerate(unique):
            gv=np.eye(n)+grids[0][ij[0]]*kr+grids[1][ij[1]]*kc
            selected_g.append(gv)
            ev=eigh(g,gv,eigvals_only=True)
            kappas[inverse==v]=ev[-1]/ev[0]
        zs,rho=bound_constants(aa,g,kr,kc,a,b)
        _,_,plain_aa=residual_matrices(x,kr,kc,reference=False)
        plain_z,plain_rho=bound_constants(plain_aa,g,kr,kc,a,b)
        trace_duality=np.einsum('iab,jba->ij',aa,np.array([np.eye(n),kr,kc]))
        diagnostics.append(dict(scenario=name,n=n,rows=len(np.unique(rr)),columns=len(np.unique(cc)),
            row_min=int(np.bincount(rr).min()),row_max=int(np.bincount(rr).max()),
            column_min=int(np.bincount(cc).min()),column_max=int(np.bincount(cc).max()),
            gram_condition=float(np.linalg.cond(f)),residual_rank=int(np.linalg.matrix_rank(q)),
            grid_size=len(grids[0])*len(grids[1]),used_grid_points=len(unique),
            nonpositive_noise_estimates=int((~good).sum()),
            kappa_median=float(np.median(kappas)),kappa_q95=float(np.quantile(kappas,.95)),
            calibration_z=zs.tolist(),calibration_rho=None if not np.isfinite(rho) else float(rho),
            calibration_bound_nonvacuous=bool(zs[0]<1 and rho<1),
            identity_reference_z=plain_z.tolist(),
            identity_reference_rho=None if not np.isfinite(plain_rho) else float(plain_rho),
            duality_error=float(np.max(np.abs(trace_duality-np.eye(3)))),
            annihilation_error=float(np.max(np.abs(aa@x))),
            commutator_frobenius=float(np.linalg.norm(kr@kc-kc@kr,'fro'))))
        np.savez_compressed(OUT/f'design_{z}.npz',x=x,kr=kr,kc=kc,rows=rr,columns=cc,
            theta=np.array([1.,a,b]),active=active,seed=SEED)
        for beta in amplitudes:
            y=noise+beta*x[:,active].sum(axis=1)
            methods={'Oracle GLS':bss(y,x,g,pairs),'OLS':bss(y,x,np.eye(n),pairs)}
            pred=np.empty((reps,2),int)
            for v,gv in enumerate(selected_g):
                ix=np.flatnonzero(inverse==v)
                pred[ix]=bss(y[ix],x,gv,pairs)
            methods['Moment-grid GLS']=pred
            correct={}
            for method,sp in methods.items():
                ok=np.all(sp==active,axis=1); correct[method]=ok
                k=int(ok.sum());lo,hi=wilson(k,reps)
                missed=2-np.isin(sp,active).sum(axis=1)
                records.append(dict(scenario=name,n=n,p=12,s=2,row_variance=a,column_variance=b,
                    beta=float(beta),method=method,repetitions=reps,successes=k,recovery=k/reps,
                    ci_low=lo,ci_high=hi,mean_missed=float(missed.mean()),mean_false=float(missed.mean())))
            difference=correct['Moment-grid GLS'].astype(float)-correct['Oracle GLS']
            paired.append(dict(scenario=name,beta=float(beta),mean_difference=float(difference.mean()),
                standard_error=float(difference.std(ddof=1)/np.sqrt(reps)),
                discordant=int(np.sum(difference!=0))))
        print(name,n,'rho',rho,'kappa95',np.quantile(kappas,.95),flush=True)
    pd.DataFrame(records).to_csv(OUT/'irregular.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
    (OUT/'irregular_audit.json').write_text(json.dumps(diagnostics,indent=2))
    return pd.DataFrame(records)

def replicated_witness():
    rng=np.random.default_rng(20261027)
    x,kr,kc,rr,cc=design(rng,False,nr=6,nc=7,p=3)
    q,f,aa=residual_matrices(x,kr,kc)
    rec=[]
    for a,b in [(1.,1.),(0.,1.),(0.,0.)]:
        g=np.eye(len(x))+a*kr+b*kc
        for m in [1,4,16,64,256,1024,4096,16384]:
            z,rho=bound_constants(aa,g,kr,kc,a,b,blocks=m)
            kap=1.5*(1+rho)/(1-rho) if z[0]<1 and rho<1 else np.nan
            rec.append(dict(blocks=m,n=m*len(x),block_n=len(x),row_variance=a,column_variance=b,
                z0=z[0],zR=z[1],zC=z[2],rho=rho,kappa_bound=kap,
                usable=bool(z[0]<1 and rho<1),gram_condition=np.linalg.cond(f)))
    df=pd.DataFrame(rec);df.to_csv(OUT/'replicated_bound.csv',index=False)
    np.savez_compressed(OUT/'replicated_design.npz',x=x,kr=kr,kc=kc,rows=rr,columns=cc)
    return df

def audit_projection_and_rounding():
    rng=np.random.default_rng(20261028);maxdual=0.;maxround=0.;maxshape=0.
    residual_mismatches=0; maxresidual=0.
    for rep in range(50):
        x,kr,kc,rr,cc=design(rng,rep%2==0,nr=7,nc=8,p=3)
        _,_,aa=residual_matrices(x,kr,kc)
        kernels=np.array([np.eye(len(x)),kr,kc])
        dual=np.einsum('iab,jba->ij',aa,kernels)
        maxdual=max(maxdual,float(np.max(np.abs(dual-np.eye(3)))))
        theta=np.r_[1.,rng.uniform(0,4,2)];z=rng.uniform(0,.1,3)
        raw=theta+rng.uniform(-1,1,3)*z
        a,b=theta[1:];g=np.einsum('i,ijk->jk',theta,kernels)
        ratio=np.clip(raw[1:]/raw[0],0,4)
        gh=kernels[0]+ratio[0]*kr+ratio[1]*kc
        wg=[eigh(k,g,eigvals_only=True)[-1] for k in (kr,kc)]
        rho=((z[1]+a*z[0])*wg[0]+(z[2]+b*z[0])*wg[1])/(1-z[0])
        err=np.max(np.abs(eigh(gh,g,eigvals_only=True)-1))
        maxshape=max(maxshape,float(err-rho))
        grids=[grid(np.bincount(rr).max()),grid(np.bincount(cc).max())]
        up=[grids[i][np.searchsorted(grids[i],ratio[i])] for i in range(2)]
        gp=kernels[0]+up[0]*kr+up[1]*kc
        eig=eigh(gp,gh,eigvals_only=True)
        maxround=max(maxround,float(max(1-eig[0],eig[-1]-1.5)))
        if rep<10:
            # Independent implementation: whiten and solve each least-squares fit.
            import scipy.linalg as sla
            pairs=np.array(list(itertools.combinations(range(x.shape[1]),2)))
            y=rng.normal(size=(7,len(x)))
            chol=np.linalg.cholesky(gp)
            wy=sla.solve_triangular(chol,y.T,lower=True).T
            wx=sla.solve_triangular(chol,x,lower=True)
            resid=[]
            for ij in pairs:
                fit=wx[:,ij]@np.linalg.lstsq(wx[:,ij],wy.T,rcond=None)[0]
                resid.append(np.sum((wy.T-fit)**2,axis=0))
            direct=pairs[np.argmin(np.array(resid),axis=0)]
            residual_mismatches+=int(np.sum(np.any(direct!=bss(y,x,gp,pairs),axis=1)))
    ans=dict(configurations=50,max_duality_error=maxdual,max_rounding_violation=maxround,
        max_shape_bound_violation=maxshape,independent_gls_cases=70,
        independent_gls_support_mismatches=residual_mismatches)
    assert maxdual<1e-8 and maxround<1e-8 and maxshape<1e-8 and residual_mismatches==0
    (OUT/'algebra_audit.json').write_text(json.dumps(ans,indent=2))

def full_theorem_witness():
    data=np.load(OUT/'replicated_design.npz')
    x,kr,kc=data['x'],data['kr'],data['kc']
    _,f,aa=residual_matrices(x,kr,kc)
    a=b=1.;m=1024;alpha=delta=.05;h=.25;beta=1.
    g=np.eye(len(x))+kr+kc
    z,rho=bound_constants(aa,g,kr,kc,a,b,delta=delta,blocks=m)
    kap=(1+2*h)*(1+rho)/(1-rho)
    grids=[grid(np.bincount(data['rows']).max()),grid(np.bincount(data['columns']).max())]
    menu=len(grids[0])*len(grids[1]);t=np.log(2*menu*2/alpha)
    threshold=12*kap*t+4*kap*np.sqrt(2*t)+2*(kap-1)
    wx=cho_solve(cho_factor(g,lower=True),x);gram=x.T@wx
    gaps=[beta**2*(gram[0,0]-gram[0,j]**2/gram[j,j]) for j in [1,2]]
    margin=m*min(gaps)
    out=dict(m=m,n=m*len(x),p=3,s=1,beta=beta,alpha=alpha,delta=delta,
        claimed_error_upper=alpha+delta,grid_size=menu,calibration_z=z.tolist(),
        rho=float(rho),kappa=float(kap),g1=float(margin),required_g1=float(threshold),
        all_sufficient_conditions_met=bool(z[0]<1 and rho<1 and margin>threshold))
    assert out['all_sufficient_conditions_met']
    (OUT/'full_theorem_witness.json').write_text(json.dumps(out,indent=2))
    print('Full theorem witness',out,flush=True)

def plot(df,bound):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.right':False,'axes.spines.top':False})
    fig,axes=plt.subplots(2,2,figsize=(9.5,7),constrained_layout=True)
    colors={'Oracle GLS':'#23354b','Moment-grid GLS':'#1b807a','OLS':'#c77b39'}
    for ax,(name,sub) in zip(axes.ravel(),df.groupby('scenario',sort=False)):
        for method,v in sub.groupby('method',sort=False):
            ax.plot(v.beta,v.recovery,label=method,color=colors[method],lw=1.8)
            ax.fill_between(v.beta,v.ci_low,v.ci_high,color=colors[method],alpha=.12)
        ax.set(title=f'{name} (n={sub.n.iloc[0]})',xlabel='Nonzero coefficient',
            ylabel='Exact support recovery',ylim=(-.03,1.03));ax.grid(alpha=.16)
    axes[0,0].legend(fontsize=8)
    fig.savefig(OUT/'irregular.pdf');fig.savefig(OUT/'irregular.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.8),constrained_layout=True)
    for (a,b),sub in bound.groupby(['row_variance','column_variance'],sort=False):
        vals=sub.rho.where((sub.z0<1)&(sub.rho>0))
        ax.loglog(sub.blocks,vals,'o-',label=f'Row variance {a:g}, column {b:g}')
    ax.axhline(1,color='#999999',ls='--',label='Usability boundary')
    ax.set(xlabel='Independent copies of the incomplete crossing',ylabel='Analytic distortion radius rho')
    ax.legend(fontsize=8);ax.grid(alpha=.17)
    fig.savefig(OUT/'replicated_bound.pdf');fig.savefig(OUT/'replicated_bound.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    started=time.perf_counter()
    audit_projection_and_rounding()
    df=incomplete_experiment();bd=replicated_witness();full_theorem_witness();plot(df,bd)
    (OUT/'runtime.json').write_text(json.dumps({'seed':SEED,'repetitions':3000,
        'wall_seconds':time.perf_counter()-started,'numpy_version':np.__version__},indent=2))
    print(df[df.beta==.3][['scenario','method','recovery']].to_string(index=False),flush=True)
