"""Additional crossed-model experiments; deterministic seed, paired responses.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_revision.py.
Outputs are synthetic except the explicitly labelled lexdec illustration.
"""
from pathlib import Path
import itertools,json,time,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_irregular import residual_matrices,moments,grid,bound_constants
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results_revision";OUT.mkdir(exist_ok=True)
SEED=2026102609

class Workspace:
    def __init__(self,x,rr,cc):
        self.x=x;self.n,self.p=x.shape
        zr=np.eye(rr.max()+1)[rr];zc=np.eye(cc.max()+1)[cc]
        self.kr=zr@zr.T;self.kc=zc@zc.T
        z=np.column_stack([zr,zc])
        u,s,_=np.linalg.svd(z,full_matrices=False);self.u=u[:,s>1e-9]
        self.br=(self.u.T@zr)@(zr.T@self.u)
        self.bc=(self.u.T@zc)@(zc.T@self.u)
        self.ux=self.u.T@x
        self.xx=x.T@x-self.ux.T@self.ux
        q=np.linalg.qr(x,mode="reduced")[0]
        zr0=zr-q@(q.T@zr);zc0=zc-q@(q.T@zc)
        v,s,_=np.linalg.svd(np.column_stack([zr0,zc0]),full_matrices=False)
        self.v=v[:,s>1e-9];self.q=q
        self.ar=(self.v.T@zr0)@(zr0.T@self.v)
        self.ac=(self.v.T@zc0)@(zc0.T@self.v)
        self.df=self.n-self.p
        self.pairs=np.array(list(itertools.combinations(range(self.p),2)))
    def gram_rhs(self,y,ratios):
        a,b=ratios
        cf=cho_factor(np.eye(len(self.br))+a*self.br+b*self.bc,lower=True)
        uy=y@self.u
        gram=self.xx+self.ux.T@cho_solve(cf,self.ux)
        rhs=y@self.x-uy@self.ux+uy@cho_solve(cf,self.ux)
        energy=np.sum(y*y,axis=1)-np.sum(uy*uy,axis=1)+np.sum(uy*cho_solve(cf,uy.T).T,axis=1)
        return gram,rhs,energy
    def choose(self,y,ratios):
        g,r,_=self.gram_rhs(y,ratios);i,j=self.pairs.T
        scores=(g[j,j]*r[:,i]**2-2*g[i,j]*r[:,i]*r[:,j]+g[i,i]*r[:,j]**2)/(g[i,i]*g[j,j]-g[i,j]**2)
        return self.pairs[np.argmax(scores,axis=1)]
    def reml(self,y,start,upper=16.):
        vy=y@self.v;res=y-self.q@(self.q.T@y)
        outside=max(float(res@res-vy@vy),1e-12)
        def objective(ab):
            h=np.eye(len(self.ar))+ab[0]*self.ar+ab[1]*self.ac
            cf=cho_factor(h,lower=True);v=cho_solve(cf,vy)
            rss=outside+vy@v
            val=2*np.log(np.diag(cf[0])).sum()+self.df*np.log(rss/self.df)
            inv=cho_solve(cf,np.eye(len(h)))
            grad=np.array([np.sum(inv*k.T)-self.df*(v@k@v)/rss for k in [self.ar,self.ac]])
            return float(val),grad
        candidates=[minimize(objective,np.clip(st,0,upper),jac=True,method="L-BFGS-B",
                      bounds=[(0,upper)]*2,options={"ftol":1e-10,"gtol":1e-6,"maxiter":150})
                    for st in [start,[1.,1.],[0.,0.]]]
        fit=min(candidates,key=lambda a:a.fun)
        if not fit.success:
            retry=minimize(lambda ab:objective(ab)[0],fit.x,method="Powell",bounds=[(0,upper)]*2,options={"xtol":1e-9,"ftol":1e-11,"maxiter":300})
            if retry.success and retry.fun<=fit.fun+1e-5:fit=retry
        return fit.x,bool(fit.success),float(fit.fun)
    def bic(self,y,ratios,maximum=3):
        g,r,e=self.gram_rhs(y[None,:],ratios);best=(self.n*np.log(e[0]/self.n),())
        for k in range(1,maximum+1):
            for ids in itertools.combinations(range(self.p),k):
                ix=list(ids);u=r[0,ix];rss=max(e[0]-u@np.linalg.solve(g[np.ix_(ix,ix)],u),1e-12)
                val=self.n*np.log(rss/self.n)+k*np.log(self.n)
                if val<best[0]:best=(val,ids)
        return best[1]

def designs(seed):
    rng=np.random.default_rng(seed);nr,nc,p=20,24,12
    rr,cc=np.indices((nr,nc));rr=rr.ravel();cc=cc.ravel()
    raw=np.column_stack([rng.normal(size=(nr,4))[rr],rng.normal(size=(nc,4))[cc],rng.normal(size=(nr*nc,4))])
    raw[:,8:]=.6*raw[:,:4]+.8*raw[:,8:]
    uniform=rng.random(nr*nc)
    for imbalance in [False,True]:
        prob=.15+.7*rr/(nr-1) if imbalance else np.full(len(rr),.65)
        keep=uniform<prob;x=raw[keep].copy();x-=x.mean(axis=0);x*=np.sqrt(len(x))/np.linalg.norm(x,axis=0)
        yield x,rr[keep],cc[keep],keep

def synthetic(reps=300,ndesigns=5):
    amps=np.array([.1,.2,.3,.4,.6,.8]);active=np.array([0,9]);records=[];pairrows=[];sensitivity=[];audits=[]
    for di in range(ndesigns):
        ds=list(designs(SEED+di))
        for si,(name,idx,a,b) in enumerate([("Missing",0,1.,1.),("Imbalanced",1,1.,1.),("Zero row",0,0.,1.),("Small row",0,.02,1.)]):
            x,rr,cc,keep=ds[idx];w=Workspace(x,rr,cc);n=len(x);rng=np.random.default_rng(SEED+1000+di)
            # Shared underlying draws across variance scenarios and missingness patterns.
            noise=(np.sqrt(a)*rng.normal(size=(reps,20))[:,rr]+np.sqrt(b)*rng.normal(size=(reps,24))[:,cc]+rng.normal(size=(reps,480))[:,keep])
            _,f,aa=residual_matrices(x,w.kr,w.kc);est=moments(noise,aa)
            raw=np.zeros((reps,2));good=est[:,0]>0;raw[good]=est[good,1:]/est[good,0,None]
            clipped=np.clip(raw,0,4);grids=[grid(np.bincount(rr).max()),grid(np.bincount(cc).max())]
            rounded=np.column_stack([grids[j][np.searchsorted(grids[j],clipped[:,j])] for j in range(2)])
            oks={k:np.zeros((reps,len(amps)),bool) for k in ["Oracle","OLS","Moment continuous","Moment rounded","REML"]}
            converged=0;boundary=0;bic_ok=0;different=0;timer=time.perf_counter()
            sens_ok={u:np.zeros(reps,bool) for u in [1.,4.,16.]}
            for rep in range(reps):
                reml,success,_=w.reml(noise[rep],clipped[rep]);converged+=success;boundary+=bool(np.max(reml)>15.999)
                y=noise[rep]+amps[:,None]*x[:,active].sum(axis=1)
                for method,ratios in [("Oracle",(a,b)),("OLS",(0,0)),("Moment continuous",clipped[rep]),("Moment rounded",rounded[rep]),("REML",reml)]:
                    chosen=w.choose(y,ratios);oks[method][rep]=np.all(chosen==active,axis=1)
                for u in sens_ok:
                    ratios=np.clip(raw[rep],0,u)
                    gg=[grid(np.bincount(rr).max(),u),grid(np.bincount(cc).max(),u)]
                    rounded_u=[gg[j][np.searchsorted(gg[j],ratios[j])] for j in range(2)]
                    sens_ok[u][rep]=np.all(w.choose(y[2:3],rounded_u)[0]==active)
                chosen=w.bic(y[2],reml);bic_ok+=tuple(chosen)==tuple(active)
            for method,v in oks.items():
                for k,amp in enumerate(amps):
                    records.append(dict(design=di,scenario=name,n=n,beta=amp,method=method,repetitions=reps,successes=int(v[:,k].sum()),recovery=v[:,k].mean()))
            for method in ["Moment continuous","Moment rounded","REML","OLS"]:
                for base in ["Oracle","REML"]:
                    if method==base:continue
                    diff=oks[method].astype(int)-oks[base].astype(int)
                    for k,amp in enumerate(amps):
                        pairrows.append(dict(design=di,scenario=name,beta=amp,method=method,baseline=base,difference=diff[:,k].mean(),variance=diff[:,k].var(ddof=1),repetitions=reps,discordant=int(np.count_nonzero(diff[:,k]))))
            for u,v in sens_ok.items():sensitivity.append(dict(design=di,scenario=name,U=u,repetitions=reps,recovery=v.mean()))
            audits.append(dict(design=di,scenario=name,n=n,F_min=float(np.linalg.eigvalsh(f).min()),reml_converged=converged,reml_upper_boundary=boundary,reps=reps,bic_recovery_beta03=bic_ok/reps,seconds=time.perf_counter()-timer))
            pd.DataFrame(records).to_csv(OUT/"recovery_by_design.csv",index=False);pd.DataFrame(pairrows).to_csv(OUT/"paired_by_design.csv",index=False)
            pd.DataFrame(sensitivity).to_csv(OUT/"sensitivity.csv",index=False);(OUT/"optimization.json").write_text(json.dumps(audits,indent=2))
            print("finished",di,name,n,"REML",converged,"/",reps,flush=True)
    df=pd.DataFrame(records);agg=df.groupby(["scenario","beta","method"],sort=False).agg(recovery=("recovery","mean"),design_min=("recovery","min"),design_max=("recovery","max"),responses=("repetitions","sum")).reset_index();agg.to_csv(OUT/"recovery.csv",index=False)
    pairs=pd.DataFrame(pairrows);rows=[]
    for keys,v in pairs.groupby(["scenario","beta","method","baseline"],sort=False):
        mean=v.difference.mean();se=np.sqrt(np.sum(v.variance/v.repetitions))/len(v)
        rows.append(dict(zip(["scenario","beta","method","baseline"],keys))|dict(difference=mean,se=se,low=mean-1.96*se,high=mean+1.96*se,discordant=int(v.discordant.sum())))
    pd.DataFrame(rows).to_csv(OUT/"paired.csv",index=False)
    fig,axs=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for ax,(name,v) in zip(axs.ravel(),agg.groupby("scenario",sort=False)):
        for method,sub in v.groupby("method",sort=False):ax.plot(sub.beta,sub.recovery,".-",label=method)
        ax.set(title=name,xlabel="Nonzero coefficient",ylabel="Exact recovery",ylim=(0,1));ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);fig.savefig(OUT/"recovery.pdf");plt.close(fig)
    return agg

def real_data():
    import pyreadr
    data=pyreadr.read_r(str(ROOT/"data/lexdec.rda"))["lexdec"]
    print("real columns",list(data),flush=True)
    # Predictors selected from documented covariates, before model fitting.
    names=["Frequency","FamilySize","Length","Trial"]
    numeric=data[names].astype(float).to_numpy()
    native=(data.NativeLanguage.astype(str)=="English").to_numpy(float)
    sex=(data.Sex.astype(str)=="M").to_numpy(float)
    x=np.column_stack([numeric,native,sex]);names+=["NativeEnglish","Male"]
    y=data.RT.to_numpy(float);rr=pd.Categorical(data.Subject).codes;cc=pd.Categorical(data.Word).codes
    # Use all released log-RT observations; no response-based trimming.
    x-=x.mean(axis=0);x/=x.std(axis=0);x=np.column_stack([np.ones(len(y)),x])
    w=Workspace(x,rr,cc);_,f,aa=residual_matrices(x,w.kr,w.kc)
    est=moments(y[None,:],aa)[0];ratios=np.clip(est[1:]/est[0],0,16) if est[0]>0 else np.zeros(2)
    reml,success,_=w.reml(y,ratios,upper=16)
    out={"dataset":"languageR::lexdec","source":"https://github.com/cran/languageR/blob/master/data/lexdec.rda","sha256":hashlib.sha256((ROOT/"data/lexdec.rda").read_bytes()).hexdigest(),"n":len(y),"subjects":len(np.unique(rr)),"words":len(np.unique(cc)),"response":"released log RT; all observations","candidates":names,"reml_ratios":reml.tolist(),"reml_converged":success,"moment_ratios":ratios.tolist(),"F_min":float(np.linalg.eigvalsh(f).min()),"models":[]}
    for method,ab in [("OLS",[0,0]),("Moment continuous",ratios),("Moment rounded",[grid(np.bincount(rr).max(),16)[np.searchsorted(grid(np.bincount(rr).max(),16),ratios[0])],grid(np.bincount(cc).max(),16)[np.searchsorted(grid(np.bincount(cc).max(),16),ratios[1])]]),("REML",reml)]:
        g,r,e=w.gram_rhs(y[None,:],ab);best=(float("inf"),None,None)
        for k in range(7):
            for selected in itertools.combinations(range(1,7),k):
                ids=[0]+list(selected);coef=np.linalg.solve(g[np.ix_(ids,ids)],r[0,ids]);rss=e[0]-r[0,ids]@coef
                bic=len(y)*np.log(rss/len(y))+(k+1)*np.log(len(y))
                if bic<best[0]:best=(float(bic),selected,coef)
        out["models"].append({"method":method,"selected":[names[j-1] for j in best[1]],"size":len(best[1]),"BIC_within_method":best[0]})
    (OUT/"lexdec.json").write_text(json.dumps(out,indent=2))
    return out

def audit():
    rng=np.random.default_rng(819);worst=0.
    x,rr,cc,_=next(designs(SEED));w=Workspace(x,rr,cc)
    for i in range(12):
        ab=rng.uniform(0,3,2);y=rng.normal(size=(3,len(x)))
        g,r,e=w.gram_rhs(y,ab);cov=np.eye(len(x))+ab[0]*w.kr+ab[1]*w.kc
        wx=np.linalg.solve(cov,x);wy=np.linalg.solve(cov,y.T)
        worst=max(worst,np.max(np.abs(g-x.T@wx)),np.max(np.abs(r-y@wx)),np.max(np.abs(e-np.sum(y*wy.T,axis=1))))
    assert worst<1e-8
    (OUT/"audit.json").write_text(json.dumps({"dense_comparison_cases":12,"max_absolute_error":float(worst)}))
if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("--reps",type=int,default=300);a.add_argument("--designs",type=int,default=5);a.add_argument("--real-only",action="store_true");args=a.parse_args()
    if args.real_only:real_data()
    else:audit();synthetic(args.reps,args.designs);real_data()

