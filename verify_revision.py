"""Independent numerical checks of the added algebra and compressed likelihood."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh
from run_revision import Workspace,designs
from run_irregular import residual_matrices
rng=np.random.default_rng(260919);worst=dict(gap_identity=0.,continuity=0.,envelope=0.,reml_objective=0.)
for rep in range(60):
    n,p=23,5;x=rng.normal(size=(n,p));v=rng.normal(size=(n,n));sigma=np.eye(n)+v@v.T/n
    W=np.linalg.inv(sigma);c=x[:,[0]];i=x[:,1];j=x[:,2]
    ri=i-c@np.linalg.solve(c.T@W@c,c.T@W@i);rj=j-c@np.linalg.solve(c.T@W@c,c.T@W@j)
    mu=x[:,0]+.7*i;t=x[:,[0,2]]
    residual=mu-t@np.linalg.solve(t.T@W@t,t.T@W@mu)
    expected=.7**2*(ri@W@ri)*(1-(ri@W@rj)**2/((ri@W@ri)*(rj@W@rj)))
    worst["gap_identity"]=max(worst["gap_identity"],abs(residual@W@residual-expected))
    z=rng.normal(size=(n,n));kr=z[:,:4]@z[:,:4].T;kc=z[:,4:8]@z[:,4:8].T
    a,b=rng.uniform(.1,2,2);eta=.01
    g=np.eye(n)+a*kr+b*kc;gp=g+eta*np.eye(n)
    wg=np.linalg.inv(gp);w=np.linalg.inv(g);y=mu+rng.normal(size=n)
    def rss(weight,cols):
        xx=x[:,cols];r=y-xx@np.linalg.solve(xx.T@weight@xx,xx.T@weight@y);return r@weight@r
    lhs=rss(w,[0,2])-rss(w,[0,1])
    rhs=rss(wg,[0,2])-rss(wg,[0,1])-2*eta*rss(wg,[0,1])
    worst["continuity"]=max(worst["continuity"],rhs-lhs)
    errors=rng.uniform(.01,.3,3);lo=np.clip((np.array([a,b])-errors[1:])/(1+errors[0]),0,4);hi=np.clip((np.array([a,b])+errors[1:])/(1-errors[0]),0,4)
    raw=np.array([1,a,b])+rng.uniform(-1,1,3)*errors;ab=np.clip(raw[1:]/raw[0],0,4)
    lower=np.eye(n)+lo[0]*kr+lo[1]*kc;upper=np.eye(n)+hi[0]*kr+hi[1]*kc;gh=np.eye(n)+ab[0]*kr+ab[1]*kc
    worst["envelope"]=max(worst["envelope"],-np.linalg.eigvalsh(gh-lower)[0],-np.linalg.eigvalsh(upper-gh)[0])
x,rr,cc,_=next(designs(2026102609));w=Workspace(x,rr,cc)
for rep in range(12):
    y=rng.normal(size=len(x));ab=rng.uniform(0,3,2);g=np.eye(len(x))+ab[0]*w.kr+ab[1]*w.kc
    wi=np.linalg.inv(g);gram=x.T@wi@x;rhs=x.T@wi@y;rss=y@wi@y-rhs@np.linalg.solve(gram,rhs)
    dense=np.linalg.slogdet(g)[1]+np.linalg.slogdet(gram)[1]-np.linalg.slogdet(x.T@x)[1]+w.df*np.log(rss/w.df)
    h=np.eye(len(w.ar))+ab[0]*w.ar+ab[1]*w.ac;vy=y@w.v;res=y-w.q@(w.q.T@y)
    comp=np.linalg.slogdet(h)[1]+w.df*np.log((res@res-vy@vy+vy@np.linalg.solve(h,vy))/w.df)
    worst["reml_objective"]=max(worst["reml_objective"],abs(dense-comp))
assert max(worst.values())<1e-8,worst
Path("results_revision/theory_audit.json").write_text(json.dumps({"random_geometric_cases":60,"dense_reml_cases":12,"max_errors":worst},indent=2))
print(worst)

