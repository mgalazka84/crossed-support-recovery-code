"""Check cycle signal margins and recompute paired intervals from saved outputs.
Run run_revision.py first to generate results_revision/paired_by_design.csv.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_revision'

def main():
    rng=np.random.default_rng(2026102610);checks=[]
    for m in (8,16,32,64):
        n=2*m;rr=np.repeat(np.arange(m),2)
        cc=np.column_stack((np.arange(m),(np.arange(m)+1)%m)).ravel()
        zr=np.eye(m)[rr];zc=np.eye(m)[cc];kr=zr@zr.T;kc=zc@zc.T
        raw=rng.normal(size=(n,4));raw-=raw.mean(axis=0)
        x=np.sqrt(n)*np.linalg.qr(raw)[0]
        for a,b in ((0.,1.),(.4,.7),(1.,1.)):
            sigma=np.eye(n)+a*kr+b*kc;ev=np.linalg.eigvalsh(sigma)
            wx=np.linalg.solve(sigma,x);h=x.T@wx;beta=.7
            gap=beta**2*min(h[0,0]-h[0,j]**2/h[j,j] for j in range(1,4))
            lower=beta**2*n/ev[-1];upper=beta**2*n/ev[0]
            assert lower-1e-10<=gap<=upper+1e-10
            checks.append(dict(n=n,a=a,b=b,g1=gap,lower=lower,upper=upper))
    pairs=pd.read_csv(OUT/'paired_by_design.csv');rows=[]
    p=pairs[(pairs.beta==.3)&(pairs.method=='Moment continuous')&(pairs.baseline=='Oracle')]
    old=pd.read_csv(OUT/'paired.csv')
    for name,g in p.groupby('scenario',sort=False):
        diff=g.difference.mean();se=np.sqrt((g.variance/g.repetitions).sum())/len(g)
        lo,hi=diff+np.array([-1,1])*1.96*se
        ref=old[(old.scenario==name)&(old.beta==.3)&(old.method=='Moment continuous')&(old.baseline=='Oracle')].iloc[0]
        assert np.allclose([diff,se,lo,hi],ref[['difference','se','low','high']].astype(float),rtol=1e-10,atol=1e-12)
        rows.append(dict(scenario=name,difference_pp=100*diff,low_pp=100*lo,high_pp=100*hi))
    OUT.mkdir(exist_ok=True)
    (OUT/'clarification_audit.json').write_text(json.dumps(dict(cycle_checks=checks,paired_continuous_minus_oracle=rows),indent=2)+'\n')
    print('Passed 12 cycle margin checks and 4 paired-interval reconstructions.')

if __name__=='__main__':main()
