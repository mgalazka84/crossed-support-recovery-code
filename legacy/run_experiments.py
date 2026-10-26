"""Reproducible numerical experiments for the EJS research note.

Gaussian balanced crossed models are simulated in their exact orthogonal
coordinates. This is an exact distributional reduction, not an approximation.
All results are synthetic. No experimental outcome establishes a proof.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 20261025

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 220,
    "axes.titleweight": "bold", "axes.labelcolor": "#23354b",
    "text.color": "#23354b", "axes.edgecolor": "#7a8591",
})


def wilson(success: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    phat = success / count
    den = 1 + z * z / count
    mid = (phat + z * z / (2 * count)) / den
    half = z / den * math.sqrt(phat * (1 - phat) / count + z * z / (4 * count * count))
    return mid - half, mid + half


def geometry_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """Equal N=3600; varying R,C; 12 orthogonal row or interaction predictors."""
    reps, p, s = 20000, 12, 2
    amplitudes = np.linspace(0.02, 1.00, 50)
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    colors = ["#1b807a", "#cf7a32", "#4266a8"]
    for ax, stratum in zip(axes, ["row", "interaction"]):
        amplitudes = np.linspace(.005,.16,80) if stratum == "interaction" else np.linspace(.02,1.,50)
        for (nr, nc), color in zip([(20, 180), (60, 60), (180, 20)], colors):
            n = nr * nc
            lam = 1.0 + nc if stratum == "row" else 1.0
            z = rng.standard_normal((reps, p))
            curve = []
            for b in amplitudes:
                a = z.copy()
                a[:, :s] += b * math.sqrt(n / lam)
                success = np.min(np.abs(a[:, :s]), axis=1) > np.max(np.abs(a[:, s:]), axis=1)
                count = int(success.sum())
                lo, hi = wilson(count, reps)
                curve.append(count / reps)
                records.append(dict(experiment="geometry", stratum=stratum, rows=nr,
                    columns=nc, n=n, p=p, s=s, beta=float(b), eigenvalue=lam,
                    information=n/lam, repetitions=reps, successes=count,
                    recovery=count/reps, ci_low=lo, ci_high=hi))
            ax.plot(amplitudes, curve, color=color, lw=2, label=f"R={nr}, C={nc}")
        ax.set(xlabel="Nonzero coefficient b", ylabel="Exact support recovery",
               title=f"{stratum.capitalize()} predictors", ylim=(-0.02, 1.02))
        if stratum == "interaction":
            ax.set_xlim(0.02, 0.16)
        ax.grid(alpha=.17)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Same 3,600 observations; information depends on predictor direction", fontsize=12)
    fig.savefig(OUT / "geometry.png")
    fig.savefig(OUT / "geometry.pdf")
    plt.close(fig)
    df = pd.DataFrame(records)
    df.to_csv(OUT / "geometry.csv", index=False)
    return df


def bss_two(z: np.ndarray, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact size-two BSS; shared or replicate-specific diagonal precision."""
    pairs = np.asarray(list(itertools.combinations(range(x.shape[1]), 2)))
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, z.shape)
    h = (z * weights) @ x
    a = weights @ (x * x)
    cross = np.column_stack([weights @ (x[:, i] * x[:, j]) for i, j in pairs])
    i, j = pairs[:, 0], pairs[:, 1]
    det = a[:, i] * a[:, j] - cross * cross
    if np.any(det <= 0):
        raise ValueError("Rank-deficient candidate support")
    scores = (a[:, j] * h[:, i]**2 - 2*cross*h[:, i]*h[:, j] +
              a[:, i]*h[:, j]**2) / det
    return pairs[np.argmax(scores, axis=1)]


def mixed_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """Mixed strata; variance calibration uses independent residual coordinates
    of the SAME Gaussian data vector. Residual df are stated explicitly.
    """
    reps, nr, nc, m = 6000, 40, 40, 6
    n, p = nr*nc, 3*m
    true_s = np.asarray([0, 13])
    lambdas = np.repeat([1+nc, 1+nr, 1], m).astype(float)
    dfs = np.asarray([nr-1-m, nc-1-m, (nr-1)*(nc-1)-m])
    # Euclidean-orthogonal columns, rotated across row/interaction eigenspaces.
    # They are generally NOT orthogonal under the true Mahalanobis metric.
    rot = np.eye(p)
    angle = 0.40
    c, sn = math.cos(angle), math.sin(angle)
    for j in range(m):
        k = 2*m+j
        rot[j,j], rot[k,k], rot[j,k], rot[k,j] = c, c, -sn, sn
    x = math.sqrt(n) * rot
    znoise = rng.standard_normal((reps, p)) * np.sqrt(lambdas)
    raw = rng.chisquare(dfs, (reps, 3)) / dfs * np.asarray([1+nc, 1+nr, 1])
    estimates = raw.copy()
    estimates[:,0] = np.maximum(raw[:,0], raw[:,2])
    estimates[:,1] = np.maximum(raw[:,1], raw[:,2])
    feasible = 1 / np.repeat(estimates, m, axis=1)
    methods = {"Oracle GLS": 1/lambdas,
               "Residual-calibrated GLS": feasible,
               "OLS (ignores random effects)": np.ones(p)}
    records = []
    paired_records = []
    for b in np.linspace(.025, .75, 30):
        paired_good = {}
        beta = np.zeros(p)
        beta[true_s] = b
        z = znoise + x @ beta
        for name, w in methods.items():
            chosen = bss_two(z, x, w)
            good = np.all(chosen == true_s, axis=1)
            paired_good[name] = good
            count = int(good.sum())
            lo, hi = wilson(count, reps)
            records.append(dict(experiment="mixed", method=name, beta=float(b),
                rows=nr, columns=nc, n=n, p=p, s=2, angle=angle,
                df_row=int(dfs[0]), df_column=int(dfs[1]), df_interaction=int(dfs[2]),
                repetitions=reps, successes=count, recovery=count/reps,
                ci_low=lo, ci_high=hi))
        difference = paired_good["Residual-calibrated GLS"].astype(float)-paired_good["Oracle GLS"]
        se = difference.std(ddof=1)/np.sqrt(reps)
        paired_records.append(dict(beta=float(b),difference=float(difference.mean()),se=float(se),low=float(difference.mean()-1.96*se),high=float(difference.mean()+1.96*se),discordant=int(np.count_nonzero(difference))))
    pd.DataFrame(paired_records).to_csv(OUT / "mixed_paired.csv",index=False)
    df = pd.DataFrame(records)
    df.to_csv(OUT / "mixed.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    for name, color in zip(methods, ["#1b807a", "#4266a8", "#cf7a32"]):
        v = df[df.method == name]
        ax.plot(v.beta, v.recovery, label=name, color=color, lw=2)
        ax.fill_between(v.beta, v.ci_low, v.ci_high, color=color, alpha=.12)
    ax.set(xlabel="Nonzero coefficient b", ylabel="Exact support recovery",
           title="Two active predictors mixing row and interaction directions",
           ylim=(-.02, 1.02))
    ax.grid(alpha=.17)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(OUT / "mixed.png")
    fig.savefig(OUT / "mixed.pdf")
    plt.close(fig)
    return df


def adverse_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """Deliberately unrestricted precision estimate; NOT a crossed-family fit."""
    reps = 100000
    records = []
    for n in [16, 64, 256, 1024, 4096, 16384, 65536]:
        eta = n**(-.25)
        amplitude = n**.125
        y1 = amplitude + rng.standard_normal(reps)
        y2 = rng.standard_normal(reps)
        r = np.sqrt(rng.chisquare(n-2, reps))
        good_oracle = y1*y1 > y2*y2
        good_bad = y1*y1 > (y2+eta*r)**2
        for name, good in [("Oracle", good_oracle), ("Unrestricted adaptive metric", good_bad)]:
            count = int(good.sum())
            lo,hi=wilson(count,reps)
            records.append(dict(experiment="counterexample", method=name, n=n,
                eta=eta, amplitude=amplitude, covariance_operator_error=eta/(1-eta),
                repetitions=reps, successes=count,recovery=count/reps,ci_low=lo,ci_high=hi))
    df=pd.DataFrame(records)
    df.to_csv(OUT/"counterexample.csv",index=False)
    fig, ax=plt.subplots(figsize=(7.4,3.6),constrained_layout=True)
    for name,color in [("Oracle","#1b807a"),("Unrestricted adaptive metric","#ba4b49")]:
        v=df[df.method==name]
        ax.plot(v.n,v.recovery,"o-",lw=2,label=name,color=color)
    ax.set(xscale="log",xlabel="Ambient dimension n",ylabel="Exact support recovery",
           ylim=(-.02,1.02),title="Operator consistency alone does not protect adaptive selection")
    ax.grid(alpha=.17)
    ax.legend(fontsize=9)
    fig.savefig(OUT/"counterexample.png")
    fig.savefig(OUT/"counterexample.pdf")
    plt.close(fig)
    return df


def numerical_audit(rng: np.random.Generator) -> dict:
    """Independent algebra checks, not tests reproducing an assertion by definition."""
    worst={"gap_identity":0.,"rank_excess":0.,"trace_violation":0.,
           "norm_violation":0.,"linear_variance_violation":0.,"bss_score_error":0.}
    for rep in range(250):
        n,p,s=19,8,3
        z=rng.standard_normal((n,n))
        sigma=z@z.T/n+np.eye(n)
        L=np.linalg.cholesky(sigma)
        U,_=qr(rng.standard_normal((n,n)))
        eig=rng.uniform(.7,1.3,n)
        invL=np.linalg.inv(L)
        W=invL.T@U@np.diag(eig)@U.T@invL
        x=rng.standard_normal((n,p))
        S=np.array([0,1,2]); T=np.array([0,3,4])
        mu=x[:,S]@rng.standard_normal(s)
        def H(support):
            a=x[:,support]
            return W@a@np.linalg.solve(a.T@W@a,a.T@W)
        D=H(S)-H(T)
        a=x[:,T]
        residual=mu-a@np.linalg.solve(a.T@W@a,a.T@W@mu)
        gap=residual@W@residual
        A=L.T@D@L
        b=L.T@D@mu
        lower,upper=float(eig.min()),float(eig.max())
        k=2
        worst["gap_identity"]=max(worst["gap_identity"],abs(mu@D@mu-gap))
        worst["rank_excess"]=max(worst["rank_excess"],np.linalg.matrix_rank(A,tol=1e-9)-2*k)
        worst["trace_violation"]=max(worst["trace_violation"],-(upper-lower)*k-np.trace(A))
        worst["norm_violation"]=max(worst["norm_violation"],np.linalg.norm(A,2)-upper)
        worst["linear_variance_violation"]=max(worst["linear_variance_violation"],b@b-upper*gap)
    # Direct full-dimensional GLS versus compressed spectral experiment.
    nr,nc,m=7,8,2
    nr0=nr*nc
    def centered_basis(size):
        return qr(np.column_stack([np.ones(size),rng.standard_normal((size,size-1))]))[0][:,1:]
    ar,bc=centered_basis(nr),centered_basis(nc)
    Qr=np.kron(ar[:,:m],np.ones((nc,1))/math.sqrt(nc))
    Qc=np.kron(np.ones((nr,1))/math.sqrt(nr),bc[:,:m])
    Qe=np.column_stack([np.kron(ar[:,j],bc[:,j]) for j in range(m)])
    Q=np.column_stack([Qr,Qc,Qe])
    xx=math.sqrt(nr0)*Q
    sigma=np.eye(nr0)+np.kron(np.eye(nr),np.ones((nc,nc)))+np.kron(np.ones((nr,nr)),np.eye(nc))
    W=np.linalg.inv(sigma)
    lam=np.repeat([1+nc,1+nr,1.],m)
    for _ in range(25):
        yy=np.linalg.cholesky(sigma)@rng.standard_normal(nr0)
        comp=Q.T@yy
        pick=bss_two(comp[None,:],math.sqrt(nr0)*np.eye(3*m),1/lam)[0]
        full_scores=[]
        pairs=list(itertools.combinations(range(3*m),2))
        for pair in pairs:
            a=xx[:,pair]
            h=a.T@W@yy
            full_scores.append(h@np.linalg.solve(a.T@W@a,h))
        assert np.array_equal(pick,pairs[int(np.argmax(full_scores))])
        worst["bss_score_error"]=max(worst["bss_score_error"],
            abs(max(full_scores)-sum(comp[pick]**2/lam[pick])))
    for name,value in worst.items():
        assert value < 1e-8, (name,value)
    audit={"seed":SEED,"matrix_identity_cases":250,"full_vs_compressed_cases":25,
           "max_errors":{k:float(v) for k,v in worst.items()},"passed":True}
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    return audit


def main():
    streams=np.random.SeedSequence(SEED).spawn(4)
    audit=numerical_audit(np.random.default_rng(streams[0]))
    geometry=geometry_experiment(np.random.default_rng(streams[1]))
    mixed=mixed_experiment(np.random.default_rng(streams[2]))
    adverse=adverse_experiment(np.random.default_rng(streams[3]))
    summary={"seed":SEED,"audit":audit,
        "geometry_nearest_beta_0_30":geometry.iloc[(geometry.beta-.30).abs().groupby(
            [geometry.stratum,geometry.rows]).idxmin()].to_dict(orient="records"),
        "mixed_at_beta_0_30":mixed[np.isclose(mixed.beta,.30)].to_dict(orient="records"),
        "counterexample_last":adverse[adverse.n==adverse.n.max()].to_dict(orient="records"),
        "maximum_binomial_standard_error":{"geometry":.5/math.sqrt(20000),
            "mixed":.5/math.sqrt(6000),"counterexample":.5/math.sqrt(100000)}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
