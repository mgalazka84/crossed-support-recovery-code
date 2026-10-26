"""Deterministic explanatory diagrams; no simulated or empirical observations.

Run: python make_explanatory_figures.py
Optional: --output-dir DIR --preview-dir DIR
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parent / "figures" / "explanatory")
parser.add_argument("--preview-dir", type=Path)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
if args.preview_dir:
    args.preview_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9.5,
                     "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, INK, GRAY = "#245A7A", "#B65B2B", "#202A33", "#C7CDD2"

def text(ax, x, y, s, size=9.5, **kw):
    return ax.text(x, y, s, ha=kw.pop("ha", "center"),
                   va=kw.pop("va", "center"), fontsize=size, color=kw.pop("color", INK), **kw)

def arrow(ax, a, b, color=INK, style="-", rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                 lw=1.1, color=color, linestyle=style,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))

def box(ax, x, y, w, h, title, detail="", color=BLUE):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, lw=1.1))
    text(ax, x+w/2, y+h*(.66 if detail else .5), title, 9.5)
    if detail:
        text(ax, x+w/2, y+h*.26, detail, 9.2, color=color)

def save(fig, name):
    fig.savefig(args.output_dir / (name+".pdf"), bbox_inches="tight", pad_inches=.04)
    if args.preview_dir:
        fig.savefig(args.preview_dir / (name+".png"), dpi=170, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

# 1. A conceptual example of incomplete crossed observations and response reuse.
fig, ax = plt.subplots(figsize=(7.2, 3.05))
fig.subplots_adjust(0, 0, 1, 1); ax.set(xlim=(0, 7.2), ylim=(0, 3.05)); ax.axis("off")
text(ax, 1.2, 2.85, "Crossed observations", 11, weight="bold")
text(ax, 4.4, 2.85, "The same scores have two uses", 11, weight="bold")
mask = np.array([[1,1,0,1,1],[1,0,1,1,0],[0,1,1,0,1],[1,1,0,1,1]])
sx, sy, step = .57, 1.0, .34
ax.add_patch(Rectangle((sx-.16, sy+step-.16), 5*step, .32,
                      facecolor=BLUE, alpha=.12, edgecolor="none"))
ax.add_patch(Rectangle((sx+3*step-.16, sy-.16), .32, 4*step,
                      facecolor=ORANGE, alpha=.13, edgecolor="none"))
for i in range(4):
    for j in range(5):
        x,y=sx+j*step,sy+(3-i)*step
        ax.plot(x,y,"o" if mask[i,j] else "x", ms=5 if mask[i,j] else 4,
                color=INK if mask[i,j] else GRAY, mew=1)
text(ax, 1.25, 2.32, "Texts", 9.5)
text(ax, .15, 1.55, "Readers", 9.5, rotation=90)
text(ax, 1.25, .48, "Shared reader and text effects\ncreate dependence; some pairs are missing.", 8.8)
arrow(ax, (2.21,1.6), (2.78,1.6))
text(ax, 2.68, 1.95, r"scores $Y$", 10)
box(ax, 3.12, 1.98, 2.25, .53, "Estimate shared noise", "Residual covariance calibration")
box(ax, 3.12, .73, 2.25, .53, "Choose relevant predictors", "Size-constrained GLS selection")
arrow(ax, (2.82,1.6), (3.14,2.23))
arrow(ax, (2.82,1.6), (3.14,.99))
arrow(ax, (4.24,1.97), (4.24,1.29), color=ORANGE)
text(ax, 4.4, 1.63, "estimated weights", 8.6, ha="left", color=ORANGE)
arrow(ax, (5.37,.99), (5.85,.99))
text(ax, 6.3, 1.5, r"Selected set $\widehat S$", 10)
for j in range(5):
    ax.add_patch(Rectangle((5.82+j*.21, .91), .14, .14,
                          facecolor=BLUE if j in (1,3) else "white", edgecolor=BLUE, lw=.8))
text(ax, 6.31, .5, "Goal: recover the\ntrue set of predictors", 8.7)
text(ax, 4.44, .16, "The guarantee must account for this reuse of the response.", 9)
save(fig, "overview")

# 2. Exact two-dimensional geometry: two rival one-dimensional model spaces.
fig, axes = plt.subplots(1,2,figsize=(7.2,3.2))
fig.subplots_adjust(left=.05,right=.98,top=.83,bottom=.12,wspace=.3)
mu=np.array([1.,1.]); sigma=np.diag([4.,.25]); white=np.diag([.5,2.])
assert np.allclose(white @ sigma @ white, np.eye(2))
assert np.allclose((white @ mu)**2, [.25,4.])
for ax, point, title, lab1, lab2 in [
    (axes[0],mu,"Equal Euclidean gaps",r"gap$^2=1$",r"gap$^2=1$"),
    (axes[1],white@mu,"Different covariance-weighted gaps",r"$d_{T_1}^2=4$",r"$d_{T_2}^2=1/4$")]:
    ax.set(xlim=(-.16,2.35),ylim=(-.16,2.35),aspect="equal");ax.axis("off")
    ax.plot([0,2.18],[0,0],color=BLUE,lw=2)
    ax.plot([0,0],[0,2.18],color=ORANGE,lw=2)
    x,y=point
    ax.plot([x,x],[0,y],color=BLUE,ls="--",lw=1.3)
    ax.plot([0,x],[y,y],color=ORANGE,ls=":",lw=1.6)
    ax.plot(x,y,"o",color=INK,ms=5)
    text(ax,1.1,2.62,title,10.3,weight="bold")
    text(ax,2.05,-.15,r"$T_1$",10,color=BLUE)
    text(ax,-.14,2.11,r"$T_2$",10,color=ORANGE)
    text(ax,x+.15,y+.05,r"$\mu$" if ax is axes[0] else r"$\Sigma^{-1/2}\mu$",10,ha="left")
    text(ax,x+.1,y*.45,lab1,10,color=BLUE,ha="left")
    text(ax,x*.5,y+.21,lab2,10,color=ORANGE)
text(axes[0],1.1,-.44,"Refit each rival model before measuring its residual.",8.7)
text(axes[1],1.1,-.44,"Whitening measures residuals in noise units.",8.7)
save(fig,"support_geometry")

# 3. The actual geometric grid in the paper, with an illustrative ratio estimate.
fig=plt.figure(figsize=(7.2,3.95))
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7.2),ylim=(0,3.95));ax.axis("off")
text(ax,1.78,3.76,"Residual calibration to support selection",10.5,weight="bold")
steps=[("1  Fix the residual space",r"$MX=0$; residual Gram matrix $F\succ0$"),
       ("2  Estimate variance components",r"$\widehat\theta_i=Y^\top A_iY$"),
       ("3  Clip ratios and round upward",r"$(\widehat a,\widehat b)\longmapsto(a_+,b_+)$"),
       ("4  Compare size-$s$ supports",r"$W_+=G(a_+,b_+)^{-1}$")]
for j,(title,detail) in enumerate(steps):
    y=2.9-j*.7
    box(ax,.2,y,3.16,.52,title,detail)
    if j<3:arrow(ax,(1.78,y),(1.78,y-.17))
gx=fig.add_axes([.57,.235,.4,.60])
h=.5;degree=4;upper=1.;levels=[0.,h/degree]
while levels[-1]<upper:levels.append(levels[-1]*(1+h))
levels=np.array(levels);raw=np.array([-.25,.75]);clipped=np.clip(raw,0,upper)
rounded=np.array([0. if u==0 else levels[levels>=u][0] for u in clipped])
assert np.allclose(rounded,[0.,.94921875])
gx.add_patch(Rectangle((0,0),1,1,facecolor=GRAY,alpha=.22,edgecolor=INK,lw=.8))
aa,bb=np.meshgrid(levels,levels);gx.scatter(aa.ravel(),bb.ravel(),s=7,color=GRAY,zorder=2)
gx.annotate("",xy=clipped,xytext=raw,arrowprops={"arrowstyle":"->","color":ORANGE,"lw":1.2})
gx.annotate("",xy=rounded,xytext=clipped,arrowprops={"arrowstyle":"->","color":BLUE,"lw":1.2})
gx.plot(*raw,"x",ms=6,color=ORANGE,mew=1.5)
gx.plot(*clipped,"o",ms=5,color=ORANGE,mfc="white",mew=1.2)
gx.plot(*rounded,"*",ms=9,color=BLUE)
for label,pt,where,col in [("raw ratios",raw,(-.28,1.24),ORANGE),
                           ("clipped",clipped,(.42,.52),ORANGE),
                           ("rounded",rounded,(.4,1.08),BLUE)]:
    gx.annotate(label,xy=pt,xytext=where,color=col,fontsize=8.8,
                arrowprops={"arrowstyle":"-","color":col,"lw":.8})
gx.text(.58,.12,"prescribed box",fontsize=8,color=INK,ha="center")
gx.set(xlim=(-.4,1.51),ylim=(-.06,1.51),xlabel=r"row variance ratio $a$",ylabel=r"column variance ratio $b$")
gx.set_xticks([0,1]);gx.set_yticks([0,1]);gx.tick_params(labelsize=8)
gx.set_title("A fixed product grid",fontsize=10.5,fontweight="bold",pad=10)
text(ax,3.6,.24,r"Under the theorem conditions: calibration failure $\delta$ + comparison failure $\alpha$",9.4)
save(fig,"residual_pipeline")

# 4. Balanced covariance directions and the independent Gaussian projections.
fig=plt.figure(figsize=(7.2,3.55));ax=fig.add_axes([0,0,1,1])
ax.set(xlim=(0,7.2),ylim=(0,3.55));ax.axis("off")
text(ax,1.75,3.35,"Three covariance directions",10.6,weight="bold")
text(ax,5.42,3.35,"Two roles inside each stratum",10.6,weight="bold")
v=np.array([1,1,-1,-1]);masks=[np.tile(v[:,None],(1,4)),np.tile(v,(4,1)),np.outer(v,v)]
assert all(np.sum(m)==0 and np.sum(m*m)==16 for m in masks)
assert all(np.sum(masks[i]*masks[j])==0 for i in range(3) for j in range(i))
labels=[("Row",r"$\lambda_R$",r"$\sigma_e^2+C\sigma_a^2$"),
        ("Column",r"$\lambda_C$",r"$\sigma_e^2+R\sigma_b^2$"),
        ("Interaction",r"$\lambda_E$",r"$\sigma_e^2$")]
for j,(mask,(name,lam,formula)) in enumerate(zip(masks,labels)):
    x=.18+1.12*j
    m=fig.add_axes([x/7.2,1.92/3.55,.92/7.2,.92/3.55]);m.imshow(mask,cmap=ListedColormap([ORANGE,BLUE]),vmin=-1,vmax=1,interpolation="nearest")
    m.set_xticks([]);m.set_yticks([])
    for spine in m.spines.values():spine.set_visible(False)
    text(ax,x+.46,3.0,name,9.2)
    text(ax,x+.46,1.68,lam+" =",10)
    text(ax,x+.46,1.38,formula,10)
text(ax,1.75,.92,r"If $\|x\|^2=n$, its information is $n/\lambda_g$.",9.8)
text(ax,1.75,.56,"Colors indicate opposite signs, not data values.",8.5)
text(ax,4.82,2.96,"Compare supports",9,color=BLUE)
text(ax,6.23,2.96,"Estimate noise",9,color=ORANGE)
for j,g in enumerate(["R","C","E"]):
    y=2.39-j*.51
    text(ax,3.8,y+.16,"$"+g+"$",10)
    box(ax,4.1,y,1.35,.33,r"$\Pi_"+g+r"Y$",color=BLUE)
    box(ax,5.62,y,1.35,.33,r"$Q_"+g+r"Y$",color=ORANGE)
text(ax,5.55,1.03,r"$\Pi_VY\;\perp\;(Q_RY,Q_CY,Q_EY)$",11)
text(ax,5.55,.63,r"$\widehat\lambda_g=\|Q_gY\|^2/\nu_g$",11,color=ORANGE)
text(ax,3.6,.12,"Independent calibration coordinates remove the grid union bound in the balanced case.",9.2)
save(fig,"balanced_split")
print("Created four deterministic explanatory figures.")
