"""Draw exact sequential three-task PCGrad vectors; only a build-time tool."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
G = np.array([[2.,0.],[-1.,1.],[-2.,1.]])
ORDERS = [[1,2],[0,2],[0,1]]


def project(orders):
    result = G.copy()
    for i, peers in enumerate(orders):
        for j in peers:
            dot = result[i] @ G[j]
            if dot < 0: result[i] -= dot/(G[j]@G[j])*G[j]
    return result


def main():
    projected = project(ORDERS)
    reversed_rows = project([[2,1],[2,0],[1,0]])
    np.testing.assert_allclose(projected, [[.6,1.2],[0,1],[0,1]])
    np.testing.assert_allclose(projected.mean(0), [1/5,16/15])
    np.testing.assert_allclose(reversed_rows.mean(0), [2/15,14/15])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(2,2,figsize=(12,9.6),layout='constrained')
    colors=['#2166AC','#C56624','#39826C']
    def arrow(ax,end,label,color,offset=(5,5),style='-'):
        ax.annotate('',xy=end,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=2.4,color=color,linestyle=style))
        ax.annotate(label,xy=end,xytext=offset,textcoords='offset points',color=color,fontsize=11)
    for ax in axes.flat:
        ax.set(xlim=(-2.7,2.8),ylim=(-.55,2.0),xticks=[-2,-1,0,1,2],yticks=[0,.5,1,1.5],
               xlabel='Gradient coordinate 1',ylabel='Gradient coordinate 2')
        ax.set_aspect('equal');ax.grid(alpha=.18)
        ax.axhline(0,color='#AEB8C0',lw=1);ax.axvline(0,color='#AEB8C0',lw=1)
        for spine in ax.spines.values():spine.set_visible(False)
    a=axes[0,0];a.set_title('1. Compute and keep all original gradients')
    for g,label,c,offset in zip(G,['$g_A=(2,0)$','$g_B=(-1,1)$','$g_C=(-2,1)$'],colors,[(0,12),(5,12),(-40,12)]):
        arrow(a,g,label,c,offset)
    a=axes[0,1];a.set_title('2. For A: visit B, then C')
    arrow(a,G[0],'Start: (2, 0)','#9AA4AC',(0,-18),':')
    arrow(a,[1,1],'After B: (1, 1)','#966D22',(6,-12),'--')
    arrow(a,[.6,1.2],'After C: (3/5, 6/5)',colors[0],(-25,18))
    a.plot([2,1,.6],[0,1,1.2],color='#7E8891',ls=':',lw=1.5,marker='o',ms=3)
    a=axes[1,0];a.set_title('3. Project B and C independently')
    arrow(a,G[1],'Original B',colors[1],(0,12),':')
    arrow(a,G[2],'Original C',colors[2],(-30,12),':')
    arrow(a,[0,1],'Both finish at (0, 1)','#604C91',(12,15))
    a.text(-2.4,-.4,'B: A then C;   C: A then B',fontsize=10,color='#526170')
    a=axes[1,1];a.set_title('4. Average only after all projections finish')
    arrow(a,G.mean(0),'Joint: (-1/3, 2/3)','#808890',(-105,15),':')
    arrow(a,projected.mean(0),'PCGrad: (1/5, 16/15)','#604C91',(10,18))
    arrow(a,reversed_rows.mean(0),'Reversed peer orders: (2/15, 14/15)','#39826C',(12,-16),'--')
    fig.suptitle('Three gradients in 2D: project each task against ORIGINAL peers, then average',fontsize=14)
    target=ROOT/'static/figures/cifar/pcgrad_three_gradients.svg'
    fig.savefig(target,facecolor='white');plt.close(fig)
    print('Built three-gradient PCGrad diagram; exact arithmetic verified.')


if __name__=='__main__':main()
