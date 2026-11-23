"""Generate the exact 2D PCGrad illustration; Matplotlib is only a build tool."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    panels = [
        ('Joint: add the original gradients', [
            ((2, 0), '#1479b8', r'$g_T=(2,0)$', (1.55, -.36)),
            ((-1, 1), '#ed7d19', r'$g_S=(-1,1)$', (-1.55, 1.25)),
            ((1, 1), '#6650a4', r'$g_{joint}=(1,1)$', (1.12, 1.15))],
            [((2,0),(1,1)),((-1,1),(1,1))]),
        ('PCGrad: project, then add', [
            ((1, 1), '#1479b8', r"$g'_T=(1,1)$", (1.12, .83)),
            ((0, 1), '#ed7d19', r"$g'_S=(0,1)$", (-1.3, 1.15)),
            ((1, 2), '#6650a4', r'$g_{PCGrad}=(1,2)$', (1.12, 2.12))],
            [((1,1),(1,2)),((0,1),(1,2))]),
    ]
    for ax, (title, vectors, addition) in zip(axes, panels):
        ax.set(xlim=(-1.7,2.9), ylim=(-.65,2.65), xticks=(-1,0,1,2), yticks=(0,1,2),
               xlabel='Shared-gradient coordinate 1', ylabel='Shared-gradient coordinate 2', title=title)
        ax.set_aspect('equal',adjustable='box')
        ax.grid(alpha=.2)
        ax.axhline(0,color='#aeb8c0',lw=1)
        ax.axvline(0,color='#aeb8c0',lw=1)
        for spine in ax.spines.values(): spine.set_visible(False)
        for a,b in addition:
            ax.plot([a[0],b[0]],[a[1],b[1]],color='#8b939a',ls='--',lw=1.2)
        for end,color,label,position in vectors:
            ax.annotate('',xy=end,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=2.8,color=color,mutation_scale=16))
            ax.text(*position,label,color=color,fontsize=11)
        ax.scatter([0],[0],s=14,color='#444444',zorder=5)
    fig.suptitle('A two-task example with sentiment weight λ = 1',fontsize=15,y=.98)
    fig.tight_layout(rect=(0,0,1,.93),w_pad=3)
    target=ROOT/'static/figures/pcgrad_2d_example.svg'
    fig.savefig(target,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(target)


if __name__ == '__main__': main()
