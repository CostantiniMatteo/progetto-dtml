import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def fit_and_plot(series, ax=None, deg=3, thresh=1.0):
    outliers = is_outlier(series, thresh=thresh)
    y = series[~outliers]
    x = list(range(len(y)))
    z = np.polyfit(x, y, deg)
    p = np.poly1d(z)
    xp = np.linspace(min(x),max(x),100)
    if ax:
        ax.plot(x, y, '.', xp, p(xp), '-')
    else:
        plt.plot(x, y, '.', xp, p(xp), '-')

    ax.set_ylim([min(y) - 0.05, max(y) + 0.05])
    ax.fill_between(xp, np.full(xp.shape, min(y) - 0.05),np.full(xp.shape, 0.5), alpha=0.1, color='red')
    ax.fill_between(xp, np.full(xp.shape,0.5), np.full(xp.shape, max(y) + 0.05), alpha=0.1, color='green')
    ax.axhline(y=0.5, linewidth=0.5, color='black', alpha=0.5)




# match_id, hero_name, won and stuff
a = pd.read_csv('matches_full.csv')
# match_id, duration
b = pd.read_csv('matches_stats.csv')
matches = pd.merge(a, b, on='match_id')
matches['lost'] = ~matches['won']
# Ther's a single match with over 16000s, the others are all less than 6673
matches = matches[matches.duration < 7000]
# Select only the needed column
matches = matches[['hero_name', 'duration', 'won', 'lost']]

# From 50s to 7000s with 5 minutes steps
ranges = np.arange(50, 7000, 300)

# Group by hero_name and duration range and count won and lost games
g = matches.groupby(['hero_name', pd.cut(matches.duration, ranges)]).sum()
# Win percentage
g['win_p'] = g['won'] / (g['won'] + g['lost'])
# Drop other columns and fills some NaN
g = g[['win_p']]
g = g.fillna(0)
g = g.reset_index()
g = g.groupby(['hero_name',])
n = g.ngroups

thresh = 1.0

f, axarr = plt.subplots(10, 11)

for i, (name, group) in tqdm(enumerate(g)):
    r, c = i // 11, i % 11
    ax = axarr[r, c]
    fit_and_plot(group.win_p, ax=ax, thresh=thresh, deg=3)
    ax.set_title(name,
        fontdict={
            'verticalalignment': 'baseline',
            'size': 8,
            # 'alpha': 0.3
        }
    )
    ax.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off',
        right='off',
        left='off')
    ax.set_yticklabels([])
    ax.set_xticklabels([])


plt.subplots_adjust(wspace=0, hspace=0.5)
# fig.tight_layout()

