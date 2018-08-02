import numpy as np
import pandas as pd
import sklearn as sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_and_plot(df, n_clusters=6, attributes=None):
    if not attributes:
        attributes = df.columns.tolist()[1:]

    (kmeans_stats, df_clustering) = kmeans_and_mean(df, n_clusters, attributes)

    fig = fig = plt.plot(14,14)
    kmeans_stats.iloc[:,:len(attributes)].plot.bar().legend(loc='upper left')
    plt.show()

    return kmeans_stats, df_clustering

def kmeans_and_mean(df, n_clusters=6, attributes=None):
    df_clustering = df[attributes]
    df_kmeans = KMeans(n_clusters=n_clusters,random_state=1000).fit(df_clustering)
    df_clustering['kmeans'] = df_kmeans.labels_

    kmeans_stats = df_clustering.groupby(['kmeans']).mean()

    return (kmeans_stats, df_clustering)


heroes_stats = pd.read_csv('heroes_stats_normalized.csv')
attributes = ['kills', 'deaths', 'assists', 'last_hits', 'hero_damage', 'hero_healing', 'tower_damage' ]

kmeans_stats, heroes_clustering = kmeans_and_plot(heroes_stats, n_clusters=5, attributes=attributes)

heroes_clustering.loc[(heroes_clustering['kmeans']==0),'heroclass'] = 'Fighter / Assassin'
heroes_clustering.loc[(heroes_clustering['kmeans']==1),'heroclass'] = 'Split Pusher'
heroes_clustering.loc[(heroes_clustering['kmeans']==2),'heroclass'] = 'Tank'
heroes_clustering.loc[(heroes_clustering['kmeans']==3),'heroclass'] = 'Carry'
heroes_clustering.loc[(heroes_clustering['kmeans']==4),'heroclass'] = 'Support'
