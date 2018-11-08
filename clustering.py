import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.decomposition import PCA
import constants as c
import matplotlib.pyplot as plt


def pca_num_comps(pca, thresh=c.pca_thresh):
    """
    keep components up to thresh variance
    """
    cum_explained = np.cumsum(pca.explained_variance_ratio_)
    return np.sum(cum_explained < thresh) + 1


def cluster_preprocess_pca(df):
    """
    fit transform pipeline for PCA
    """
    df2 = df.copy()
    df2['applications'] = df.applications.apply(np.log)
    pipe = make_pca_pipe(components=None)
    pipe.fit(df2)
    trans_mat = pipe.transform(df2)
    comps_keep = pca_num_comps(pipe.named_steps['pca'])
    return pipe, trans_mat[:, :comps_keep]


def search(X, num_clusters):
    """
    num_clusters is list of number of centers to search
    """
    models = []
    for num in num_clusters:
        amod = {}
        amod['mod'] = KMeans(n_clusters=num, n_jobs=1)
        amod['mod'].fit(X)
        preds = amod['mod'].predict(X)
        if num > 1:
            amod['score'] = silhouette_score(X, preds)
        else:
            amod['score'] = -1
        models.append(amod)
    return models


def filter_clusters(clusters, min_improvement=c.clust_min_improve):
    """
    only use the clustering model with the best score and those with fewer centers
    """
    scores = [m['score'] for m in clusters]
    bscore = scores[0]
    best_idx = 0
    for idx in range(1, len(scores)):
        score = scores[idx]
        if score > bscore *(1 + min_improvement):
            bscore = score
            best_idx = idx
        # print (idx, bscore, best_idx)
    # best_idx = np.argmax(scores)
    return clusters[:best_idx+1]


def make_pca_pipe(components=2):
    return make_pipeline(Imputer(), StandardScaler(), PCA(n_components=components))


def distance_to_euclidean(X):
    return np.linalg.norm(X, axis=1)


def make_pca_df(model, X, idx):
    """
    """
    clusters_assignments = change_cluster_numbers(np.ravel(model.predict(X))) + 1
    names = ['Composite {}'.format(n) for n in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=names, index=idx)
    df['Cluster'] = clusters_assignments
    return df


def closest_to_centroid(model, X, idx, top=10):
    """
    return observations closest to cluster centroids
    """
    distances_array = distance_to_euclidean(model.transform(X))
    clusters_assignments = np.ravel(model.predict(X)) + 1
    df = pd.DataFrame({'distances': distances_array, 'cluster': clusters_assignments}, index=idx)
    df['rnk'] = df.groupby('cluster').rank()
    df['num_clusters'] = np.max(clusters_assignments)
    return df.loc[df.rnk <= top, :]


def top_pca_influences(pca, df, top=4):
    """
    see what contributes to the two components
    """
    comp_df = pd.DataFrame(np.round(pca.components_, 4), columns = df.keys()).loc[0:top-1].applymap(np.abs)
    new_idx = ['Composite {}'.format(n+1) for n in range(0, top)]
    comp_df['component'] = new_idx
    return comp_df.set_index('component')


def change_cluster_numbers(array):
    """
    change order so that cluster number is in order of counts descending
    """
    clust, cnts = np.unique(array, return_counts=True)
    new_clusters = np.argsort(-cnts)
    mapping = {idx: i for idx, i in zip(clust[new_clusters], range(0, array.size))}
    func = np.vectorize(lambda v: mapping[v])
    return func(array)


# def weight_group(df, groupings=c.clust_wgts):
#     """
#     NOT USED
#     groupings is a list of dictionaries detailing which features
#         should be grouped together and weighted as a single feature
#     """
#     for dic in groupings:
#         total = np.sum(list(dic.values()))
#         for key in dic:
#             df[key] = df[key] * dic[key] / total
#     return df


# def cluster_preprocess(df):
#     """
#     NOT USED
#     """
#     df_wgt = weight_group(df)
#     pipe = make_pipeline(Imputer(), StandardScaler())
#     df_trans = pipe.fit_transform(df_wgt)
#     return pipe, df_trans
