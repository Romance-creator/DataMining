# Part 2: Cluster Analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster, metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    data = pd.read_csv(data_file)
    data = data.drop(['Channel', 'Region'], axis=1)
    return data


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    df0 = df.describe()
    df1 = df0.loc[['min', 'max']]
    df2 = df0.loc[['mean', 'std']]
    df2 = df2.round(0)
    result = pd.concat([df2, df1])
    result = result.T
    return result


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    df_standard = (df - df.mean()) / df.std()
    return df_standard

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
    kmeans = KMeans(n_clusters=k, init='random', max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit(df)
    label_array = labels.labels_
    label_series = pd.Series(label_array)
    return label_series


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10)
    labels = kmeans.fit(df)
    array = labels.labels_
    label_series = pd.Series(array)
    return label_series


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    clustering = AgglomerativeClustering(n_clusters=k).fit(df)
    label_array = clustering.labels_
    label_series = pd.Series(label_array)
    return label_series


# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X, y):
    score = silhouette_score(X, y)
    return score


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    df_standard = (df - df.mean()) / df.std()
    result = []
    Algorithm = ['Kmeans', 'Agglomerative']
    data = ['Original', 'Standard']
    for i in Algorithm:
        for j in data:
            for k in [3, 5, 10]:
                if data == 'Original':
                    X = df
                else:
                    X = df_standard
                if i == 'Kmeans':
                    for n in range(10):
                        m = cluster.KMeans(n_clusters=k, init='random')
                        m.fit(X)
                        y = pd.Series(m.labels_)
                        score = metrics.silhouette_score(X, y, metric='euclidean')
                        result.append([i, j, k, score])
                else:
                    a = cluster.AgglomerativeClustering(n_clusters=k, linkage='single', affinity='euclidean')
                    a.fit(X)
                    y = pd.Series(a.labels_)
                    score = metrics.silhouette_score(X, y, metric='euclidean')
                    result.append([i, j, k, score])
    ce = pd.DataFrame(columns=['Algorithm', 'data', 'k', 'Silhouette Score'], data=result)
    return ce


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    return max(rdf['Silhouette Score'])


# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
# reference :lab7_solution
def scatter_plots(df):
    array = np.array(df)
    CLUSTER_MARKERS = ['bo', 'rv', 'c^', 'm<', 'y>', 'ks', 'bp', 'r*', 'cD', 'mP']
    M = len(array)
    K = 3
    ac = cluster.AgglomerativeClustering(n_clusters=K, linkage='single', affinity='euclidean')
    ac.fit(array)
    for a in range(5):
        for b in range(6):
            if a < b:
                plt.figure()
                for j in range(M):
                    plt.plot(array[j][0], array[j][1], CLUSTER_MARKERS[ac.labels_[j]])
                plt.xlabel(str(a))
                plt.ylabel(str(b))
                plt.title('agglomerative clustering, K=' + str(K))
                plt.savefig('agg_k3_(' + str(a) + '-' + str(b) + ').png')
                plt.close()

