import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def modify_data_for_clustering(all_boroughs_data):

    attributes = ['LATITUDE', 'LONGITUDE']
    new_data = []
    for index, row in all_boroughs_data.iterrows():
        if(row['LATITUDE'] != 0):
            new_data.append([row['LATITUDE'], row['LONGITUDE']])

    return pd.DataFrame(new_data, columns=attributes, index=None)

def plot_data(new_data):

    colors = ['red', 'green', 'black', 'blue', 'yellow']
    idx=0
    for data in new_data:
        lat_list = data['LATITUDE'].tolist()
        print(lat_list)
        long_list = data['LONGITUDE'].tolist()
        plt.scatter(long_list, lat_list, s=4, color=colors[idx])
        idx+=1


    red_patch = mpatches.Patch(label='Queens', color='red')
    green_patch = mpatches.Patch(label='Bronx', color='green')
    black_patch = mpatches.Patch(label='Brooklyn', color='black')
    blue_patch = mpatches.Patch(label='Manhattan', color='blue')
    yellow_patch = mpatches.Patch(label='Staten Island', color='yellow')
    plt.legend([red_patch, green_patch, black_patch, blue_patch, yellow_patch], ('Queens', 'Bronx', 'Brooklyn', 'Manhattan', 'Staten Island'),
               loc='upper left', prop={'size':20})
    plt.ylim(40.5, 40.92)
    plt.xlim(-74.26, -73.7)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def perform_kmeans_clustering(new_data):

    my_list = []
    for index, row in new_data.iterrows():
        my_list.append([row['LONGITUDE'], row['LATITUDE']])

    my_data_array = np.array(my_list)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(my_data_array)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    x=new_data['LONGITUDE'].tolist()
    y=new_data['LATITUDE'].tolist()
    print("Centroids :- ", centroids)

    color_scheme = ['r.', 'g.', 'k.', 'b.', 'y.']
    for idx in range(len(my_data_array)):
        plt.plot(x[idx], y[idx], color_scheme[labels[idx]], markersize = 10)
        print('Happening', idx)

    plt.ylim(40.5, 40.92)
    plt.xlim(-74.26, -73.7)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def perform_dbscan_clustering(new_data):

    X = new_data.iloc[:, [0,1]].values
    print(X)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

    dbscan = DBSCAN(eps=0.005, min_samples=10)
    y = dbscan.fit(X)
    print('I am here')
    outliers_df = pd.DataFrame(X)
    print(Counter(y.labels_))
    print(outliers_df[y.labels_==-1])

    fig = plt.figure()
    ax=fig.add_axes([.1, .1, .7, .7])
    colors = y.labels_
    ax.scatter(X[:, 1], X[:, 0], c=colors, s=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def perform_agglomerative_clustering(new_data):
    pass

def main():

    all_boroughs_data = pd.read_csv('NY_collisions_filtered_data_by_date.csv')
    queens_data = modify_data_for_clustering(pd.read_csv('QUEENS.csv'))
    bronx_data = modify_data_for_clustering(pd.read_csv('BRONX.csv'))
    brooklyn_data = modify_data_for_clustering(pd.read_csv('BROOKLYN.csv'))
    manhattan_data = modify_data_for_clustering(pd.read_csv('MANHATTAN.csv'))
    staten_island_data = modify_data_for_clustering(pd.read_csv('STATEN_ISLAND.csv'))
    new_data = modify_data_for_clustering(all_boroughs_data)
    # print(queens_data)
    # plot_data([queens_data, bronx_data, brooklyn_data, manhattan_data, staten_island_data])
    perform_kmeans_clustering(queens_data)
    # perform_dbscan_clustering(bronx_data)
    # perform_dbscan_clustering(queens_data)
    # perform_dbscan_clustering(brooklyn_data)
    # perform_dbscan_clustering(manhattan_data)
    # perform_dbscan_clustering(staten_island_data)

    # new_data.to_csv('Modified_for_clustering.csv')


if __name__ == '__main__':
    main()