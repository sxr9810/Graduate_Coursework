
"""

Title: Agglomerative Clustering
Author: Sailee Rumao (sxr9810)
Date: 10/31/2018

"""

# Approach:

#1. Created a class to store all the clusters as objects.
#2. Calculated euclidean distance and its minimum for all objects.
#3. Merged the two clusters with minimum distance and repeated this process until there was one cluster left in the lost.
#4. Correlation plot
#5. Dendogram.



#Importing packages

import pandas as pd
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.cluster.hierarchy import dendrogram,linkage



def main():

    # importing data
    shopping_data = pd.read_csv("D:/APP STATS/720/Homework/HW_06/HW_AG_SHOPPING_CART_v805.csv")

    #calling the function for object creation
    clusters,clusters_original = cluster_object_list(shopping_data)

    #calling the perform_all function.
    perform_all(clusters, clusters_original)


    # correlation matrix
    correlation_values = shopping_data.iloc[:, 1:].corr()
    # correlation_values.to_csv("correaltion_matrix.csv")

    # Correlation plot
    sns.heatmap(correlation_values,
                xticklabels=correlation_values.columns,
                yticklabels=correlation_values.columns)
    plt.title('Correlation Plot')
    plt.show()


    # Dendogram

    linkage_distance = linkage(shopping_data.iloc[:, 1:], method='centroid', metric='euclidean')
    dendrogram(linkage_distance, truncate_mode='lastp')
    plt.title('Agglomerative clustering Dendogram')
    plt.xlabel('ID')
    plt.ylabel('distance')
    plt.show()




class Cluster_class:
    """
        create a class for an object(cluster) that stores:
        1. guest id
        2. list of the 12 attributes
        3. distance between a cluster and other clusters (initially empty)
        """


    def __init__(self, guest_id,merged_list_id,attributes,distance):

        self.guest_id = guest_id
        self.merged_list_id = merged_list_id
        self.attributes = attributes
        self.distance = distance


def cluster_object_list(shopping_data):

    """
        Defining objects for clusters based on the class created above,Cluster_class.
        :param shopping_data: The given shopping data
        :return: A list cluster that contains 337 objects (clusters) with parameters guest_id and attributes obtained by parsing shopping data using pandas
        distance is an empty list initially.
        """

    distance = []
    cluster = []
    cluster_original = []
    for index,row in shopping_data.iterrows():
        cluster_index = Cluster_class(row[0],[row[0]],row[1:13],distance)
        cluster_duplicate_id = Cluster_class(row[0],[row[0]],row[1:13],distance)
        cluster += [cluster_index]
        cluster_original += [cluster_duplicate_id] #Creating duplicate of the cluster.

    return cluster,cluster_original


def distance(object, other_object):
    """
        This function calculates the euclidean distance for all attributes in the data. (used in the function euclidean_distance)
        :param object:It is the cluster (at a time) with respect to which we want to find the euclidean distance for the attributes.
        :param other_object: These are all the other clusters to find the distance with respect to the cluster above.
        :return:returns the distance for all objects(clusters) calculated with repsect to the other objects (clusters)
        """

    dist = 0
    for ind_x in range(len(object.attributes)):
        dist += math.pow((object.attributes[ind_x]- other_object.attributes[ind_x]),2) #computing euclidean distance
    dist = math.sqrt(dist)
    return dist


def Euclidean_distance(clusters):
    """
        This function uses the distance function to calculate distance for all attributes of all objects.
        The final list of distances is appended to distance of respective object in clusters (list of objects)
        :param clusters: List of objects(clusters)
        :return: list of clusters appended with distances for each object.
        """

    for object in clusters:
        euclidean = []
        for other_object in clusters:
            if other_object.guest_id == object.guest_id:
                attribute_distance = math.inf #setting the distance to be infinite for object under consideration.
            else:
                attribute_distance = distance(object, other_object)
            euclidean = euclidean + [attribute_distance]
        object.distance = euclidean

    return clusters



def Minimum_distance_clusters(clusters):
    """
    This function is used to find the clusters that have minimum distance.
    :param clusters:the list of all objects(clusters).
    :return: The two clusters which have minimum distance.(which are to be merged).

    """

    Minimum_dist = math.inf
    for my_obj in clusters:
        for idx in range(len(my_obj.distance)):
            if(my_obj.distance[idx] < Minimum_dist):
                Minimum_dist = my_obj.distance[idx]
                min_object_1 = my_obj
                min_object_2 = clusters[idx]

    return min_object_1,  min_object_2


def agglomerative(clusters,clusters_original,object_1,object_2):
    """
    These function merges the two clusters with minimum distance.It appends the merged cluster id and
     stores in a new list which is added to the object class.merged_list_id.
     It removes the two individual clusters_ids that are merged from cluster.guest_id.
     we still have the entire original cluster data in clusters_original(list of objects).

    :param clusters: the list of objects(clusters) to be modified(after merging).
    :param clusters_original:duplicate of clusters.
    :param object_1:Object with minimum distance obtained from the function Minimum_distance_clusters.
    :param object_2:The other Object with minimum distance obtained from the function Minimum_distance_clusters.


    :return: The modified cluster list.
             The size of the smallest of the two clusters.
    """
    distance = []
    object_id_attributes = []

    merged_list = object_1.merged_list_id + object_2.merged_list_id #merging the two attributes
    for x in range(len(clusters[0].attributes)):
        object_center = 0
        for ix in merged_list:
            object_center += (clusters_original[ix-1].attributes[x])
        object_center = object_center/len(merged_list)    #recomputing centers.
        object_id_attributes += [object_center]



    clusters += [Cluster_class(min(object_1.guest_id,object_2.guest_id),object_1.merged_list_id+object_2.merged_list_id
                               ,object_id_attributes, distance)]

    clusters.remove(object_1)
    clusters.remove(object_2)


    min_size = min(len(object_1.merged_list_id), len(object_2.merged_list_id))

    return clusters, min_size

def perform_all(clusters, clusters_original):

    """
    This function performs agglomerative clustering until there is one cluster left.
    :return: The Minimum size list (of smallest of the clusters)
             The list of centers in the last and second last cluster.
             The list of cluster id's in the last two cluster.
    """

    size_list = []
    clusters_list = []
    clusters_list_ids =[]
    while(len(clusters)>1):
        clusters = Euclidean_distance(clusters)
        min_object_1, min_object_2 = Minimum_distance_clusters(clusters)
        clusters_list += [[min_object_1.attributes,min_object_2.attributes]]
        clusters_list_ids += [[min_object_1.merged_list_id,min_object_2.merged_list_id]]
        clusters, min_size = agglomerative(clusters, clusters_original, min_object_1, min_object_2)
        size_list += [min_size]


    print('Minimum size list',size_list)
    print('The second last cluster is list :', clusters_list[-2])
    print('The cluster centers for the last cluster is :',clusters_list[-1])
    print('The cluster ids are:',clusters_list_ids[-1])
    print('The cluster ids are:',clusters_list_ids[-2])



if __name__=='__main__':
    main()