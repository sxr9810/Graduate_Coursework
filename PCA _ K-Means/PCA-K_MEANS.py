
"""
Title:PCA and K-Means
Author: Sailee Rumao
Date:11/14/2018

"""

##Import Packages
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main():

    ## 1. reading in the data
    data = pd.read_csv("D:/APP STATS/720/Homework/HW_08/HW_AG_SHOPPING_CART_v805.csv")
    #deleting ID (assigned attribute)
    data = data.drop("ID",axis=1)

    #Centering the data
    centered_data = data - data.mean()

    # 2. Finding the Covariance matrix
    Covariance = data.cov()

    # 3. Finding the Eigen values and the corresponding eigen vectors
    eigen_vector_1, eigen_vector_2, eigen_values,sorted_list = eigen(Covariance)

    # 4. Sorted list of eigen values
    print(sorted_list)

    # 5. Normalizing the Eigen values
    cumulative_list = normalized_eigen_values(sorted_list)

    #Plotting the scree plot
    scree_plot(cumulative_list)

    # 6. Printing the first two eigen vectors
    print('Eigen_Vector_1', eigen_vector_1)
    print('Eigen_Vector_2', eigen_vector_2)

    # 7. Projecting the original Agglomeration data on the first two Eigen Vectors.

    #First Principle component
    PC1 = centered_data.mul(eigen_vector_1,axis = 1)
    PC_1 = PC1.sum(axis=1)

    #second principle component
    PC2 = centered_data.mul(eigen_vector_2,axis = 1)
    PC_2 =PC2.sum(axis=1)

    #Plotting the first two principle components
    Scatter_plot(PC_1,PC_2)

    # 8. & 9. K-means Clustering
    centroids = K_means(PC_1,PC_2)
    print('The centroids of the 3 clusters are','\n',centroids)

    # 10. Re-projection of centroids to the first two Eigen vectors
    centroid_re_projection(data,centroids, eigen_vector_1, eigen_vector_2)


def eigen(Covariance):
    """
    This function computes the eigen values and corresponding eigen vectors from the covariance matrix.

    :param Covariance: Covariance matrix created above.(line 39)
    :return: The first two eigen vectors and all the eigen values
    """

    eigen_values, eigen_vectors = np.linalg.eig(Covariance.values)
    eigen_vectors = eigen_vectors.transpose()

    #sorting the eigen values from highest to lowest and storing their indices.
    sorted_list = -np.sort(-eigen_values)
    sorted_index = np.argsort(-eigen_values)

    return eigen_vectors[sorted_index[0]],eigen_vectors[sorted_index[1]],eigen_values,sorted_list


def normalized_eigen_values(sorted_list):
    """
    This function normalizes the sorted Eigen values and computes the cumulative sum of these normalized eigen values.
    :param eigen_values:The eigen values obtained from the function eigen above
    :return:List of cumulative sum of the normalized eigen values.
    """

    #Initializing
    standarized_eigen_values_list = []
    summation_eigen_values = sum(sorted_list)
    cum_sum = 0
    cumulative_list = [0]

    #loop to compute normalized eigen values.
    for eig_value in range(len(sorted_list)):
        normalized_eig = sorted_list[eig_value]/summation_eigen_values
        standarized_eigen_values_list += [normalized_eig]

    #loop to store the list of cumulative sum of the normalized values.
    for normalized_eig_value in range(len(standarized_eigen_values_list)):
        cum_sum += standarized_eigen_values_list[normalized_eig_value]
        cumulative_list += [cum_sum]

    return cumulative_list


def scree_plot(cumulative_list):
    """
    This function is used to plot the Cumulative sum of normalized eigen values(also known as the scree plot).
    :param cumulative_list: list of cumulative sum of the normalized values computed
    in the function normalized_eigen_values above
    :return: The scree plot
    """
    PC = [0,1,2,3,4,5,6,7,8,9,10,11,12]

    plt.style.use('ggplot')
    plt.plot(PC,cumulative_list,'o',color='black',linestyle='dashed')
    plt.xlabel("Principle Components")
    plt.ylabel("Proportion of Cumulative Variance")
    plt.title("Scree Plot")
    plt.show()


def Scatter_plot(PC_1,PC_2):
    """
    This function is used to plot the Projected data points obtained by projecting them on the two eigen vectors.
    :param PC_1: The data points obtained by projection on the first eigen vector.
    :param PC_2: he data points obtained by projection on the second eigen vector.
    :return: Scatter plot of the projected data.(plot of the two principle components obtained after projection)
    """

    plt.style.use("ggplot")
    plt.scatter(PC_1,PC_2)
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle component 2")
    plt.title("Scatter plot of projected data")
    plt.show()


def K_means(PC_1,PC_2):
    """
    This function performs K-means clustering on the new projected data
    :param PC_1:The data points obtained by projection on the first eigen vector.
    :param PC_2:The data points obtained by projection on the second eigen vector.
    :return: The centers of the three clusters.
    """

    #Concatenating PC1 and PC2 to a dataframe.
    KMeans_data = pd.concat([PC_1,PC_2],axis=1,keys=['PC_1','PC_2'])

    #creating a object of 3 clusters
    kmeans = KMeans(n_clusters=3)
    #Function that performs the K-means clustering on the data
    kmeans.fit(KMeans_data)

    #Extracting labels from the K-means clusters obtained.
    labels = kmeans.predict(KMeans_data)
    #Extracting the center co-ordinates of each cluster
    centroids = kmeans.cluster_centers_

    #Mapping the labels obtained to three colors using map and lambda function.
    colmap = {1:'r',2:'g',0:'b'}
    colors = map(lambda X :colmap[X],labels)
    colors1 = list(colors)

    #plotting the new projected data points and coloring by labels.
    plt.scatter(KMeans_data['PC_1'],KMeans_data['PC_2'],color=colors1)
    plt.xlabel("Principle component 1")
    plt.ylabel("Principle component 2")
    plt.title("K-Means Clustering")

    #Plotting the centroids of each clusters on the respective clusters.
    for indx,centroid in enumerate(centroids):
        plt.scatter(*centroid,color= 'black')  #* is used to take each co-ordinates in centroid(tuple) as one point.
    plt.show()

    return centroids


def centroid_re_projection(data,centroids,eigen_vector_1,eigen_vector_2):
    """
    This function finds the prototype amounts of each cluster based obtained by re-projecting
    centroid and the two eigen vectors and adding the respective means of the attributes.
    :param data:
    :param centroids: The center points(co-ordinates) of the three respective clusters.
    :param eigen_vector_1: The first eigen vector
    :param eigen_vector_2: The second eigen vector
    :return: Re-projected data for every attributes on each of the clusters .
    """

    re_projection_1 = []
    re_projection_2 = []

    for x in centroids:
        re_projection_1 = x[0]*eigen_vector_1
        re_projection_2 = x[1]*eigen_vector_2
        re_projection_sum = re_projection_1 + re_projection_2
        Re_centered_data = re_projection_sum + data.mean()
        print('cluster centers for centroid',x,'\n',Re_centered_data)


if __name__ == '__main__':
    main()