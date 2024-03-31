# -*- coding: utf-8 -*-
"""
Utils

@author: elizac
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


def preprocess_datasets(t_col = 0.6, t_row = 1):
    '''
    Function that:
        - Loads all datasets
        - Uses the unnamed column as index
        - Drops the columns that are not important/repetitive, and those are:
            In DS1:
                - The "unit" column which is "C" for all.
                - The "id_field_values" column which is a unique value per row (does not provide any relevant info).
                - The "id_ftp" field which is confirmed by the data owner as of no use.
            In DS2:
                - The "field" column which is "powerClass" for all.
                - The "id_trx_status" column which is a unique value per row (does not provide any relevant info).
        - Renames the "value" column which is a common name in DS1 and DS2, to "value_temp" in DS1 and splits it into two columns in DS2 "power_watt" and "power_dbm".
        - Rearranges the datasets into 1 dataset, by rearranging the following rows for each id_audit into columns:
            - The "field" column from DS1.
            - The "branch_header" column from DS2. Note that we have 8 branches (A -> H). However, branches have two values for their power (Watt and dBm), so they will result in 2*8 = 16 columns.
        - Merge all datasets to form 1.
        - Clean up the dataset by:
            - Keeping only temperature values between -40°C and 125°C.
            - Removing columns that have too many missing values. This is controlled by the parameter "t" which defines the tolerated percentage of non-missing values for a column to be considered.
        - Returns 1 dataset with only 1 row for each id_audit.

    Parameters
    ----------
    t_col : Parameter to keep only the columns with at least t_col% non-NA values.
    t_row: Parameter to keep only the rows with at least t_row% non-NA values.
    
    Returns
    -------
    ds : Pandas DataFrame. Shape: (?, 93)
        Final dataset.

    '''
    # 1. Load the datasets:
    ds1 = pd.read_csv("Datasets/dataset-1.csv", delimiter=";", index_col=[0]).drop(columns=["Unit"])
    ds2 = pd.read_csv("Datasets/dataset-2.csv", delimiter=";", index_col=[0]).drop(columns=["ConfigurationID"])
    ds3 = pd.read_csv("Datasets/dataset-3.csv", delimiter=";", index_col=[0])
    
    # 2. Preprocess Dateset 1:
    ds1.rename(columns = {'MeasurementID':'SensorID'}, inplace = True) # Give a more meaningful name to the column (Optional)
    ds1_ = ds1.pivot(index='ElementID', columns=['SensorID'], values=['Value']) # Pivot the sensor values as columns
    ds1_ = ds1_.droplevel(0, axis=1) # Remove the 2nd level which the pivot function adds
    
    # 3. Preprocess Dataset 2:
    ds2[['Value_Primary','Value_Secondary']] = ds2['Value'].str.split(' \[',expand=True) # split the "Value" field into 2 columns "Value_Primary" and "Value_Secondary"
    ds2_1 = ds2.pivot(index='ElementID', columns=['Configuration'], values=['Value_Primary']).rename(columns={"Data Transmission Protocol": "DTP", "Power Source": "PS", "Control Interface": "CI"}) # Pivot the "Configuration" as columns and "Value_Primary" as values. Then rename the columns to short names.
    ds2_2 = ds2.pivot(index='ElementID', columns=['Configuration'], values=['Value_Secondary']).rename(columns={"Data Transmission Protocol": "DTP", "Power Source": "PS", "Control Interface": "CI"}) # Pivot the "Configuration" as columns and "Value_Secondary" as values. Then rename the columns to short names.
    ds2_1 = ds2_1.droplevel(0, axis=1) # Remove the 2nd level which the pivot function adds.
    ds2_2 = ds2_2.droplevel(0, axis=1) # Remove the 2nd level which the pivot function adds.
    ds2_ = ds2_1.join(ds2_2, on=['ElementID']) # Merge both ds2_1 and ds2_2 together to form ds2_
    
    # 4. Merge all 3 datasets together to form the final dataset:
    ds = ds1_.join(ds2_, on=['ElementID'])
    ds = ds.join(ds3, on=['ElementID'])
    
    # 5. Clean up:
    #   5.1. Keep only the humidity measurements that are within a desired/valid range: 0.4 to 0.85.
    for i in range(74):
        ds.iloc[:,i] = np.where((-0.4 <= ds.iloc[:,i]) & (ds.iloc[:,i] <= 0.85), ds.iloc[:,i], np.nan)
    
    #   5.2. Remove the columns that don't have enough values: >=t_col% of rows have no value, e.g. >=50% of rows must have values.
    ds = ds.dropna(thresh=t_col*ds.shape[0], axis=1)
    
    #   5.3 Remove the rows that don't have enough values: >=t_row% of features have no value.
    ds = ds.dropna(thresh=t_row*ds.shape[1], axis=0)
    
    return ds


def exploreDataset(X, draw_scatterplots_featurepairs = False, draw_corr_heatmap = True):
    # 1. Draw scatter plots for all each feature pairs.
    if (draw_scatterplots_featurepairs):
        for i in range(X.shape[1]-1):
            for j in range(X.shape[1]-1):
                if (i!=j):
                    plt.figure(figsize=(12,9))
                    plt.scatter(X.iloc[:,i], X.iloc[:,j])
                    plt.xlabel(X.columns[i])
                    plt.ylabel(X.columns[j])
                    plt.title('Visualization of '+X.columns[i]+' vs. '+X.columns[j])
    # 2. Correlation matrix heatmap:
    if (draw_corr_heatmap):
        plt.figure(figsize=(15,12))
        X_corr = X.corr()
        sns.heatmap(X_corr)


def model_kmeans(X, C = 3, draw_scatterplots = False):
    '''
    Fits the dataset to a K-Means model and returns clusters centroids and the cluster to which each sample belongs.
    Parameters
    ----------
    X : Input dataset.
    C : Number of clusters, optional.
    draw_scatterplots : Boolean, optional. False by default.

    Returns
    -------
    clusters_centers : Center of each cluster along the feature axis.
    y_kmeans : The cluster to which each sample (row) belongs, after fitting the data.

    '''
    np.random.seed(10)
    km = KMeans(n_clusters = C, n_init = 100, init='k-means++', random_state= 42)
    y_kmeans = km.fit_predict(X)
    clusters_centers = np.array(km.cluster_centers_)
    return y_kmeans, clusters_centers, km

def elbow(X, C):
    '''
    Elbow method.

    Parameters
    ----------
    X : TInput dataset.
    C : Number of clusters.

    Returns
    -------
    None.

    '''
    inertia = []
    for cluster in range(1,C):
        kmeans = KMeans(n_clusters = cluster, init='k-means++')
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    # Converting the results into a dataframe and plotting them
    df = pd.DataFrame({'Cluster':range(1,C), 'Inertia':inertia})
    plt.figure(figsize=(20,20))
    plt.plot(df['Cluster'], df['Inertia'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


def optimal_pca_components_count(X, p):
    '''
    Checks the optimal number of PCA components to keep p% of the data variance:
    Parameters
    ----------
    X : Dataset.
    p : Number between 0 and 100.
    '''
    pca = PCA(n_components=p)
    pca.fit(X)
    print("Optimal number of PCA components to keep "+str(p*100)+"% of data variance: "+str(pca.n_components_))


def cum_exp_variance_plot(X, pca):
    '''
    Draws the cumulative exponential variance for the PCA components.

    Parameters
    ----------
    X : Input dataset.
    pca : PCA result.

    Returns
    -------
    None.

    '''
    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)
    plt.figure(figsize=(7,7))
    plt.bar(range(1, X.shape[1]+1), exp_var, align='center', label='Individual explained variance')
    for index in range(1,X.shape[1]+1):
        plt.text(x=index , y =exp_var[index-1]+1 , s=f"{round(exp_var[index-1],2)}%")
    plt.step(range(1, X.shape[1]+1), cum_exp_var, where='mid', label='Cumulative explained variance', color='red')

    plt.ylabel('Explained variance percentage')
    plt.xlabel('Principal component index')
    plt.xticks(ticks=list(range(1, X.shape[1]+1)))
    plt.show()


def plot_clusters(X, X_cluster_0, X_cluster_1, X_cluster_2, clusters_centers):
    '''
    Assuming 3 clusters.

    Parameters
    ----------
    X : Inpur dataset.
    X_cluster_0 : Dataset samples belonging to cluster 0.
    X_cluster_1 : Dataset samples belonging to cluster 1.
    X_cluster_2 : Dataset samples belonging to cluster 2.
    clusters_centers : Clusters centers.

    Returns
    -------
    None.

    '''
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if (i!=j and i<j and i<X.shape[1]/2):
                plt.figure(figsize=(15,10))
                plt.scatter(x = X_cluster_0[:,i], y = X_cluster_0[:,j], s = 1, c = "royalblue", marker = "o", label="Cluster 0")
                plt.scatter(x = X_cluster_1[:,i], y = X_cluster_1[:,j], s = 1, c = "cornflowerblue", marker = "o", label="Cluster 1")
                plt.scatter(x = X_cluster_2[:,i], y = X_cluster_2[:,j], s = 1, c = "lightsteelblue", marker = "o", label="Cluster 2")
                plt.scatter(x = clusters_centers[:,i], y = clusters_centers[:,j], s = 50, c = 'red', marker = "x", label = 'Centroids')
                plt.title('Feature #'+str(i)+" vs Feature #"+str(j))
                plt.xlabel('Feature #'+str(i))
                plt.ylabel('Feature #'+str(j))
                plt.legend()
                plt.show()


def plot_clusters_3d(X_cluster_0, X_cluster_1, X_cluster_2, clusters_centers):
    '''
    3D plot of the 3 clusters.

    Parameters
    ----------
    X_cluster_0 : Dataset samples belonging to cluster 0.
    X_cluster_1 : Dataset samples belonging to cluster 1.
    X_cluster_2 : Dataset samples belonging to cluster 2.
    clusters_centers : Clusters centers.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(25,20))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X_cluster_0[:,0], X_cluster_0[:,1], X_cluster_0[:,2], color = "royalblue", label="Cluster 0")
    ax.scatter3D(X_cluster_1[:,0], X_cluster_1[:,1], X_cluster_1[:,2], color = "cornflowerblue", label="Cluster 1")
    ax.scatter3D(X_cluster_2[:,0], X_cluster_2[:,1], X_cluster_2[:,2], color = "lightsteelblue", label="Cluster 2")
    
    ax.scatter3D(clusters_centers[:,0], clusters_centers[:,1], clusters_centers[:,2], s = 100, alpha = 1, color = "red", marker = "x", label = "Centroids")
    ax.set_title("3D Clusters")
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_zlabel('Feature 2')
    plt.legend()
    plt.show()


def plot_clusters_3d_see_centroids(X_cluster_0, X_cluster_1, X_cluster_2, clusters_centers):
    '''
    3D plot of the 3 clusters, but with more focus on the centroids as they may not be visible if too many samples happen to cover the centers.

    Parameters
    ----------
    X_cluster_0 : Dataset samples belonging to cluster 0.
    X_cluster_1 : Dataset samples belonging to cluster 1.
    X_cluster_2 : Dataset samples belonging to cluster 2.
    clusters_centers : Clusters centers.

    Returns
    -------
    None.
    '''
    plt.figure(figsize=(25,20))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X_cluster_0[:,0], X_cluster_0[:,1], X_cluster_0[:,2], color = "royalblue", s = 10, alpha = 0.05, label="Cluster 0")
    ax.scatter3D(X_cluster_1[:,0], X_cluster_1[:,1], X_cluster_1[:,2], color = "cornflowerblue", s = 10, alpha = 0.05, label="Cluster 1")
    ax.scatter3D(X_cluster_2[:,0], X_cluster_2[:,1], X_cluster_2[:,2], color = "lightsteelblue", s = 10, alpha = 0.05, label="Cluster 2")
    
    ax.scatter3D(clusters_centers[:,0], clusters_centers[:,1], clusters_centers[:,2], s = 100, alpha = 1, color = "red", marker = "x", label = "Centroids")
    ax.set_title("3D Clusters")
    plt.legend()
    plt.show()


def plot_clusters_with_outliers(X, X_cluster_0, X_cluster_1, X_cluster_2, clusters_centers, outliers_cluster_0, outliers_cluster_1, outliers_cluster_2):
    '''
    Plot clusters and encricle the outliers.

    Parameters
    ----------
    X: Input dataset.
    X_cluster_0 : Dataset samples belonging to cluster 0.
    X_cluster_1 : Dataset samples belonging to cluster 1.
    X_cluster_2 : Dataset samples belonging to cluster 2.
    clusters_centers : Clusters centers.
    outliers_cluster_0: Detected outliers in cluster 0.
    outliers_cluster_1 Detected outliers in cluster 1.
    outliers_cluster_2: Detected outliers in cluster 2.

    Returns
    -------
    None.
    '''
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if (i!=j and i<=X.shape[1]/2) and i<j:
                plt.figure(figsize=(15,10))
                plt.scatter(x = X_cluster_0[:,i], y = X_cluster_0[:,j], s = 1, c = "royalblue", marker = "o", label="Cluster 0")
                plt.scatter(x = X_cluster_1[:,i], y = X_cluster_1[:,j], s = 1, c = "cornflowerblue", marker = "o", label="Cluster 1")
                plt.scatter(x = X_cluster_2[:,i], y = X_cluster_2[:,j], s = 1, c = "lightsteelblue", marker = "o", label="Cluster 2")
                plt.scatter(x = outliers_cluster_0[:,i], y = outliers_cluster_0[:,j], marker="o", facecolor="None", edgecolor="salmon",s=100, label = "Cluster 0 Outliers")
                plt.scatter(x = outliers_cluster_1[:,i], y = outliers_cluster_1[:,j], marker="o", facecolor="None", edgecolor="lawngreen",s=100, label = "Cluster 1 Outliers")
                plt.scatter(x = outliers_cluster_2[:,i], y = outliers_cluster_2[:,j], marker="o", facecolor="None", edgecolor="orange",s=100, label = "Cluster 2 Outliers")
                plt.scatter(x = clusters_centers[:,i], y = clusters_centers[:,j], s = 100, c = 'red', marker = "x", label = 'Centroids')
                plt.title('Feature #'+str(i)+" vs Feature #"+str(j))
                plt.xlabel('Feature #'+str(i))
                plt.ylabel('Feature #'+str(j))
                plt.legend()
                plt.show()

def predict_one_sample(input_vector, km, clusters_centers, outlier_distance_0, outlier_distance_1, outlier_distance_2):
    '''
    Make a prediction.
    
    Parameters
    ----------
    input_vector : Single input vector of shape (1,N) where N is the number of features.
    km: K-Means model output.
    clusters_centers : Clusters centers.
    outliers_cluster_0: Detected outliers in cluster 0.
    outliers_cluster_1 Detected outliers in cluster 1.
    outliers_cluster_2: Detected outliers in cluster 2.

    Returns
    -------
    None.
    '''
    # 1. Use the predict function to assign the new sample to a cluster
    y_pred = km.predict(input_vector.reshape(1,clusters_centers.shape[1])).item()

    print("\nThis sample belongs to cluster #" + str(y_pred))
    
    # 2. Compute the distance of this sample to the centroid of the cluster, and compare it to the qth percentile
    if y_pred == 0:
        distance = cdist(clusters_centers[0].reshape(1,clusters_centers.shape[1]), input_vector.reshape(1,clusters_centers.shape[1]), 'euclidean').item()
        print("Distance after which the sample is considered an outlier: " + str(outlier_distance_0))
        print("Actual distance from the centroid of cluster #" + str(y_pred) + ": " + str(distance))
        if (distance > outlier_distance_0):
            print("\nPrediction: Outlier\n")
        else:
            print("\nPrediction: Not an outlier\n")
    elif y_pred == 1:
        distance = cdist(clusters_centers[1].reshape(1,clusters_centers.shape[1]), input_vector.reshape(1,clusters_centers.shape[1]), 'euclidean').item()
        print("Distance after which the sample is considered an outlier: " + str(outlier_distance_1))
        print("Actual distance from the centroid of cluster #" + str(y_pred) + ": " + str(distance))
        if (distance > outlier_distance_1):
            print("\nPrediction: Outlier\n")
        else:
            print("\nPrediction: Not an outlier\n")
    elif y_pred == 2:
        distance = cdist(clusters_centers[2].reshape(1,clusters_centers.shape[1]), input_vector.reshape(1,clusters_centers.shape[1]), 'euclidean').item()
        print("Distance after which the sample is considered an outlier: " + str(outlier_distance_2))
        print("Actual distance from the centroid of cluster #" + str(y_pred) + ": " + str(distance))
        if (distance > outlier_distance_2):
            print("\nPrediction: Outlier\n")
        else:
            print("\nPrediction: Not an outlier\n")
    else:
        print("Error making a prediction")


def predict_batch(X, outlier_detection_method, km, clusters_centers, distances_cluster_0, distances_cluster_1, distances_cluster_2, X_cluster_0, X_cluster_1, X_cluster_2, q_audits_percentile, q_audits_std):
    '''
    Predict a batch of samples.

    Parameters
    ----------
    X : Input dataset.
    outlier_detection_method : String. Values: 'Percentile' or 'StandardDeviation'.
    km : K-Means model output.
    clusters_centers : Clusters centers.
    distances_cluster_0 : Distance of each sample of cluster 0 from the center.
    distances_cluster_1 : Distance of each sample of cluster 1 from the center.
    distances_cluster_2 : Distance of each sample of cluster 2 from the center.
    X_cluster_0 : Dataset samples belonging to cluster 0.
    X_cluster_1 : Dataset samples belonging to cluster 1.
    X_cluster_2 : Dataset samples belonging to cluster 2.
    q_audits_percentile : Defines the percentile upon which data points that have distances higher than q_audits_percentile% of the rest are considered outliers.
    q_audits_std : Defines the standard deviation upon which data points that have distances further away from the center than q_audits_std% are considered outliers.

    Returns
    -------
    prediction : Array of pedictions.
    number_of_outliers : Number of outliers detected.
    outlier_percentage : Percentage of outliers found in the input data.

    '''
    y_pred = km.predict(X)
    
    if outlier_detection_method == 'Percentile':
        distances = np.zeros([X.shape[0], 1])
        distances[y_pred == 0] = cdist(clusters_centers[0].reshape(1,clusters_centers.shape[1]), X[y_pred == 0, :], 'euclidean').reshape(X_cluster_0.shape[0], 1)
        distances[y_pred == 1] = cdist(clusters_centers[1].reshape(1,clusters_centers.shape[1]), X[y_pred == 1, :], 'euclidean').reshape(X_cluster_1.shape[0], 1)
        distances[y_pred == 2] = cdist(clusters_centers[2].reshape(1,clusters_centers.shape[1]), X[y_pred == 2, :], 'euclidean').reshape(X_cluster_2.shape[0], 1)
    
        prediction = np.zeros([X.shape[0], 1])
        prediction[y_pred == 0] = np.where(distances[y_pred == 0] > np.percentile(distances_cluster_0, q_audits_percentile), 1, 0) # 1: outlier, 0: not an outlier
        prediction[y_pred == 1] = np.where(distances[y_pred == 1] > np.percentile(distances_cluster_1, q_audits_percentile), 1, 0)
        prediction[y_pred == 2] = np.where(distances[y_pred == 2] > np.percentile(distances_cluster_2, q_audits_percentile), 1, 0)
    
        number_of_outliers = prediction[np.where(prediction == 1)].sum()
        outlier_percentage = number_of_outliers/X.shape[0]
    else: # if outlier_detection_method == 'StandardDeviation'
        prediction = np.zeros([X.shape[0], 1])
        prediction[y_pred == 0] = np.where(distances_cluster_0 > q_audits_std * (np.median(distances_cluster_0) + np.std(distances_cluster_0)), 1, 0) # 1: outlier, 0: not an outlier
        prediction[y_pred == 1] = np.where(distances_cluster_1 > q_audits_std * (np.median(distances_cluster_1) + np.std(distances_cluster_1)), 1, 0) # 1: outlier, 0: not an outlier
        prediction[y_pred == 2] = np.where(distances_cluster_2 > q_audits_std * (np.median(distances_cluster_2) + np.std(distances_cluster_2)), 1, 0) # 1: outlier, 0: not an outlier
    
        number_of_outliers = prediction[np.where(prediction == 1)].sum()
        outlier_percentage = number_of_outliers/X.shape[0]
    return prediction, number_of_outliers, outlier_percentage


def sensor_detections_visualization(measurements_predictions, measurements_names):
    '''
    Visualize the number of outliers detected for each Measurement. A measurement can be a sensor, KPI, etc.

    Parameters
    ----------
    measurements_predictions : Predictions for each measurement.
    measurements_names : Measurements names.

    Returns
    -------
    None.

    '''
    # Remove "Sensor_" and keep just the sensor name, for better plot visibility.
    if measurements_names[0].startswith("Sensor_"):
        for i in range(len(measurements_names)):
            measurements_names[i] = measurements_names[i].split("_")[1]
    measurements_detections = np.sum(measurements_predictions[1:], axis=0)[1:]
    plt.figure(figsize=(25,10))
    plt.bar(measurements_names, measurements_detections, align='center', label='')
    for i in range(len(measurements_names)):
        plt.text(x=i-0.4 , y =measurements_detections[i]+1.4 , s=f"{measurements_detections[i]}")

    plt.ylabel('Measurements Detections')
    plt.xlabel('Measurements')
    plt.show()
