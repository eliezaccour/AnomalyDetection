# -*- coding: utf-8 -*-
"""
Anomaly Detection
- K-Means
- Autoencoder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import tensorflow as tf
from tensorflow.keras import layers
import utils as utils

# 1. Hyperparameters Definition (1):
t_col = 0.6 # Parameter to keep only the columns with at least t_col% non-NA values.
t_row = 1 # Parameter to keep only the rows with at least t_row% non-NA values.
N_ = 3 # The new y-axis shape of the dataset.
K = 3 # Number of clusters.
q_elements_percentile = 98 # Data points that have distances higher than q% of the rest are considered outliers
q_elements_std = 1.5
mu_sensors_percentile = 98 # Sensors that have values higher in magnitude (positive or negative) than mu_sensors_percentile% of the rest are considered anomalous
mu_sensors_std = 1.5

# 2. Preprocess dataset.
ds = utils.preprocess_datasets(t_col, t_row)

# 3. Hyperparameters Definition (2):
sensor_names = list(ds.iloc[:,:48].columns)
# id_elements = ds.index
id_elements = np.array([ds.index]).T


def get_anomalous_elements(ds, method='KMeans', reduce_dim=False, outlier_detection_method='Percentile'):
    '''
    Identifies anomalous elements using k-means clustering.

    Parameters
    ----------
    ds : Pandas DataFrame
        Input dataset.
    reduce_dim : Boolean, optional
        Instructs whether to do dimensionality reduction (PCA) or not. The default is False.
    outlier_detection_method : String, optional
        Possible values: 'Percentile', 'StandardDeviation'
        Sets the method to detect outliers. The default is 'Percentile'.
          - If it is set to 'Percentile', it assumes an outlier any sample that has a distance from the center of its corresponding cluster higher than q% of the other samples (i.e. elements).
          - If it is set to 'StandardDeviation', it assumes an outlier any sample that has a distance from the center q_elements_std times higher than the median distance + the standard deviation.

    Returns
    -------
    None.

    '''
    print("Started scanning for anomalous elements.")
    if reduce_dim == True:
        print("Started dimensionality reduction")
        pca = PCA(n_components=N_)
        ds = pca.fit_transform(ds)
    else:
        ds = ds.to_numpy() # PCA transforms the Pandas DF to a Numpy array. But if it's not applied, let's transform it explicitly.
    
    print("Started K-Means clustering")
    # K-Means:
    y_kmeans, clusters_centers, km = utils.model_kmeans(ds, K)
    # Separate the data by cluster.
    ds_cluster_0 = ds[y_kmeans == 0]
    ds_cluster_1 = ds[y_kmeans == 1]
    ds_cluster_2 = ds[y_kmeans == 2]
    # Compute the distance between the data points and the centroid of the cluster to which they belong. Euclidean distance is used.
    distances_cluster_0 = cdist(clusters_centers[0].reshape(1,clusters_centers.shape[1]), ds_cluster_0, 'euclidean').T
    distances_cluster_1 = cdist(clusters_centers[1].reshape(1,clusters_centers.shape[1]), ds_cluster_1, 'euclidean').T
    distances_cluster_2 = cdist(clusters_centers[2].reshape(1,clusters_centers.shape[1]), ds_cluster_2, 'euclidean').T
    if outlier_detection_method == 'Percentile':
        # Compute the distances based on which to select outliers (these are the distances beyond which an outlier is detected):
        outlier_distance_0 = np.percentile(distances_cluster_0, q_elements_percentile)
        outlier_distance_1 = np.percentile(distances_cluster_1, q_elements_percentile)
        outlier_distance_2 = np.percentile(distances_cluster_2, q_elements_percentile)
        # Compute the outliers in each cluster, which are defined as the data points that have a distance from the centroid that is higher than q% of the other data points.
        outliers_cluster_0 = ds_cluster_0[np.where(distances_cluster_0 > outlier_distance_0), :][0,:,:]
        outliers_cluster_1 = ds_cluster_1[np.where(distances_cluster_1 > outlier_distance_1), :][0,:,:]
        outliers_cluster_2 = ds_cluster_2[np.where(distances_cluster_2 > outlier_distance_2), :][0,:,:]
    else: # i.e. if outlier_detection_method == 'StandardDeviation'
        outliers_cluster_0 = ds_cluster_0[np.where(distances_cluster_0 > q_elements_std * (np.median(distances_cluster_0) + np.std(distances_cluster_0))), :][0,:,:]
        outliers_cluster_1 = ds_cluster_1[np.where(distances_cluster_1 > q_elements_std * (np.median(distances_cluster_1) + np.std(distances_cluster_1))), :][0,:,:]
        outliers_cluster_2 = ds_cluster_2[np.where(distances_cluster_2 > q_elements_std * (np.median(distances_cluster_2) + np.std(distances_cluster_2))), :][0,:,:]
    if reduce_dim==True: # Only do this if dimensionality is reduced, otherwise you'll get too many plots.
        utils.plot_clusters(ds, ds_cluster_0, ds_cluster_1, ds_cluster_2, clusters_centers) # Visualize the clusters and centroids (2D plots of each feature pair)
        utils.plot_clusters_with_outliers(ds, ds_cluster_0, ds_cluster_1, ds_cluster_2, clusters_centers, outliers_cluster_0, outliers_cluster_1, outliers_cluster_2) # Visualize the clusters with the detected outliers encircled
        if N_ == 3: # Only do this if the dataset was reduced to a 3-feature dataset which can be plotted in 3D
            utils.plot_clusters_3d(ds, ds_cluster_0, ds_cluster_1, ds_cluster_2, clusters_centers) # Visualize the clusters in 3D
            utils.plot_clusters_3d_see_centroids(ds, ds_cluster_0, ds_cluster_1, ds_cluster_2, clusters_centers) # Visualize the clusters in 3D with centroids
    print("K-Means clustering completed successfully.")
    print("Identifying outliers in the dataset")
    elements_prediction, number_of_outliers, outlier_percentage = utils.predict_batch(ds, outlier_detection_method, km, clusters_centers, distances_cluster_0, distances_cluster_1, distances_cluster_2, ds_cluster_0, ds_cluster_1, ds_cluster_2, q_elements_percentile, q_elements_std)
    print("K-Means found " + str(number_of_outliers) + " of the samples to be anomalous elements, out of the total of " + str(ds.shape[0]) + " elements.")
    print("Percentage of outliers in the data: " + str(round(outlier_percentage*100,2)) + "%")
        
    
    elements_prediction = pd.DataFrame(elements_prediction,
                                      index=id_elements[:,0],
                                      columns=['Prediction'])
    return elements_prediction


def get_anomalous_sensors(ds, elements_prediction, outlier_detection_method='Percentile'):
    '''
    Identifies anomalous sensors using an autoencoder model.

    Parameters
    ----------
    ds : Pandas DataFrame
        Input dataset. It's an array of size (M,N).
    elements_prediction : Pandas DataFrame
        The result of the element anomaly detection. It's an array of size (M, 1).
    outlier_detection_method : String, optional
        Possible values: 'Percentile', 'StandardDeviation'
        Sets the method to detect outliers. The default is 'Percentile'.
          - If it is set to 'Percentile', it assumes an outlier any sensor that has a loss higher than mu_sensors_percentile% of the other sensors within the same element.
          - If it is set to 'StandardDeviation', it assumes an outlier any sensor that has a loss mu_sensors_std times higher than the median error + the standard deviation of all sensor losses within the same element.

    Returns
    -------
    sensors_prediction : TYPE
        DESCRIPTION.

    '''
    print("Started scanning for anomalous sensors.")
    X_train = ds[elements_prediction.iloc[:,0] == 0].iloc[:,:48] # Train the model only on the non-anomalous samples
    X_validation = ds.iloc[0:1000].iloc[:,:48] # Validation set using the first 1000 samples (anomalous + non-anomalous)
    X_test = ds[elements_prediction.iloc[:,0] == 1].iloc[:,:48] # Test set is all the anomalous elements for which we want to identify faulty snesors

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validation = sc.fit_transform(X_validation)
    X_test = sc.transform(X_test)

    # Input layer
    input = tf.keras.layers.Input(shape=(48,))
    # Encoder layers
    encoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(8, activation='relu'),
      layers.Dense(4, activation='relu')])(input)
    # Decoder layers
    decoder = tf.keras.Sequential([
          layers.Dense(8, activation="relu"),
          layers.Dense(16, activation="relu"),
          layers.Dense(32, activation="relu"),
          layers.Dense(48, activation="sigmoid")])(encoder)
    # Create the autoencoder
    autoencoder = tf.keras.Model(inputs=input, outputs=decoder)
    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mae')
    print("Started autoencoder training.")
    # Fit the autoencoder
    autoencoder.fit(X_train, X_train,
                    epochs=2000,
                    batch_size=64,
                    validation_data=(X_validation, X_validation),
                    shuffle=True)
    print("Training completed.")
    print("Scanning the input dataset for anomalies.")
    # for predict anomalies/outliers in the training dataset
    sensors_prediction = autoencoder.predict(X_test)
    sensors_prediction = sc.inverse_transform(sensors_prediction)
    X_test = sc.inverse_transform(X_test)
    # return sensors_prediction
    sensors_prediction = pd.DataFrame(sensors_prediction, index=id_elements[elements_prediction==1], columns=sensor_names)
    print("Identifying faulty sensors based on " + outlier_detection_method)
    prediction_sensors_error = X_test - sensors_prediction
    # Identify faulty sensors based on percentile or standard deviation:
    sensors_prediction = pd.DataFrame(0,
                                  index=id_elements[elements_prediction==1],
                                  columns=sensor_names)
    if outlier_detection_method == 'Percentile':
        for id_element in id_elements[elements_prediction==1]:
            for sensor_name in sensor_names:
                sensors_prediction.loc[id_element, sensor_name] = np.where( abs(prediction_sensors_error.loc[id_element, sensor_name]) > np.percentile(abs(prediction_sensors_error.loc[id_element, :]), mu_sensors_percentile), 1, 0)
    else: # if outlier_detection_method == 'StandardDeviation'
        for id_element in id_elements[elements_prediction==1]:
            for sensor_name in sensor_names:
                sensors_prediction.loc[id_element, sensor_name] = np.where( abs(prediction_sensors_error.loc[id_element, sensor_name]) > mu_sensors_std * (abs(np.median(prediction_sensors_error.loc[id_element, :])) + abs(np.std(prediction_sensors_error.loc[id_element, :]))), 1, 0)
    number_of_outliers = sensors_prediction.sum().sum()
    avg_number_of_outliers = round(number_of_outliers/X_test.shape[0],2)
    print("The autoencoder model found " + str(number_of_outliers) + " sensors to be anomalous, with an average of " + str(avg_number_of_outliers) + " faulty sensors per element.")
    print("Percentage of faulty sensors per element: " + str(round((avg_number_of_outliers/sensors_prediction.shape[1])*100,2)) + "%")
    return sensors_prediction



elements_prediction = get_anomalous_elements(ds, method='KMeans', reduce_dim=True, outlier_detection_method='StandardDeviation')
sensors_prediction = get_anomalous_sensors(ds, elements_prediction, outlier_detection_method='Percentile')
