# KMeans-clustering-on-the-Employee_income
Apply KMeans clustering on the Employee_income.xlsx dataset and identify exact clusters using Elbow method to model the algorithm

Unsupervised Learning: KMeans Clustering on Employee Income Data
This notebook demonstrates the application of KMeans clustering on the Employee Income dataset to identify distinct employee groups based on their age and income.

Steps Involved:
Data Loading and Exploration: The Employee_income.xlsx dataset is loaded and explored to understand the distribution of Age and Income.
Data Preprocessing: Min-Max Scaling is applied to the 'Age' and 'Income' features to normalize their ranges and prevent dominance of features with larger values.
KMeans Clustering: The KMeans algorithm is applied to the scaled data.
Determining the Optimal Number of Clusters (Elbow Method): The Elbow method is used to determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters.
Evaluating Clustering Performance (Silhouette Score and Silhouette Visualizer): The Silhouette score and Silhouette Visualizer are used to evaluate the quality of the clusters.
Model Persistence: The trained KMeans model and the scaler object are saved using pickle for future deployment.
Files:
Employee_income.xlsx: The dataset used for clustering.
kmeans.pkl: The saved KMeans model.
scaler.pkl: The saved MinMaxScaler object.
Requirements:
pandas
matplotlib
seaborn
sklearn
yellowbrick
