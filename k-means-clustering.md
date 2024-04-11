**K-means Clustering**

K-means clustering is a popular unsupervised machine learning algorithm that partitions a set of data points into a specific number (k) of clusters. The goal is to group data points that are similar to each other based on a chosen distance metric (usually Euclidean distance) and minimize the within-cluster variance. Below are the steps involved in the K-means algorithm:
![image](https://github.com/nehakardam/ml-datascience-coding-/assets/70997776/51207ba6-5909-4425-aad7-c1a3a23f8eb4)

1. **Data Preparation:**
   - Ensure your data is suitable for K-means clustering. It typically works well with numerical data representing different features or characteristics of the data points.
   - Handle missing values or outliers if necessary, as they can affect the clustering process.

2. **Specify the Number of Clusters (k):**
   - This is a crucial step, and there's no one-size-fits-all solution. You might need to experiment with different values of k to find the optimal clustering that best captures the inherent structure in your data. Techniques like the elbow method or silhouette analysis can aid in this decision-making process.

3. **Initialize Centroids:**
   - Centroids are the representatives of each cluster. K-means starts by randomly selecting k data points as the initial centroids. These centroids will be iteratively refined throughout the algorithm.

4. **Assignment Step:**
   - For each data point, calculate the distance to all the current centroids.
   - Assign each data point to the cluster that has the closest centroid (based on the distance metric).

5. **Centroid Update Step:**
   - After all data points have been assigned to clusters, recompute the centroid for each cluster. The new centroid is the mean (average) of the data points belonging to that cluster.

6. **Repeat Steps 4 and 5:**
   - Continue iterating between the assignment step and the centroid update step until a stopping criterion is met. This criterion is typically based on whether the centroids have moved significantly between iterations. If the movement is minimal (indicating convergence), the algorithm stops.

7. **Result:**
   - Once the algorithm converges, each data point will be assigned to a specific cluster, and you will have a set of k clusters that represent groups of similar data points.

**Key Points to Consider:**

- K-means is sensitive to the initial placement of centroids. Different initializations can lead to slightly different clusterings. You can run the algorithm multiple times with different random initializations to check for consistency in the results.
- The algorithm assumes that the clusters are roughly spherical in shape. It may not perform well for data with elongated or irregular clusters.
- K-means does not provide a mechanism to automatically determine the optimal number of clusters (k). You'll need to use domain knowledge or evaluation metrics to guide this decision.

By following these steps and considering the key points, you can effectively apply K-means clustering to uncover hidden patterns and structures within your unlabeled data.

**K-means Clustering Algorithm Implementation**

import numpy as np

def kmeans(X, k, max_iteration=100):
    # Initialize the centroids
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    
    for _ in range(max_iteration):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        index = np.argmin(distances, axis=0)
        
        # Update the centroids
        new_centroids = np.array([X[index == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, index

# Example usage
# Generate some random data
np.random.seed(0)
data = np.random.randn(100, 2)
# Number of clusters
k = 3
# Run K-Means algorithm
centroids, labels = kmeans(data, k)
print("Final centroids:")
print(centroids)
print("Final cluster assignment:")
print(labels)
