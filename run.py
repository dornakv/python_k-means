import numpy as np
import kmeans
import matplotlib.pyplot as plt

#Generate random data around centroids_initial
dimensions = 2
amount_of_clusters = 4
amount_of_data_in_cluster = 50

centroids_initial = np.empty([amount_of_clusters, dimensions]);
points = np.empty([amount_of_clusters*amount_of_data_in_cluster, dimensions]);

for i in range(amount_of_clusters):
	centroids_initial[i,:] = np.random.uniform(low = -10, high = 10 , size=(dimensions,))

	points[i*amount_of_data_in_cluster:(i+1)*amount_of_data_in_cluster,:] = np.random.uniform(low = -1, high = 1 ,size=(amount_of_data_in_cluster, dimensions)) + centroids_initial[i,:]

centroids, gama, iters = kmeans.Kmeans(points,amount_of_clusters,10,1000,10)





plt.scatter(points[:,0], points[:,1], s=1)
plt.scatter(centroids_initial[:,0], centroids_initial[:,1], s=10)
plt.scatter(centroids[:,0], centroids[:,1], s=30, c="red")
plt.show();
