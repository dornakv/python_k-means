import numpy as np

def GenerateRandomLeftStochastic(K, amount_of_data):
	gama = np.random.rand(K, amount_of_data)
	gama_sum = np.sum(gama,0) #for each cluster
	return np.divide(gama, gama_sum)

def Kmeans_int(data, K, eps, max_iters, norm=None):
	#data[amount_of_data, dimensions]
	amount_of_data = data[:,0].size
	dimensions = data[0,:].size

	gama = GenerateRandomLeftStochastic(K, amount_of_data) #gama[k, amount_of_data]
	centroids = np.zeros([K, dimensions])
	L = float("inf")
	for i in range(max_iters):
		gama_sum = np.sum(gama,1) #for each datapoint
		for k in range(K):
			if gama_sum[k] != 0:
				centroids[k,:] = np.divide(np.sum(np.multiply(data.transpose(), gama[k,:]),1), gama_sum[k])
			gama[k, :] = np.linalg.norm(data - centroids[k,:],axis=1, ord=norm)
		idx = np.argmin(gama,axis=0)
		gama = np.zeros([K, amount_of_data])
		for j in range(idx.size):
			gama[idx[j],j] = 1 #TODO must be better way than for cycle

		L_old = L
		L = np.linalg.norm(data - np.dot(gama.transpose(), centroids), ord=norm)
		if (L_old - L) < eps:
			break
	return centroids, gama, L, i

def Kmeans(data, K, eps, max_iters, restarts, norm=None):
	amount_of_data = data[:,0].size
	dimensions = data[0,:].size
	centroids_best = np.zeros([K, dimensions])
	gama_best = np.random.rand(K, amount_of_data)
	L_best = float("inf")
	iters_total = 0

	for i in range(restarts):
		centroids, gama, L, i = Kmeans_int(data, K, eps, max_iters, norm)
		if L < L_best:
			L_best = L
			centroids_best = centroids
			gama_best = gama

		iters_total += i

	return centroids_best, gama_best, iters_total
