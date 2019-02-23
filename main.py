import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

if __name__=="__main__":
	
	#read dataset
	df=pd.read_csv(sys.argv[1], low_memory=False)
	df = df.apply(pd.to_numeric, errors='coerce')

	#seperate features and labels from csv
	label=df.columns[len(df.columns)-1]
	df=df.drop(df.columns[len(df.columns)-1], axis=1)

	df=df.fillna(0)

	X=np.array(df)

	# normalized_X=normalize(df, norm='l2')

	# print(normalized_X)

	kmeans = KMeans(n_clusters=2)
	kmeans.fit(X)

	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_

	print(centroids)
	print(str(labels))

	np.savetxt("kmeans_labels.csv", np.array([label,labels]), delimiter=",")