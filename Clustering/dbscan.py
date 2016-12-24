
from mpl_toolkits.mplot3d import axes3d
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import csv
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing.data import StandardScaler
from sklearn.neighbors import NearestNeighbors


#DBSCAN functions
def DBSCAN_SIM(e,mpt, values):
    #Run DBscan on the data set
    dbsc = DBSCAN(eps = e, min_samples = mpt).fit(values)
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True
    
    # print core_samples
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print n_clusters_
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')    
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
        ax2.set_title('Estimated number of clusters: %d for minPts = %d, Radius = %.2f' %( n_clusters_ , mpt , e),loc="left")
    #     ax1.set_title('Scatter Plot')
        ax2.set_xlabel('x axis')
        ax2.set_ylabel('y axis')
        ax2.set_zlabel('z axis')
        #Plotting core samples
        xy = values[class_member_mask & core_samples]
        ax2.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        #PLotting outliers
        xy = values[class_member_mask & ~core_samples]
        ax2.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)    
    plt.show()

# KNN function to perform the elbow method for 
def KNN(k,values):
    X = values
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    y = sorted(distances[:,2])
    plt.plot(range(0,len(X)),sorted(distances[:,k-1]))
    plt.xlabel("Data Points")
    plt.ylabel("Radius")
    plt.xlim(0,len(X))
#     plt.ylim(0,2)
    plt.title("Neighborhood Radius plot (Elbow Method DBScan minPts = " + str(k) + " )")
    plt.show()

def printVals(values):
    x = values[:,0]
    y = values[:,1]
    z = values[:,2]
    fig = plt.figure()
    
    for c,m in [('r','o')]:
        
        ax1 = fig.add_subplot(111, projection='3d')    
        ax1.scatter(x,y,z,c=c)
        ax1.set_title("Scatter Plot")
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
    plt.show()

#Read the data set
my_list = []
f = open("PATHAK%20NEETISH.csv",'rb')
reader = csv.reader(f)
for row in reader:
#     print row
    my_list.append(row)

#Store the read values in an numpy array
values = np.array(my_list).astype(np.float)

dbScan_array = [[3,4.5],[4,4.8],[5,6.2],[6,7.1],[7,7.9],[8,8.5]]

for i in range(0,len(dbScan_array)):
    minPts = dbScan_array[i][0]
    eps = dbScan_array[i][1]
    
    #Runn KNN to find the neighborhood Radius
    KNN(minPts,values)
    
    #Run DBSCAN by identifying right neighbor hood radius values
    DBSCAN_SIM(eps, minPts, values)
    
