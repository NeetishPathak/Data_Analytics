#Necessary imports
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans,vq
import numpy as np
import csv


#Read the data
my_list = []
f = open("PATHAK%20NEETISH.csv",'rb')
reader = csv.reader(f)
for row in reader:
#     print row
    my_list.append(row)
#Store the read values in an numpy array
values = np.array(my_list).astype(np.float)

x = values[:,0]
y = values[:,1]
z = values[:,2]

#Use Kmeans clustering with K=2
clf = KMeans(n_clusters=2)
clf.fit(values)

centroids = clf.cluster_centers_
labels =  clf.labels_

colors = ["g","r","c","magenta"]

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')    
ax1.set_title("Scatter Plot k= %d" %2, loc="left")
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')
 
# ax1.scatter(x,y,z,c=labels)
for i in range(len(values)):
    ax1.scatter(values[i][0],values[i][1],values[i][2],c=colors[labels[i]])
ax1.scatter(centroids[:,0],centroids[:,1],centroids[:,2],marker='x',s=150,linewidth=5,c='k')
plt.show()


#For elbow method test the K means for k=1-10
K = range(1,10)
KM = [kmeans(values,k) for k in K]
centroids = [cent for (cent,var) in KM]   # cluster centroids
D_k = [cdist(values, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
# avgWithinSS = [sum(d)/values.shape[0] for d in dist]
avgWithinSS = [sum(d)**2 for d in dist]
# print dist
# print values.shape[0]


##plot ###
kIdx = 2 #(best k identified as 3)
# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.show()

from collections import Counter
# k-means clustering with k =3
clf = KMeans(n_clusters=3)
clf.fit(values)

centroids = clf.cluster_centers_
labels =  clf.labels_
# print Counter(labels)
colors = ["g","r","c","magenta"]

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')    
ax1.set_title("Scatter Plot k=%d" %3, loc="left")
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')
 
# ax1.scatter(x,y,z,c=labels)
for i in range(len(values)):
    ax1.scatter(values[i][0],values[i][1],values[i][2],c=colors[labels[i]])
ax1.scatter(centroids[:,0],centroids[:,1],centroids[:,2],marker='x',s=150,linewidth=5,c='k')
plt.show()
