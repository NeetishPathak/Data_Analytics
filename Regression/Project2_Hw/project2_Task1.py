import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import pandas as pd
from pandas import DataFrame
from matplotlib import cm as cm

def correlation_matrix(df):    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 100)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Correlation Matrix')
    labels= [0] + list(df)
    ax1.set_xticklabels(labels,fontsize=10)
    ax1.set_yticklabels(labels,fontsize=10)
    # Add colorbar
    fig.colorbar(cax, ticks=[0,.25,.5,.75,.8,.85,.90,.95,1])
    plt.show()

#print Statistics function
def printStats(X, mean, variance):
    print X + " Statistics"
    print "Mean: " + str(mean)
    print "Variance: " + str(variance) + "\n"

def createFrange(X):
    minVal = min(X).astype(np.float)
    maxVal = max(X).astype(np.float)
    i = minVal
    range = [0.0]
    step = (maxVal - minVal)/10.0
    while(i <= maxVal + step):
        range.append(i)
        i += step
    return range
        
    
def showHistogram(X, strX):
    plt.xlabel(strX)
    plt.ylabel("Frequency")
    plt.title(strX + " Histogram")
    bins = createFrange(X)
    plt.xticks(bins)
    plt.hist(X,bins,histtype='bar',rwidth=0.8)
    plt.show()

#read the csv file
file_reader = open("PATHAK%20NEETISH.csv",'rb')
file_object = csv.reader(file_reader)
#skip the first line while reading values
header= file_object.next()

#capture the data from the file in the data[] list
data = []
for row in file_object:
    data.append(row)
#create an array
data = np.array(data).astype(float)

#Read first column/variable value
X1 = data[0::,0].astype(np.float)
mean_X1 = np.mean(X1)
var_X1 = np.var(X1)
printStats("X1",mean_X1, var_X1)
showHistogram(X1, "X1")

#Read X2 values
X2 = data[0::,1].astype(np.float)
mean_X2 = np.mean(X2)
var_X2 = np.var(X2)
printStats("X2",mean_X2, var_X2)
showHistogram(X2, "X2")

#Read X3 values
X3 = data[0::,2].astype(np.float)
mean_X3 = np.mean(X3)
var_X3 = np.var(X3)
printStats("X3",mean_X3, var_X3)
showHistogram(X3, "X3")

#Read X4 values
X4 = data[0::,3].astype(np.float)
mean_X4 = np.mean(X4)
var_X4 = np.var(X4)
printStats("X4",mean_X4, var_X4)
showHistogram(X4, "X4")

#read X5 values
X5 = data[0::,4].astype(np.float)
mean_X5 = np.mean(X5)
var_X5 = np.var(X5)
printStats("X5",mean_X5, var_X5)
showHistogram(X5, "X5")

#read X5 values
Y = data[0::,5].astype(np.float)
mean_Y = np.mean(Y)
var_Y = np.var(Y)
printStats("Y",mean_Y, var_Y)
showHistogram(Y, "Y")

#print np.corrcoef(data)
df = pd.read_csv('PATHAK%20NEETISH.csv')
print "Correlation Matrix"
print df.corr()
# print df.describe()
correlation_matrix(df)