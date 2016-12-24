import numpy as np
import math
from pandas import Series, DataFrame
import pandas as pd
from quandl.model.dataset import Dataset
import matplotlib.pyplot as plt
from numpy.core.numeric import NaN


mainDataSet = 1
rangeM = 500
if mainDataSet == 1:
    df = pd.read_csv("NewFData.csv")
    dataset = df['X'].tolist()
    trainingDataLim = len(dataset) - 500
    testDataLim = len(dataset) - trainingDataLim
    ds_train = dataset[16:trainingDataLim]
    ds_test = dataset[trainingDataLim:len(dataset)]

else:
    #test data for debugging
    dataset = [1,2,3,4,5,6,7,8,9,10,11]#[2,3,4,1,4,3,5,2,5,6,1,7,3,4,5,6,4,3,1,2,4,6]
    trainingDataLim = 9
    testDataLim = len(dataset) - trainingDataLim
    ds_train = dataset[1:trainingDataLim]
    ds_test = dataset[trainingDataLim:len(dataset)]
    
def printVal(x):
    if mainDataSet == 0:
        print x

#Check for stationarity of data
'''
For data to be stationary, the mean,variance should be constant over time
Also, the ocvariance, should not depence on time
'''
def test_stationarity(ts):
    #Find rolling statistics
    rolMean = ts.rolling(window=12, center=False).mean()
    rolStd = ts.rolling(window=12, center=False).std()    
    #plot rolling statistics
    orig = plt.plot(ts, color='blue', label='original')
    mean = plt.plot(rolMean, color='red', label='Rolling Mean')
    std = plt.plot(rolStd, color='k', label='rolling std')
    plt.legend(loc='best')
    plt.title('Comparison for stationarity of data')
    plt.xlabel('X')
    plt.ylabel('Time series values')
    plt.show(block='False')
'''
If the plot shows a constant mean and variance around the data, then it is stationary
'''    
test_stationarity(df['X'])
       
#Function to calculate the mean of a list
def mean(vals):
    sum = 0.0
    for x in range(0,len(vals)):
        sum += vals[x]
    return sum/len(vals)

#Function to calculate the mean of the trianing data
def calcRMSE(sma, ds_train):
    mse = 0.0
    count = 0;
    for i in range(0,len(sma)):
        if math.isnan(sma[i]) == False:
            diff = sma[i] - ds_train[i]
            mse += (diff**2)
            count += 1
    if count == 0:
        return np.NaN
    return (mse/count)**0.5


# *********************************Simple Moving Average****************************

#Function to calculate the smoothing moving average
def SMA(trainData, window):
    predictedData = []
    for x in range(0, len(trainData)):
        if x < window:
            predictedData.append(NaN)
        else:
            list = [trainData[x-1-i] for i in range(0,window)]
            predictedData.append(mean(list))
    return predictedData

def simpleMovingAverage(vals, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(vals, weights, 'valid')
    return smas

# sma = simpleMovingAverage(dataset, 4)
def calcSimpleMovingAverage(ds, window):
    sma1 = SMA(ds, window)
    return sma1

def mainForOneMovingAvg():
    m = input("Input the value of m : ")
    calcSimpleMovingAverage(ds_train, m)
    
    
def mainForRMSE():
    rmseList = []
    for m in range(1,rangeM):
        sma = calcSimpleMovingAverage(ds_train,m)
        printVal(sma)
        rmse = calcRMSE(sma,ds_train)
        printVal("rmse is " + str(rmse))
        rmseList.append(rmse)
    printVal(len(rmseList))
    plt.plot(range(1,rangeM),rmseList)
    plt.xlabel("Window Size")
    plt.ylabel("RMSE")
    plt.title("Simple Moving Average: RMSE vs m")
    plt.xlim(1,rangeM + 1)
    plt.ylim(0,max(rmseList) + 1)
    plt.show()
#     print rmseList
    x = sorted(rmseList)
    return rmseList.index(min(rmseList))+1,min(rmseList)


def plotOnBestM(ds,optM,datatype):
    sma = calcSimpleMovingAverage(ds,optM)
    printVal(sma)
    predicted = sma
    actual = ds
    orig = plt.plot(actual, color='blue', label='original')
    predict = plt.plot(predicted, color='red', label='predicted')
    plt.xlabel("x")
    plt.ylabel('data points')
    plt.title("Best m plot - Actual and Predicted " +  str(datatype))
    plt.legend(loc='best')
    plt.show()    
    plt.scatter(actual,predicted,color='k')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Scatter plot for Actual and Predicted " + str(datatype))
    plt.show()

def runSimpleMovingAvg():
    
    #Running simple moving average for m = 4
    smaEx = calcSimpleMovingAverage(ds_train, 4)
    rmseEx = calcRMSE(smaEx, ds_train)
    print "Simple moving Average RMSE for m = 4 : %.4f"%rmseEx
    print "Running simple moving average for m = 1-500..."
    #Running Simple moving average for m = 1-500
    minIndex, minRmse = mainForRMSE()
    print "Minimum RMSE is " + str(minRmse) + " at m =" + str(minIndex)
    #Plotting Training Data for best m
    plotOnBestM(ds_train, minIndex, "Training Data")
    #plotting test data for best m
    plotOnBestM(ds_test, minIndex, "Test Data")
    #calculate RMSE for test data for m = minIndex
    smaTestD = calcSimpleMovingAverage(ds_test, minIndex)
    rmseTestD = calcRMSE(smaTestD, ds_test)
    print "Simple moving Average RMSE(Test data) for m = %d : %.4f"%(minIndex,rmseTestD)



# ******************************Exponential Smoothing**************************

def exponentialSmoothing(ds,a):
    s = [0 for _ in range(0,len(ds))]
    for x in range(1,len(ds)):
        s[x] = a * ds[x-1] + (1-a)*s[x-1]        
#     print s
    return s

def exponentialSmoothingSingle(ds_train):
    a = input("Enter the value of a between 0-1")
    return exponentialSmoothing(ds_train, a)

def exponentialSmoothingRange(ds_train):
    a = 0.1
    rmseArray = []
    while a < 1.0:
        rmseArray.append(calcRMSE(exponentialSmoothing(ds_train, a), ds_train))
        a += 0.1
#     print str(rmseArray) + " for exponentialSmoothing"
    plt.plot(range(0,len(rmseArray)),rmseArray)
    plt.xlabel("a (x10e-1)")
    plt.ylabel("RMSE")
    plt.title("RMSE for exponential smoothing")
    plt.xlim(1,len(rmseArray) + 1)
    plt.ylim(0,max(rmseArray) + 1)
    plt.show()
    return (0.1 + rmseArray.index(min(rmseArray))/10.0), min(rmseArray)
            
def predictedexpData(ds, a, datatype):
    predictedData = exponentialSmoothing(ds, a)
    actual = ds
    orig = plt.plot(actual, color='blue', label='original')
    predict = plt.plot(predictedData, color='red', label='predicted')
    plt.xlabel("x")
    plt.ylabel('data points')
    plt.title("Best plot (Exponential) - Actual and Predicted " +  str(datatype))
    plt.legend(loc='best')
    plt.show()    
    plt.scatter(actual,predictedData,color='k')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.xlim(min(ds)-10,max(ds)+10)
    plt.ylim(min(predictedData)-10,max(predictedData)+10)
    plt.title("Scatter plot for Actual and Predicted " + str(datatype))
    plt.show()

def runExpSmoothing():
    minA, minexpRmse = exponentialSmoothingRange(ds_train)
    print "MinA (alpha) is : " + str(minA)
    print "MinexpRmse on training Data is : " + str(minexpRmse)
    #Plotting Training data vs predicted Data
    predictedexpData(ds_train, minA, "Training Data")
    #Plotting Test data vs predicted Data
    predictedexpData(ds_test, minA, "Test Data")
    #RMSE for test data
    expRmseTest = calcRMSE(exponentialSmoothing(ds_test, minA), ds_test)
    print "Exponential smoothing RMSE for test Data = %.4f"%expRmseTest
    
# **********************************AR() Model*************************************
#PACF plot:
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA

def runARModel():
    delta = 0.1
#     lag_pacf = pacf(df, nlags=20, method='ols')
    lag_pacf = pacf(ds_train, nlags=20, method='yw')
#     print lag_pacf
    upperInt = 1.96/np.sqrt(len(df))
    intPoint = -1

    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.xlabel("lags")
    plt.ylabel("PACF")
    plt.tight_layout()    
    plt.show()
    
    for i in range(0,len(lag_pacf)):
        if abs(lag_pacf[i]-upperInt) <= delta:
            print "p value using PACF is " + str(i)
            intPoint = i
            break
    
    model = ARIMA(ds_train, order=(intPoint, 0, 0)) #p=4, d=0, q=0  
    results_AR = model.fit(disp=0)
    print "Parameters of Autoregressive Model AR(%d) are:" %intPoint
    print results_AR.params
    plt.plot(ds_train, color='blue', label='Training Set')
    plt.plot(results_AR.fittedvalues, color='red', label="AR_fitted")
    plt.legend(loc='best')
    plt.xlabel("Time")
    plt.ylabel("Time series values")
    plt.title('AR(%d) Model with RMSE: %.4f' %(intPoint, ((sum((results_AR.fittedvalues-ds_train)**2))/len(ds_train))**0.5))
    plt.show()
    print "RMSE on Training Data: is %.4f" %((sum((results_AR.fittedvalues-ds_train)**2))/len(ds_train))**0.5
    
    #Run AR model for test data
    model_test = ARIMA(ds_test, order=(intPoint, 0, 0)) #p=4, d=0, q=0  
    results_AR_test = model_test.fit(disp=0)
    plt.plot(ds_test, color='blue', label='Test Set')
    plt.plot(results_AR_test.fittedvalues, color='red', label="AR_fitted")
    plt.legend(loc='best')
    plt.xlabel("Time")
    plt.ylabel("Time series values")
    plt.title('AR(%d) Model on test Data with RMSE: %.4f' %(intPoint, ((sum((results_AR_test.fittedvalues-ds_test)**2))/len(ds_test))**0.5))
    plt.show()
    print "RMSE on test Data: is %.4f" %((sum((results_AR_test.fittedvalues-ds_test)**2))/len(ds_test))**0.5
    return


def main():
    while 1:
        modelType = input("What model do you want to Test ? Enter the number \n \
        Simple Moving Average : 1 \n \
        Exponential Smoothing : 2 \n \
        AR(p) : 3 \n \
        Exit : 0 \n")
        
        if modelType == 1 :
            print "Simple Moving Average Model"
            runSimpleMovingAvg()
            print  "Simple moving Average Simulation Finished" 
        elif modelType == 2:
            print "Exponential Smoothing Model"
            runExpSmoothing()
            print "Exponential Smoothing Simulation Finished"
        elif modelType == 3:
            print "Auto Regressive Model"
            runARModel()
            print "Auto Regressive Model Simulation Finished"
        else:
            print "Exiting...\n"
            break
main()