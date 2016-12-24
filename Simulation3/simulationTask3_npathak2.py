import math
import random

#import matplotlib.pyplot as plt
#from matplotlib import style

NUM_TIMES = 50
IOT_DEVICE_COUNT = 1000

#Poisson(Exponential) random value generator
def expoStochasticVar(iat):
    
    rndm = random.random()
    nxtTime = ((-1) * math.log(rndm))*(iat)
    #a = "%.2f" % nxtTime
    return round(nxtTime,2)

#Shortest time value among the next arrival time, service completion time and retransmission arrival time
def shortestVal(trcArr):   
    if len(trcArr[1]) != 0 and len(trcArr[2]) != 0 and len(trcArr[4]) != 0:
        if trcArr[1][0] <= trcArr[2][0] and trcArr[1][0] < trcArr[4][0][0]:
            return 1
        elif trcArr[2][0] < trcArr[4][0][0]:
            return 2
        else:
            return 4
    if len(trcArr[1]) != 0 and len(trcArr[2]) != 0 and len(trcArr[4]) == 0:
        if trcArr[1][0] <= trcArr[2][0]:
            return 1
        else:
            return 2  
    if len(trcArr[1]) != 0 and len(trcArr[2]) == 0 and len(trcArr[4]) != 0:
        if trcArr[1][0] < trcArr[4][0][0]:
            return 1
        else:
            return 4
    if len(trcArr[1]) != 0 and len(trcArr[2]) == 0 and len(trcArr[4]) == 0:
        return 1  
    if len(trcArr[1]) == 0 and len(trcArr[2]) != 0 and len(trcArr[4]) != 0:
        if trcArr[2][0] < trcArr[4][0][0]:
            return 2
        else:
            return 4
    if len(trcArr[1]) == 0 and len(trcArr[2]) != 0 and len(trcArr[4]) == 0:
            return 2
    if len(trcArr[1]) == 0 and len(trcArr[2]) == 0 and len(trcArr[4]) != 0:
        return 4
        
    if len(trcArr[1]) == 0 and len(trcArr[2]) == 0 and len(trcArr[4]) == 0:
        return 0  

#print the current processing list
def printVal(trcArr, sQueue, f):
    if len(trcArr[2]) > 0:
        srvcVal = "["+str("%.2f" %trcArr[2][0]) +  str(", %d" %trcArr[2][1])+"]"
    else:
        srvcVal = str(trcArr[2])
        
    if len(trcArr[1]) > 0:
        arrVal = "["+str("%.2f" %trcArr[1][0]) +  str(", %d" %trcArr[1][1])+"]"
    else:
        arrVal = str(trcArr[1])
    '''
    print  str('%.2f' % trcArr[0]) \
            + '%30s'% str(arrVal) \
            + '%30s'% str(srvcVal) \
            + '%30s'% str(trcArr[3]) \
            + '%30s'% str(sQueue) \
            + "         " + str(trcArr[4])
    '''
    f.write(  str( '%.2f' % trcArr[0]) \
            + '%30s'% str(arrVal) \
            + '%30s'% str(srvcVal) \
            + '%30s'% str(trcArr[3]) \
            + '%30s'% str(sQueue) \
            + "         " + str(trcArr[4])
            + "\n") 

'''
#Functions for plotting curves

style.use('fivethirtyeight')
#style.use('ggplot')

fig = plt.figure()

def getCoordinates(repList):
    ax = []
    
    for i in range(len(repList)):
        x = i
        ax.append(x)
    
    return ax
    
def createPlots(repTList, repDList):
    axt = getCoordinates(repTList)
    ayt = repTList
    axd = getCoordinates(repDList)
    ayd = repDList
    ax1 = fig.add_subplot(121)
    ax1.set_title("Repetitions vs T")
    ax1.set_xlabel("Repetitions")
    ax1.set_ylabel("T")
    ax1.set_ylim([0,200])
    ax1.plot(axt,ayt,'r', linewidth = 1.0)
    ax2 = fig.add_subplot(122)
    ax2.set_title("Repetitions vs D")
    ax2.set_xlabel("Repetitions")
    ax2.set_ylabel("D")
    ax2.set_ylim([0,200])
    ax2.plot(axd,ayd,'b', linewidth=1.0)
    plt.show()

def createServicePlots(serviceQueue,Vals,yLabel,Title):
    ax = serviceQueue
    ay = Vals
    plt.plot(ax,ay,linewidth=1.5)
    plt.xlabel("Buffer Size")
    plt.ylabel(yLabel)
    plt.xlim([serviceQueue[0]-2,serviceQueue[len(serviceQueue)-1] + 1])
    #plt.ylim([0,max(Vals) + 10])
    plt.title(Title)
    plt.show()
       
def createTPlots(repTList):
    axt = getCoordinates(repTList)
    ayt = repTList
    plt.plot(axt, ayt, 'b')
    plt.xlabel("Repetition")
    plt.ylabel("Mean (T)")
    plt.show()

def createDPlots(repDList):
    axd = getCoordinates(repDList)
    ayd = repDList
    plt.plot(axd, ayd, 'r')
    plt.xlabel("Repetitions (N)")
    plt.ylabel("Mean (D)")
    #plt.title("Mean Waiting Time (D) vs Repetitions (N)")
    plt.show()
'''

def simulateIOTReqs(arrItvl, srvItvl,rtItvl,qMaxLen,fileS,num_Times):
    #Assigning the system constants
    vm = [arrItvl,srvItvl,qMaxLen,rtItvl]
    
    arrItvl = vm[0]
    rtItvl = vm[3]
    srvItvl = vm[1]
    qMaxLen = vm[2]
    
    #Setting seed value
    #random.seed(2000)
    
    #initializing the simulation
    firstArvl = expoStochasticVar(arrItvl)
    aCount = 1
    
    #Assigning initial values
    mc = 0
    cla = [firstArvl,1]
    cls = []
    rtQueue = []
    sQueue = []
    qLen = 0
    dCount = 0
    
    #Initializing list to keep the next processing record
    trcArr = [mc,cla,cls,qLen,rtQueue]
    
    #Initializing the dictionary to keep track of all IOT devices
    iotDevicesRecord = dict(dict())
    iotDevicesRecord[aCount] =  {'Arrival_Time':firstArvl,'Srvc_Entry':0, 'Dep_Time':0,
                                 'total_waiting':0,'total_time':0}
    
    #open the output File
    #f = open('output2.txt','w').close()
    #f = open('output2.txt','w')
    
    '''
    #print the first arrival values 
    print "MC" + '%30s' % "CLA" + '%30s' % "CLS" + '%30s' %"Queue Len" + '%30s' % "Queue" + "    CLR"
    print  str(trcArr[0]) \
                + '%30s'% str(trcArr[1]) \
                + '%30s'% str(trcArr[2]) \
                + '%30s'% str(trcArr[3]) \
                + '%30s'% str(sQueue) \
                + "         " + str(trcArr[4])
     '''
    '''               
    f.write("MC" + '%30s' % "CLA" + '%30s' % "CLS" + '%30s' %"Queue Len" + '%30s' % "Queue" + "    CLR" + "\n")
    f.write(str(trcArr[0]) \
                + '%30s'% str(trcArr[1]) \
                + '%30s'% str(trcArr[2]) \
                + '%30s'% str(trcArr[3]) \
                + '%30s'% str(sQueue) \
                + "         " + str(trcArr[4]) + "\n")
    '''
    #simulation loop for device requests 
    while dCount < IOT_DEVICE_COUNT:#mc <= MAX_TIME:
            
        i = shortestVal(trcArr)    
            
        if i == 1: #next arrival time has the shortest value
            mc = trcArr[i][0]
            
            if qLen == qMaxLen:
                rtQueue.append([mc + expoStochasticVar(rtItvl),trcArr[i][1]])
                if(len(rtQueue) > 0):
                    rtQueue = sorted(rtQueue,key=lambda x:x[0])        
            else:
                qLen += 1
                sQueue.append(trcArr[i][1])
                if(qLen == 1):
                    x = [mc + srvItvl, trcArr[i][1]]
                    trcArr[2] = x
            
            
            aCount += 1
            if aCount <= IOT_DEVICE_COUNT :
                trcArr[i][0] += expoStochasticVar(arrItvl)
                trcArr[i][1] += 1
                iotDevicesRecord[aCount] = {'Arrival_Time':firstArvl,'Srvc_Entry':0, 'Dep_Time':0,'total_waiting':0,'total_time':0}
                iotDevicesRecord[aCount]["Arrival_Time"] = trcArr[i][0]
                iotDevicesRecord[aCount]["Srvc_Entry"] = trcArr[i][0]
                iotDevicesRecord[aCount]["total_waiting"] = iotDevicesRecord[aCount]["Srvc_Entry"] - iotDevicesRecord[aCount]["Arrival_Time"]
            else:
                trcArr[i] = []
                
        elif i == 2:
            mc = trcArr[i][0]
            
            iotDevicesRecord[trcArr[i][1]]["Dep_Time"] = trcArr[i][0]
            iotDevicesRecord[trcArr[i][1]]["total_time"] = iotDevicesRecord[trcArr[i][1]]["Dep_Time"] - iotDevicesRecord[trcArr[i][1]]["Arrival_Time"]
            if(len(sQueue) > 1):
                trcArr[i][0] += srvItvl
                trcArr[i][1] = sQueue[1]
            else:
                trcArr[i] = []
            del sQueue[:1]    
            qLen -= 1
            dCount += 1
            
        elif i == 4:
            mc = rtQueue[0][0]
            
            if qLen == qMaxLen:
                rtQueue.append([mc + expoStochasticVar(rtItvl),rtQueue[0][1]])
                if(len(rtQueue) > 0):
                    rtQueue = sorted(rtQueue, key=lambda x: x[0])
            else:
                qLen += 1
                sQueue.append(rtQueue[0][1])
                iotDevicesRecord[rtQueue[0][1]]["Srvc_Entry"] = rtQueue[0][0]
                iotDevicesRecord[rtQueue[0][1]]["total_waiting"] = iotDevicesRecord[rtQueue[0][1]]["Srvc_Entry"] - iotDevicesRecord[rtQueue[0][1]]["Arrival_Time"]
                if(qLen == 1):
                    x = [mc + srvItvl, rtQueue[0][1]]
                    trcArr[2] = x
                
            del rtQueue[0]
            trcArr[4] = rtQueue
        
        trcArr[0] = mc
        trcArr[3] = qLen
        #printVal(trcArr, sQueue, f)
    
    #print dCount
    #print aCount
    
    first1000Devices = []
    for key, value in iotDevicesRecord.iteritems():
        #f.write(str(key) + " " + str(value))
        #f.write("\n")
        first1000Devices.append(value)
        
    
    #first1000Devices = iotDevicesRecord[0:IOT_DEVICE_COUNT]
    first1000Devices = first1000Devices[:1000]
    
    #print first1000Devices
    #mean total Time Values
    if len(first1000Devices) > 0:
        mean_T = sum([float(device['total_time']) for device in first1000Devices]) / len(first1000Devices)
        #sorted deviceArray for 95th Percentile value
        sortedDevArray = sorted(first1000Devices, key=lambda x:x["total_time"])
        devTValues = [float(i["total_time"]) for i in sortedDevArray]
        pct95_T = devTValues[int(0.95 * IOT_DEVICE_COUNT - 1)]
    else:
        mean_T = 0
        pct95_T = 0
    
    #mean waiting time values
    devWithsomeWaitingTime = filter(lambda x: x["total_waiting"] > 0.0, first1000Devices)
    if len(devWithsomeWaitingTime) > 0:
        mean_D =  sum([float(device["total_waiting"]) for device in devWithsomeWaitingTime]) / len(devWithsomeWaitingTime)
        #sorted deviceArray for devices with waiting time
        sortedWaitTimeArray = sorted(devWithsomeWaitingTime, key=lambda x:x["total_waiting"]);
        pct95_D = sortedWaitTimeArray[int(0.95 * len(devWithsomeWaitingTime) - 1)]["total_waiting"]
    else:
        mean_D = 0
        pct95_D = 0
    
    
    #departure_Time of Last Device
    totalReqProcessTime =  first1000Devices[IOT_DEVICE_COUNT - 1]["Dep_Time"]
    fileS.write("%10s" % str(NUM_TIMES - num_Times) \
                + "%30s" % str("%.2f" % mean_T) \
                + "%30s" % str("%.2f" % pct95_T) \
                + "%20s" % str("%.2f" % mean_D) \
                + "%30s" % str("%.2f" % pct95_D) \
                + "%20s" % str("%.2f" % totalReqProcessTime)
                + "\n")
    
    #createPlots(first1000Devices)
    
    #f.close()
    return mean_T, pct95_T ,mean_D, pct95_D, totalReqProcessTime


#mathematical functions
def calcMean(lst):
    return sum([float(m) for m in lst])/len(lst)

def calcStandardDev(lst, mean):
    lstdiff = [((x-mean)**2) for x in lst]
    var = sum(lstdiff)/len(lst)
    std = (var**(0.5))
    return std

def calcConfItvl(repetitions, grandMean, grandStd):
    sigByN = 1.96/(repetitions**(0.5))
    ciLim = sigByN * grandStd
    ciMax = grandMean + ciLim
    ciMin = grandMean - ciLim
    return (ciMin, ciMax)
    
def computerConfidenceItvl(repetitions,grandTStdQ,grandTMeanQ):
    sigByN = 1.96/(repetitions**(0.5))
    confidenceTLim = [sigByN * tSd for tSd in grandTStdQ]
    confidenceItvlTMax = [x + y for x,y in zip(grandTMeanQ, confidenceTLim)]
    confidenceItvlTMin = [x - y for x,y in zip(grandTMeanQ, confidenceTLim)] 
    confidenceItvlT = [(x,y) for x,y in zip(confidenceItvlTMin, confidenceItvlTMax)]
    return confidenceItvlT

        

def simulate(arrItvl, srvItvl,rtItvl,qMaxLen,repetitions):
    
    num_Times = repetitions
    
    #open the output File
    fileName = "S" + str(srvItvl) + "Tables.txt"
    
    fileS = open(fileName,'w').close()
    fileS = open(fileName,'w')
    
    tabString = "%10s" % "Repetitions" \
                + "%30s" % "Mean_T" \
                + "%30s" % "95th_Percentile_of_T" \
                + "%20s" % "Mean D" \
                + "%30s" % "95th_Percentile_of_T)" \
                + "%20s" % "P(total_Time) \n"
                
    fileS.write(tabString)
    
    meanTList = []
    pct95TList = []
    meanDList = []
    pct95DList = []
    totalTimeList = []
    
    while num_Times > 0:
        meanT, pct95T, meanD, pct95D, totalTime = simulateIOTReqs(arrItvl, srvItvl,rtItvl,qMaxLen,fileS,num_Times)
        meanTList.append(meanT)
        pct95TList.append(pct95T)
        meanDList.append(meanD)
        pct95DList.append(pct95D)
        totalTimeList.append(totalTime)
        num_Times -= 1
    
    grandTMean =  calcMean(meanTList)
    grandDMean =  calcMean(meanDList)
    grandTStd = calcStandardDev(meanTList, grandTMean)
    grandDStd = calcStandardDev(meanDList, grandDMean)
    grandPct95TMean = calcMean(pct95TList)
    grandPct95DMean = calcMean(pct95DList)
    grandTotalTime = calcMean(totalTimeList)
    grandTotalTimeStd = calcStandardDev(totalTimeList, grandTotalTime)
    
    '''
    print grandTMean
    print grandDMean
    print grandTStd
    print grandDStd
    print grandPct95TMean
    print grandPct95DMean
    print grandTotalTime
    print grandTotalTimeStd
    '''
    confidenceItvlT = calcConfItvl(repetitions, grandTMean, grandTStd)        
    confidenceItvlD = calcConfItvl(repetitions, grandDMean, grandDStd)
    confidenceItvlTot = calcConfItvl(repetitions, grandTotalTime, grandTotalTimeStd)
    
    print "IOT certification of 1000 devices completed for " + str(repetitions) + " repetitions for service Interval " + str(srvItvl) + " and Buffer size " + str(qMaxLen) + "\n" 
    print "Mean (T): " + str(grandTMean) + ", 95th Percentile (T): " +  str(grandPct95TMean) + ", Confidence Interval(T): " + str(confidenceItvlT) + "\n"
    print "Mean (D): " + str(grandDMean) + ", 95th Percentile (D): " +  str(grandPct95DMean) + ", Confidence Interval(D): " + str(confidenceItvlD) + "\n"
    print "Mean (P): " + str(grandTotalTime) + ", Confidence Interval(P): " + str(confidenceItvlTot) + "\n"
    
     
    fileS.close()
    #plt.title("Service Time = " + str(srvItvl))
    #createPlots(meanTList, meanDList)
    #createTPlots(meanTList)
    #plt.title("Service Time = " + str(srvItvl))
    #createDPlots(meanDList)
    
    return grandTMean, grandDMean, grandTStd, grandDStd,grandPct95TMean, grandPct95DMean, grandTotalTime, grandTotalTimeStd
    

def main():
    
    grandTMeanQ = []
    grandDMeanQ = []
    grandTStdQ = []
    grandDStdQ = []
    grandT95Q = []
    grandD95Q = []
    grandTotalQ = []
    grandTotalSdQ = []
    
    if 1:
        arrItvl = input('Enter the mean interarrival time (1/lambda)\n')#vm[0]
        srvItvl = input('Enter the service time (s)\n')#vm[1]
        rtItvl = input('Enter the mean orbiting time (1/d)\n')#vm[3]
        qMaxLen = input('Enter the buffer size (B)\n')#vm[2]
        #MAX_TIME = input('Enter the Master clock termination time\n')
        repetitions = input('Enter the no. of repetitions (N)\n')
        #random.seed(2000)
        x,y,z,w,p1,p2, t, tsd = simulate(arrItvl, srvItvl,rtItvl,qMaxLen,repetitions)
        grandTMeanQ.append(x)
        grandDMeanQ.append(y)
        grandTStdQ.append(z)
        grandDStdQ.append(w)
        grandT95Q.append(p1)
        grandD95Q.append(p2)
        grandTotalQ.append(t)
        grandTotalSdQ.append(tsd)
        
    else:
        arrItvl = 17.98
        rtItvl = 10
        #qMaxLen = [5,6,7,8,9,10]
        repetitions = 50
        serviceQueue = [i for i in range(5,200,5)]
        #serviceQueue = [16]
        srvItvl = 16   
        for j in serviceQueue:
            qMaxLen = j
            x,y,z,w,p1,p2, t, tsd = simulate(arrItvl, srvItvl,rtItvl,qMaxLen,repetitions)
            grandTMeanQ.append(x)
            grandDMeanQ.append(y)
            grandTStdQ.append(z)
            grandDStdQ.append(w)
            grandT95Q.append(p1)
            grandD95Q.append(p2)
            grandTotalQ.append(t)
            grandTotalSdQ.append(tsd)
            print j
        print grandTMeanQ
        print grandDMeanQ
        print grandTStdQ
        print grandDStdQ
        print grandT95Q
        print grandD95Q
        print grandTotalQ
        print grandTotalSdQ
        
        confidenceItvlT = computerConfidenceItvl(repetitions,grandTStdQ,grandTMeanQ)        
        confidenceItvlD = computerConfidenceItvl(repetitions,grandDStdQ,grandDMeanQ)
        confidenceItvlTotQ = computerConfidenceItvl(repetitions, grandTotalSdQ, grandTotalQ)
        
        print confidenceItvlT
        print confidenceItvlD
        print confidenceItvlTotQ
        
        '''
        createServicePlots(serviceQueue, grandTMeanQ, "Mean of T", "Service vs Mean(T) Plot")
        createServicePlots(serviceQueue, grandT95Q, "95th Percentiles of T", "Service vs 95th Percentile(T) Plot")
        createServicePlots(serviceQueue, confidenceItvlT, "T confidence Interval", "Service vs Cf Interval (T) Plot")
        createServicePlots(serviceQueue, grandDMeanQ, "Mean  of D", "Service vs Mean(D) Plot")
        createServicePlots(serviceQueue, grandD95Q, "95th Percentiles of D", "Service vs 95th Percentile(D) Plot")
        createServicePlots(serviceQueue, confidenceItvlD, "D confidence Interval", "Service vs Cf Interval (D) Plot")
       
        createServicePlots(serviceQueue, grandTMeanQ, "Mean of T", "Buffer Size vs Mean(T) Plot")
        createServicePlots(serviceQueue, grandT95Q, "95th Percentiles of T", "Buffer Size vs 95th Percentile(T) Plot")
        createServicePlots(serviceQueue, confidenceItvlT, "T confidence Interval", "Buffer Size vs Cf Interval (T) Plot")
        createServicePlots(serviceQueue, grandDMeanQ, "Mean  of D", "Buffer Size vs Mean(D) Plot")
        createServicePlots(serviceQueue, grandD95Q, "95th Percentiles of D", "Buffer Size vs 95th Percentile(D) Plot")
        createServicePlots(serviceQueue, confidenceItvlD, "D confidence Interval", "Buffer Size vs Cf Interval (D) Plot")        
       ''' 

main()

