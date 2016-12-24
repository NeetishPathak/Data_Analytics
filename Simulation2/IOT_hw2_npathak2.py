

#Assigning the system constants
vm = [6,10,2,5]

arrItvl = input('Enter the mean inter-arrival time\n')#vm[0]
rtItvl = input('Enter the mean orbiting time\n')#vm[3]
srvItvl = input('Enter the service time\n')#vm[1]
qMaxLen = input('Enter the buffer size\n')#vm[2]
MAX_TIME = input('Enter the Master clock termination time\n')

firstArvl = 2
#Assigning initial values
mc = 0
cla = 2
cls = 0
qLen = 0
rtQueue = []
count = 0
f = open('output1.txt','w').close()
f = open('output1.txt','w')

trcArr = [mc,cla,cls,qLen,rtQueue]

print "MC" + "    CLA" + "    CLS" + "    Queue" + "    CLR"
print  str(trcArr[0]) + "    " + str(trcArr[1]) + "    " + str(trcArr[2]) + "    " + str(trcArr[3]) + "    " + str(trcArr[4])     
f.write("MC" + "    CLA" + "    CLS" + "    Queue" + "    CLR" + "\n")
f.write(str(trcArr[0]) + "    " + str(trcArr[1]) + "    " + str(trcArr[2]) + "    " + str(trcArr[3]) + "    " + str(trcArr[4]) + "\n")

def shortestVal(rtQ):
    
    if trcArr[2] == 0 and len(trcArr[4]) == 0:
        return 1  
         
    if len(trcArr[4]) != 0 and trcArr[4][0] <= trcArr[1] and  trcArr[4][0] <= trcArr[2]:
        return 4
    elif trcArr[1] <= trcArr[2]:
        return 1
    else:
        return 2

def printVal(trcArr):
    print  str(trcArr[0]) + "    " + str(trcArr[1]) + "    " + str(trcArr[2]) + "    " + str(trcArr[3]) + "    " + str(trcArr[4])     
    f.write(str(trcArr[0]) + "    " + str(trcArr[1]) + "    " + str(trcArr[2]) + "    " + str(trcArr[3]) + "    " + str(trcArr[4]) + "\n")
while mc <= MAX_TIME:
    
    i = shortestVal(trcArr)    
        
    if trcArr[2] == 0:
        trcArr[2] += cla + srvItvl
        #qLen += 1
        
    if i == 1:
        mc = trcArr[i]
        trcArr[i] += arrItvl
        
        if qLen == qMaxLen:
            #del rtQueue[0]
            rtQueue.append(mc + rtItvl)
            rtQueue.sort()   
        else:
            qLen += 1
            
    elif i == 2:
        mc = trcArr[i]
        trcArr[i] += srvItvl
        qLen -= 1
        
        count += 1
    else:
        mc = rtQueue[0]
        if qLen == qMaxLen:
            
            rtQueue.append(mc + rtItvl)
        else:
            qLen += 1
        del rtQueue[0]
        trcArr[4] = rtQueue
    
    trcArr[0] = mc
    trcArr[3] = qLen
    printVal(trcArr)

f.close()