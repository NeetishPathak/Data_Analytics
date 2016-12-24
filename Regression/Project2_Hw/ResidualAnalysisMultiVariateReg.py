import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from matplotlib import style
import numpy.random as random

#degrees of freedom for simple linear regression
p = 3
style.use('fivethirtyeight')

def showPlot():
    plt.show()
    return
    

def best_fit_slope_and_intercept(xVals, yVals):
    m = ( ((mean(xVals) * mean(yVals)) - mean(xVals*yVals)) /
          (mean(xVals)**2 - mean(xVals**2)) )
    b = mean(yVals) - m*mean(xVals)
    return m, b

#calculate SSM
def squared_model(y_orig,y_line):
    return sum((y_line-mean(y_orig))**2)

#Calculate SSE
def squared_error(y_orig, y_line):
    return sum((y_orig - y_line)**2)

#calculate SST
def squared_total(y_orig, y_line):
    return sum((y_orig - mean(y_orig))**2)

#calculate MSM
def mean_squared_model(y_orig,y_line,p):
    dof= p-1
    return squared_model(y_orig, y_line)/dof

#calculate MSE
def mean_squared_error(y_orig,y_line,p):
    dof = len(y_orig)-p
    return squared_error(y_orig, y_line)/dof

#calculate MST
def mean_squared_total(y_orig, y_line):
    dof = len(y_orig)-1
    return squared_total(y_orig, y_line)/dof

#calculate F value
def calc_F(y_orig,y_line,p):
    return mean_squared_model(y_orig, y_line, p)/mean_squared_error(y_orig, y_line, p)
    
#Calculate R^2
def coefficient_of_determination(y_orig, y_line):
    squared_error_regr = squared_error(y_orig, y_line)
    return 1 - (squared_error_regr/squared_total(y_orig, y_line))

def calcChiVals(obs, expec):
    i = 0
    chs = 0
    while i < len(obs):
        diff = obs[i] - expec[i]
        if(expec[i] != 0):
            chs += np.power(diff,2)/expec[i]
        i += 1
#             chs = np.sum(np.power(diff,2)/expec)
    return chs



#Read the csv and make a dataframe
df = pd.read_csv('PATHAK%20NEETISH.csv')
# print df.head()

print "Linear regresssion model X1, Y"
x1Vals = np.array(df['X1'],dtype = np.float64)
x2Vals = np.array(df['X2'],dtype = np.float64)
x3Vals = np.array(df['X3'],dtype = np.float64)
x4Vals = np.array(df['X4'],dtype = np.float64)
x5Vals = np.array(df['X5'],dtype = np.float64)

yVals = np.array(df['Y'],dtype = np.float64) 

#Multivariate regression
#Using all predictor variables
coeff = [2.7693,2.4599,4.7774,4.4627,5.8932, -1.7921] #Include all X1,X2,X3,X4,X5
coeff = [2.7694,2.5228,0.0,4.4618,5.8899,-1.4444] #Drop X3
coeff = [2.8742,318.7123,0,0,0,-450.6407] #X1-X2
coeff = [0,312.4168,0,0,97.5023,118.1945] #X2-X5
coeff = [0,0,0,4.4085,97.7841,548.3237] #X4-X5
coeff = [2.8856,0,0,4.4993,0,-14.6123] #X1-X4
yfit = coeff[0] * x1Vals + coeff[1] * x2Vals + coeff[2] * x3Vals + coeff[3] * x4Vals + coeff[4] * x5Vals + coeff[5]



#Print s_square
error_var = squared_error(yVals,yfit)/len(yVals)
print "s_square= " + str(error_var)
print "RMSE= " + str((mean_squared_error(yVals, yfit, p))**0.5) 

r_squared = coefficient_of_determination(yVals, yfit)
print "R_squared: " + str(r_squared)

F_val = calc_F(yVals, yfit, p)
print "F Statistics: " + str(F_val)

#Residual Analysis
yresid = yVals - yfit
SSresid = np.sum(np.power(yresid,2))
SStotal = len(yVals) * np.var(yVals)
rsq = 1 - SSresid/SStotal
print "R_squared: " + str(rsq)

random.seed(100)
#Q-Q plot of Residuals
yresid_sorted = np.sort(yresid)
norm = random.normal(0, np.std(yresid_sorted),len(yresid_sorted))
norm_sorted = np.sort(norm)
plt.plot(norm_sorted,yresid_sorted,"o")
z = np.polyfit(norm_sorted,yresid_sorted, 1)
p = np.poly1d(z)
plt.plot(norm_sorted,norm_sorted,"k--", linewidth=2)
plt.title("Normal Q-Q plot", size=28)
plt.xlabel("Theoretical quantiles", size=24)
plt.ylabel("Experimental quantiles", size=24)
plt.tick_params(labelsize=16)
showPlot()


#chi-square test
#Create a histogram from residuals
import scipy.stats as stats
binsCount = 33
a = np.max(np.abs(yresid))
maxVal = a
step = np.int((2*maxVal)/(binsCount)) + 1
xb = [x for x in range(-1*np.int(maxVal+1),np.int(maxVal + 1),step)]
okok = plt.hist(yresid,xb,histtype='bar',color='b',label="residuals")
okok1 = plt.hist(norm,xb,histtype='step',color='k',label="normal")
plt.legend(loc='upper left')
# chs, p = stats.chisquare(okok[0],okok1[0])
chs = calcChiVals(okok[0], okok1[0])
print "Critical value for Chi-squared test: " + str(chs) #+  str(calcChiVals(okok[0], okok1[0]))
# print str(chs), str(p)
showPlot()
crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 29)   # Df = number of variable categories - 1

print("Critical value: ") + str(crit)
# print(crit)
p_value = 1 - stats.chi2.cdf(x=chs,  # Find the p-value
                             df=29)
# print("P value")
# print(p_value)


#Scatter Plot of Residuals
plt.title("Scatter Plot (residual vs Yfit)")
plt.ylabel("Residuals")
plt.xlabel("Yfit")
plt.scatter(yfit,yresid)
showPlot()
plt.title("Scatter Plot (residual vs Y)")
plt.ylabel("Residuals")
plt.xlabel("Y")
plt.scatter(yVals,yresid)
showPlot()

# import statsmodels.api as sm
# import pylab
# sm.qqplot(yresid, scale=np.std(yresid), line='45')
# pylab.show()
