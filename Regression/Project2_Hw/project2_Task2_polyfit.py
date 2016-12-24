import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from matplotlib import style
from numpy.lib.polynomial import polyfit

#degrees of freedom for linear regression with polyfit
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

#Read the csv and make a dataframe
df = pd.read_csv('PATHAK%20NEETISH.csv')
# print df.head()

print "Linear regresssion model X1, Y (Polynomial Fit)"
xVals = np.array(df['X1'],dtype = np.float64)
yVals = np.array(df['Y'],dtype = np.float64) 

#Polynomial Fits
pl1 = polyfit(xVals, yVals, 1)
pl2 = polyfit(xVals, yVals, 2)
pl3 = polyfit(xVals, yVals, 3)
# print pl1
print ("[a0, a1, a2]")
print pl2
# print pl3

plt.scatter(xVals, yVals)
plt.xlabel("X1")
plt.ylabel("Y")
plt.title("Best Fit Curve")

# yfit = pl1[0] * xVals + pl1[1]
yfit = pl2[0] * (xVals**2) + pl2[1] * (xVals) + pl2[2]
# yfit = pl3[0] * (xVals**3) + pl3[1] * (xVals**2) + pl3[2] * (xVals) + pl3[3]
# print yfit
# print yVals
plt.plot(xVals, yfit, 'm', linewidth=0.1)
showPlot()

#Print s_square

error_var = squared_error(yVals,yfit)/len(yVals)
print "s_square= " + str(error_var)
print "RMSE= " + str((mean_squared_error(yVals, yfit, p))**0.5) 

r_squared = coefficient_of_determination(yVals, yfit)
print "R_squared: " + str(r_squared)

F_val = calc_F(yVals, yfit, p)
print "F Statistics: " + str(F_val)
