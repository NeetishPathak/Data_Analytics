project3.py code is written to simulate the forecasting models as described in Project 3 of IOT Analytics course

Once the program is run, 
1.) First, it generates a a plot of the data "NewFdata.csv" which is useful to analysze the stationarity of the data

2.) Once the data plot is closed, the user is presented with fours options to choose from
 
What model do you want to Test ? Enter the number 
         Simple Moving Average : 1 
         Exponential Smoothing : 2 
         AR(p) : 3 
         Exit : 0 

Selecting the no. corresponding to the model runs the simulation for that model and generates plots and RMSE/parameter values as asked
in the Project document.


3.) An example run would be as given below. Please close the generated plots to proceed in the simulation. 0 should be selected to exit from the simulation loop
		 
What model do you want to Test ? Enter the number 
         Simple Moving Average : 1 
         Exponential Smoothing : 2 
         AR(p) : 3 
         Exit : 0 		 
		 
3
Auto Regressive Model
p value using PACF is 4
Parameters of Autoregressive Model AR(4) are:
[  1.20412795e+03   5.42060881e-01   7.98105665e-02   1.70751438e-01
  -2.63419621e-02]
RMSE on Training Data: is 18.3122
RMSE on test Data: is 18.2154
Auto Regressive Model Simulation Finished