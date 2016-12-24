import pandas as pd
from pandas.stats.api import ols
raw_data = pd.read_csv("PATHAK%20NEETISH.csv") #any dataset will work. You can get the data from my github
raw_data.head(3)
raw_data['X1_sq'] = raw_data['X1']**2
raw_data['X1_cube'] = raw_data['X1']**3
# result = ols(y=raw_data['Y'],x=raw_data[['X1','X4']])
#result = ols(y=raw_data['Y'],x=raw_data[['X4','X5']])
result = ols(y=raw_data['Y'],x=raw_data[['X1_cube','X1_sq', 'X1']])
print result
