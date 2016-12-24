import numpy as np
import pandas as pd
#from matplotlib import cm, pyplot as plt
import csv

from hmmlearn import hmm

#csv = np.genfromtxt ('PATHAK NEETISH.csv', delimiter=",")
import csv
with open('PATHAK NEETISH.csv', 'rb') as f:
    reader = csv.reader(f)
    results = list(reader)

data = pd.read_csv('PATHAK NEETISH.csv', 'rb')
#print data
#data = np.array(data)

#train_data = np.array(data[:1600])
train_data = data[:5]
'''
test_data = np.array(data[1600:])
print train_data
print len(train_data)
print test_data
print len(test_data)
'''
#X = np.array(train_data)
#print np.array([X]).T
#X = np.array([[1,2,3,4,5]]).T
X = train_data
print X
#X.reshape(-1,1)
model = hmm.MultinomialHMM(n_components=3).fit(X)
#model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000).fit(X)
# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print 'done'
'''
print("Transition matrix")
print(model.transmat_)
print()

'''
'''
print("Start Prob matrix")
print(model.emissionprob_)
print()
'''

'''
print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()
'''
