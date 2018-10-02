print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#plt.rcParams['font.sas-serig']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

xaaa = []
yaaa = []
#print(range(1,60000,1000))

for m in range(1,60000,1000):
    

    print(m)
    xaaa.append(m)

    # import some data to play with
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(0)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    print(X.shape, y.shape)
    
    # split train and test
    X_train, y_train = X[0:499, :], y[0:499]
    X_test, y_test = X[500:568, :], y[500:568]
    
    
    print(X_test.shape, y_test.shape)
    
    syn0 = 2*np.random.random((30,4)) - 1
    syn1 = 2*np.random.random((4,1)) - 1
    for j in range(m):
        l1 = 1/(1+np.exp(-(np.dot(X_train,syn0))))
        l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
        y_train = y_train.reshape(499,1)
        l2_delta = (y_train - l2)*(l2*(1-l2))
        #print(np.shape(y_train - l2))
        #print(np.shape(l2*(1-l2)))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += X_train.T.dot(l1_delta)
        
    l1 = 1/(1+np.exp(-(np.dot(X_test,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    y_test = y_test.reshape(68,1)
    for i in y_test:
        if i > 0.5:
            i = 1
        else:
            i = 0
            
    error = y_test - l2
    error = abs(error)
    
    #print((len(error) - sum(error)) / len(error))
    #print(len(error))
    #print(l3)
    
    yaaa.append((len(error) - sum(error)) / len(error))
    
print(xaaa)
plt.figure()
plt.plot(xaaa,yaaa)
plt.show()


