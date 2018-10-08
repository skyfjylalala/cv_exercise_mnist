print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
#from sklearn import datasets
import mnist_data

#plt.rcParams['font.sas-serig']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

xaaa = []
yaaa = []
#print(range(1,60000,1000))

for m in range(1,6000,100):
    

    print(m)
    xaaa.append(m)

    # import some data to play with
    dataset = mnist_data.fetch_traingset();
    
    
    # split train and test
    X_train, y_train = dataset['images'][:],dataset['labels'][:];

        # shuffle
    X_train = np.array(X_train);
    y_train = np.array(y_train);
    X_train = X_train.reshape(60000,784)
    y_train = y_train.reshape(60000,1)
    idx = np.arange(X_train.shape[0])
    np.random.seed(0)
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    # standardize
    #mean = std = idx
    #print(mean.shape)
    #mean = X_train[idx].mean(axis=0)
    #std = X_train[idx].std(axis=0)
    #X_train[idx] = (X_train[idx] - mean[idx]) / std[idx]
    
    #print(X_train.shape, y_train.shape)

    dataset = mnist_data.fetch_testingset();
    X_test, y_test = dataset['images'][:],dataset['labels'][:];

    # shuffle
    X_test = np.array(X_test);
    X_test = X_test.reshape(10000,784)
    idx = np.arange(X_test.shape[0])
    np.random.seed(0)
    np.random.shuffle(idx)
    y_test = np.array(y_test);
    y_test = y_test.reshape(10000,1)
    X_test = X_test[idx]
    y_test = y_test[idx]
    
    # standardize
    #mean = std = idx
    #mean = X_test[idx].mean(axis=0)
    #std = X_test[idx].std(axis=0)
    #X_test[idx] = (X_test[idx] - mean[idx]) / std[idx]
    #print(X_test.shape, y_test.shape)
    
    syn0 = 2*np.random.random((784,800)) - 1
    syn1 = 2*np.random.random((800,500)) - 1
    syn2 = 2*np.random.random((500,200)) - 1
    syn3 = 2*np.random.random((200,1)) - 1

    for j in range(m):
        l1 = np.maximum(np.dot(X_train,syn0),0)
        l2 = np.maximum(np.dot(l1,syn1),0)
        l3 = np.maximum(np.dot(l2,syn2),0)
        l4 = np.maximum(np.dot(l3,syn3),0)

        l4_delta = (y_train - l4)*(l4*(1-l4))
        l3_delta = l4_delta.dot(syn3.T) * (l3 * (1-l3))
        l2_delta = l3_delta.dot(syn2.T) * (l2 * (1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))

        syn3 += l3.T.dot(l4_delta)
        syn2 += l2.T.dot(l3_delta)
        syn1 += l1.T.dot(l2_delta)
        syn0 += X_train.T.dot(l1_delta)
        
    l1 = np.maximum(np.dot(X_test,syn0),0)
    l2 = np.maximum(np.dot(l1,syn1),0)
    l3 = np.maximum(np.dot(l2,syn2),0)
    l4 = np.maximum(np.dot(l3,syn3),0)
    
        
        #for i in y_test:
        #    if i > 0.5:
        #        i = 1
        #else:
        #    i = 0
        
    #print(error.shape,y_test.shape,l2.shape)
    error = y_test - l4
    errorsum = sum(abs(error))
            
            #print((len(error) - sum(error)) / len(error))
        #print(len(error))
        #print(l3)
        
    yaaa.append(errorsum)
        
print(xaaa)
plt.figure()
plt.plot(xaaa,yaaa)
plt.show()


