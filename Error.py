import numpy as np
import mnist as mnist
import math
import matplotlib.pyplot as plt
import random
(train_x, train_y), (test_x, test_y) = mnist.load()
def Error(x,y):
    
    A = [train_x[train_y==i,:] for i in range(9)]

    train = np.concatenate((A[x], A[y]), axis=0)

    n=5000

    y1=np.ones(len(A[x]))
    y2=np.ones(len(A[y]))*(-1)

    trainy=np.concatenate((y1,y2))
    index_value = random.sample(list(enumerate(train)), n)
    indexes = []
    for idx, val in index_value:
        indexes.append(idx)
    xtrain=[train[i] for i in indexes]
    ytrain=[trainy[i] for i in indexes]
    
    B= [test_x[test_y==i,:] for i in range(9)]

    test = np.concatenate((B[x],B[y]),axis=0)

    m=len(test)
    y1=np.ones(len(B[x]))
    y2=np.ones(len(B[y]))*(-1)
    ytest=np.concatenate((y1,y2))
    def K(x,y):
        return math.exp(-(np.linalg.norm(x-y))**2/(784))
    def f(l):
        b=np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                b[i][j]=K(test[i],xtrain[j])
        b=np.matrix(b) 
        I=np.identity(n)
        k=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                k[i][j]=K(xtrain[i],xtrain[j])
        k=k+l*I
    
        k=np.matrix(k)
        k_inv=np.linalg.inv(k)
        y=ytrain
        temp1=np.matmul(b,k_inv)
        temp2=np.matmul(temp1,y)
        return temp2
    y_bar=np.average(ytest)
    ybar=np.ones(m)*y_bar
    z=[0, 0.01,0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
    error=[]
    for l in z:
        num=0
        den=0
        y_pred1=f(l)
        y_pred2=np.ravel(y_pred1)
        te = np.where(y_pred2<0, -np.ones_like(y_pred2), np.ones_like(y_pred2))
        num = np.linalg.norm(te - ytest)**2
        den = np.linalg.norm(ybar-ytest)**2
        error.append(num/den)
    return error