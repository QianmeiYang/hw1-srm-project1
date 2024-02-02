import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE

    y_hat = xTr.T @ w
    gradient = np.zeros((np.shape(w)[0], 1))
    loss = 0

    for i in range(np.shape(yTr)[1]):
        loss = loss + math.log(1 + np.exp( -yTr[0,i] * y_hat[i,0] ) )
        gradient += np.reshape(np.divide(-xTr[:,i] * yTr.T[i,0],  1 + np.exp(yTr[0,i] * w.T @ xTr[:,i])), [-1, 1])


    return loss, gradient

