from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
   y_hat =  xTr.T @ w
   gradient = np.zeros((np.shape(w)[0], 1))
   loss = 0
   
   for i in range(np.shape(xTr)[1]):
     loss = loss + max(0, 1 - yTr[0,i]) * y_hat[i,0]
     
   loss = loss + lambdaa * (w.T @ w)
   gradient += 2 * lambdaa * w - xTr @ yTr.T
     
   
   return loss, gradient
