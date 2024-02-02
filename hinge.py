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
   
   for i in range(np.shape(yTr)[1]):
         loss = loss + max(0, 1 - yTr[0,i] * y_hat[i,0])
         if 1 - yTr[0,i] * y_hat[i,0] > 0:
            gradient = gradient - (yTr[0,i] * xTr[:,i]).reshape(w.shape[0], 1)

   loss = loss + lambdaa * (w.T @ w)
   gradient = gradient + 2 * lambdaa * w
     
   
   return loss, gradient
