
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from scipy import io

def trainspamfilter(xTr,yTr):
    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Consider optimizing the input parameters for your loss and GD!


    # (not the most successful) EXAMPLE:
    #print("...using ridge")
    #f = lambda w : ridge(w,xTr,yTr,1)

    #w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-09,1000)

    # YOUR CODE HERE

    #f = lambda w : hinge(w,xTr,yTr,0.1)
    f = lambda w: logistic(w, xTr, yTr)
    #f = lambda w: ridge(w, xTr, yTr, 0.1)
    #w_trained = grdescent(f, np.zeros((xTr.shape[0],1)), 1e-01, 1000) #hinge
    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)), 1e-03, 1000) #ridge,logistic

    io.savemat('w_trained.mat', mdict={'w': w_trained})
    return w_trained



