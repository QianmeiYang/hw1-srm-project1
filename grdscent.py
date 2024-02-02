
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    
    i_iter = 0
    loss, gradient = func(w0)

    w = w0
    while i_iter < maxiter:

        w = w - stepsize*gradient
        last_loss, last_gradient = loss, gradient
        loss, gradient = func(w)
        if stepsize >= eps:
            if loss < last_loss:
                stepsize *= 1.01
            else:
                stepsize *= 0.5
        else:
            break
        if np.linalg.norm(gradient) <= tolerance:
            break
        i_iter += 1
        
    return w
