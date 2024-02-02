from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit
from checkgradHingeAndRidge import checkgradHingeAndRidge
from ridge import ridge

from scipy import io
import numpy as np

# load the data:
data = io.loadmat('data/data_train_default.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr,xTv,yTr,yTv = valsplit(X,Y)
print(np.shape(yTr))
print(np.shape(xTr))

#e = 0.01
#lambdaa = 0.5
#w = np.array([[0.1]]*1024)
#a = 2 * xTr @ np.transpose(xTr) @ w
#print(a)
#print(checkgradHingeAndRidge(ridge, w, e, xTr, yTr, lambdaa))


# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr,yTr)

# evaluate spam filter on test set using default threshold
spamfilter(xTv,yTv,w_trained)
