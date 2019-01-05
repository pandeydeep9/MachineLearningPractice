#now play with the activation functions
import theano.tensor as T
from theano import function
import numpy as np

np.set_printoptions(suppress=True)


# sigmoid Activation
a = T.dmatrix('a')

#nnet is a theano package which has activation functions
f_a = T.nnet.sigmoid(a)

#the input to the sigmoid function f_a is the matrix a
f_sigmoid = function([a],[f_a])

print("Sigmoid:      ", f_sigmoid([[-1,-100,-10,0,1]]))

#The tanh activation
b = T.dmatrix('b')
f_b = T.tanh(b)
f_tanh = function([b], [f_b])
print("Tanh:         ", f_tanh([[-1,-100,-10,0,1]]))

#Fast sigmoid
c = T.dmatrix('c')
f_c = T.nnet.ultra_fast_sigmoid(c)
f_fast_sigmoid = function([c],[f_c])
print("Fast Sigmoid: ",f_fast_sigmoid([[-1,-100,-10,0,1]]))

#softplus
d = T.dmatrix('d')
f_d = T.nnet.softplus(d)
f_softplus = function([d],[f_d])
print("Softplus:     ",f_softplus([[-1,-100,-10,0,1]]))

#relu
e = T.dmatrix('e')
f_e = T.nnet.relu(e)
f_relu = function([e],[f_e])
print("Relu:         ",f_relu([[-1,-100,-10,0,.1,.2,1,10]]))

#softmax
f = T.dmatrix('f')
f_f = T.nnet.softmax(f)
f_softmax = function([f],[f_f])
print("Softmax:      ",np.sum( f_softmax([[-1,-100,-10,0,1]]) ) )
print("Softmax:      ",f_softmax([[-1,0,1]]), np.sum(f_softmax([[-1,0,1]]) ) )
