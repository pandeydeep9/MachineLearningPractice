#Gradients
'''
Cost must be a scalar(i.e the one to find gradient of)
'''
import theano.tensor as T
from theano import function
from theano import shared
import numpy as np

x = T.dmatrix('x')
y = shared(np.array([[5,6]]))

# z = x^3 + xy
z = T.sum( ((x*x) + y) * x )
f = function(inputs = [x],outputs = [z])

g = T.grad(z,[x])
g_f = function([x],g)

print("Original: ", f([[2,3]]) )
print("ORiginal gradient: ", g_f([[2,3]]))

y.set_value(np.array([[6,7]]))

print("Updated: ", f([[2,3]]) )
print("Updated gradient: ", g_f([[2,3]]))