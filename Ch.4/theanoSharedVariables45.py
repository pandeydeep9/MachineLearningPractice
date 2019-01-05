#Shared Variables(what are they??)
'''
A function defined using the shared variable computes its o/p using current value
 of shared variable. So the function can return multiple different values for 
 same input.
 -> Allows user to define function with internal state.
 Internal state can be updated arbitrarily.(COOL)
'''

import theano.tensor as T
from theano import function
from theano import shared
import numpy as np

x = T.dmatrix('x')
y = shared(np.array([[4,5,6]]))

z = x + y
f = function(inputs = [x],outputs = [z])

print("Original Shared Value: ", y.get_value() )#The way to access shared variable value
print("ORiginal Function Evaluation: ", f([[1,2,3]]))

y.set_value(np.array([[5,6,7]]))

print("Original Shared Value: ", y.get_value() )#The way to access shared variable value
print("ORiginal Function Evaluation: ", f([[1,2,3]]))