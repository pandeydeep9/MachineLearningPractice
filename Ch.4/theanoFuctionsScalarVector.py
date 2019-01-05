#now combint scalars with vectors
import numpy as np
import theano.tensor as T
from theano import function

#matxis defined now before they can be used, every matrix given a unique name
a = T.dmatrix('a')
b = T.dmatrix('b')
c = T.dmatrix('c')
d = T.dmatrix('d')

#more scalars defined
p = T.dscalar('p')
q = T.dscalar('q')
r = T.dscalar('r')
s = T.dscalar('s')
u = T.dscalar('u')

#complicated function
e = ( ( (a * p) + (b - q) - (c + r) ) * d/s ) * u

f = function([a,b,c,d,p,q,r,s,u],e)

a_data = np.array([[1,2],[4,3]]) #Just a simple matrix
b_data = np.array([[5,6],[7,8]])
c_data = np.array([[9,10],[11,12]])
d_data = np.array([[13,14],[15,16]])
#print(a_data[1][0])

normally = ( ( (a_data * 1.0) + (b_data - 2.0) - (c_data + 3.0) ) * d_data/4.0 ) * 5.0
fromTheano = f(a_data,b_data,c_data,d_data,1,2,3,4,5)

print("Expected: ", normally)
print ("Theano:" , fromTheano)

