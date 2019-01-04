import theano.tensor as T
from theano import function

#scalars defined before they can be used, every scalar given a unique name
a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = T.dscalar('d')
e = T.dscalar('e')


f = ((a - b + c) * d) / e

#function is a theano construct  which relates inputs(a,b,c,d,e) to outputs(f)
g = function([a,b,c,d,e],f)

print("Expected: ((1 - 2 + 3)* 4) / 5.0 =", ((1 - 2 + 3)* 4) / 5.0)
print("Theano: ((1 - 2 + 3)* 4) / 5.0 =", g(1,2,3,4,5) )

