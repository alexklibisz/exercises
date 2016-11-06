import theano
from theano import tensor

# define two scalars
a = tensor.dscalar()
b = tensor.dscalar()

# create an expression, c
c = a + b

# convert the expression into a callable object that
# takes a and b values and computes a value for c
f = theano.function([a,b], c)

# bind 1.5 to 'a' and 2.5 to 'b', evaluate 'c'
result = f(1.5,2.5)

print(result) # prints 4.0
