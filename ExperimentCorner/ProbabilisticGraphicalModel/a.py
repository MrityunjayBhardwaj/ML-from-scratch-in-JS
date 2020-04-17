import numpy as np

def isQuasiConvex(x,y,f):
	a = np.random.random();
	return (f(a*(x) + (1-a)*(y)) <= (Math.max( f(x), f(y) )) );



def f(x):
  sum = 0;
  for i in range(x.shape[0]):
  	sum += np.abs(x[i])
    print(sum)
    if sum > 1:
        return i
  if sum <= 1:
    print(x, np.Infinity, sum)
    return np.Infinity



x_ = np.random.rand([5,1]);
y_ = np.random.rand([4,1]);

print(isQuasiConvex(x_,y_,f))
