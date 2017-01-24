import matplotlib.pyplot as plt
import numpy as np

# load data
data = np.loadtxt('ex1data1.txt', delimiter= ',')

# split data into x and y arrays
X,y = np.split(data,2, axis=1)

# add column of 1's before x values
X = np.concatenate((np.ones((X.shape)),X), axis = 1)

# plt.scatter(x,y)
# plt.show()


def gradientDescent(t0, t1, alpha, X, y):
	# one step of GD
	return t0, t1

def costFunction(theta, X, y):
	# computes cost function
	J = (1.0 / (2 * X.shape[0])) * sum(np.square(np.dot(X,theta) - y))
	return J

def main():
	mincost = 5
	maxIter = 1000
	alpha = 0.01
	t0 = 0
	t1 = 0
	J = costFunction(t0, t1, X, y)

	t0History = [t0]
	t1HIstory = [t1]
	JHistory = [J]

	for iterations in range(0,maxIter):
		t0, t1 = gradientDescent(t0, t1, X, y)
		J = costFunction(t0, t1, X, y)

		t0History.append(t0)
		t1HIstory.append(t1)
		JHistory.append(J)
theta = np.array([[0],[0]])
print theta
print costFunction(theta, X, y)

# xtest = np.array([[2],
# 				  [3],
# 				  [5]])
# print np.concatenate((np.ones((xtest.shape)),xtest), axis = 1)

# ytest = np.array([[2],
# 				  [3],
# 				  [5]])

# theta = np.array([[2,3]])

# # print xtest
# # print
# # print theta


# # print (1.0 / (2 * xtest.shape[0])) * sum(np.square(np.dot(xtest,theta.T) - ytest))

