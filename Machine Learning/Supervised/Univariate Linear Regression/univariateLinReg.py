import matplotlib.pyplot as plt
import numpy as np

# load data
data = np.loadtxt('ex1data1.txt', delimiter= ',')

# split data into x and y arrays
X,y = np.split(data,2, axis=1)

# add column of 1's before x values
X = np.concatenate((np.ones((X.shape)),X), axis = 1)


def gradientDescent(theta, alpha, X, y):
	# one step of GD
	m = X.shape[0]

	temp0 = theta[0][0] - alpha * (1.0 / m) * sum(np.dot(X,theta) - y)
	temp1 = theta[1][0] - alpha * (1.0 / m) * sum((np.dot(X,theta) - y) * X[:,1].reshape(m,1))

	theta[0][0] = temp0
	theta[1][0] = temp1

	return theta

def costFunction(theta, X, y):
	# computes cost function
	J = (1.0 / (2 * X.shape[0])) * sum(np.square(np.dot(X,theta) - y))
	return J

def main():
	mincost = 5
	maxIter = 10000
	alpha = 0.01
	theta = np.array([[0],[0]],dtype='float32')
	J = costFunction(theta, X, y)

	t0History = [theta[0][0]]
	t1History = [theta[1][0]]
	JHistory = [J]

	for iterations in range(0,maxIter):
		theta = gradientDescent(theta, alpha, X, y)
		J = costFunction(theta, X, y)

		t0History.append(theta[0][0])
		t1History.append(theta[1][0])
		JHistory.append(J)

	
	plt.plot(JHistory)
	plt.show()
	
	plt.scatter(X[:,1],y, s=20, marker='.')
	xx = np.arange(5,23)
	yy = theta[0][0]+theta[1][0]*xx
	plt.plot(xx,yy)
	plt.show()
	print(theta)





main()