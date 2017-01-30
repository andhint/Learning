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

def calculateResiduals(X, y, theta):
	E = y - (np.dot(X,theta))
	return E


if __name__ == "__main__":
	mincost = 5
	maxIter = 3000
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

	print('y = {} + {} * x'.format(theta[0][0], theta[1][0]))

	plt.figure(figsize=(10,10))
	plt.subplot(2,2,1)
	plt.scatter(X[:,1],y, s=5, marker='.')
	# plot our best fit line
	xs = np.arange(5,25)
	ys = theta[0][0]+theta[1][0]*xs
	plt.xlabel('Population of City in 10,000s')
	plt.ylabel('Profit in $10,000s')
	plt.plot(xs,ys)

	plt.subplot(2,2,2)
	plt.xlabel('Iterations of GD')
	plt.ylabel('Cost J(theta)')
	plt.plot(JHistory)
	
	plt.subplot(2,2,3)
	plt.xlabel('Iterations')
	plt.ylabel('Value of theta0')
	plt.plot(t0History)

	plt.subplot(2,2,4)
	plt.xlabel('Iterations')
	plt.ylabel('Value of theta1')
	plt.plot(t1History)
	plt.show()

	E = calculateResiduals(X, y, theta)
	plt.scatter(X[:,1], E, s=10, marker='.')
	plt.xlabel('Observed Value')
	plt.ylabel('Residual')
	plt.grid()
	plt.show()