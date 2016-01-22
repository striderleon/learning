# -*- coding: utf-8 -*- 

from numpy import loadtxt, zeros


#Evaluate the linear regression
def compute_cost(Z, T, v):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = T.size

    predictions = Z * v

    sqErrors = (predictions - T) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(Z, T, v, eta, num_iters):
    '''
    Performs gradient descent to learn v
    by taking num_iters gradient steps with learning
    rate eta
    '''
    m = T.size
    J_history = zeros(shape=(num_iters, 1))

    v = 0

    for i in range(num_iters):

        predictions = Z * v
        
        #这里用到了导数
        errors = (predictions - T) * Z

        v = v - eta * (1.0 / m) * errors.sum()

        J_history[i, 0] = compute_cost(Z, T, v)

    return v, J_history


#Load the dataset
data = loadtxt('profits-population.txt', delimiter=',')

Z = data[:, 0]
T = data[:, 1]


#number of training samples
m = T.size

#Initialize the parameter v
v = 0

#Some gradient descent settings
iterations = 1000
#eta= 0.01
eta = 0.01

#compute and display initial cost
print 'initial cost: %f' %compute_cost(Z, T, v)

v, J_history = gradient_descent(Z, T, v, eta, iterations)

print 'v: %f' %v

#compute and display final cost
print 'final cost: %f' %compute_cost(Z, T, v)

