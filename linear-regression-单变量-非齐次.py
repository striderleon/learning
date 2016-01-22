from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour


#Evaluate the linear regression
def compute_cost(Z, T, V):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = T.size

    predictions = Z.dot(V).flatten()

    sqErrors = (predictions - T) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(Z, T, V, eta, num_iters):
    '''
    Performs gradient descent to learn V
    by taking num_iters gradient steps with learning
    rate eta
    '''
    m = T.size
    J_history = zeros(shape=(num_iters, 1))
    
    V[0][0] = 0

    for i in range(num_iters):

        predictions = Z.dot(V).flatten()

        errors_x1 = (predictions - T) * Z[:, 0]
        errors_x2 = (predictions - T) * Z[:, 1]

        V[0][0] = V[0][0] - eta * (1.0 / m) * errors_x1.sum()
        V[1][0] = V[1][0] - eta * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(Z, T, V)

    return V, J_history


#Load the dataset
data = loadtxt('profits-population.txt', delimiter=',')

Z = data[:, 0]
T = data[:, 1]


#number of training samples
m = T.size

#Add a column of ones to X (interception data)
it = ones(shape=(m, 2))
it[:, 1] = Z

#Initialize V parameters
V = zeros(shape=(2, 1))

#Some gradient descent settings
iterations = 1500
eta = 0.01

#compute and display initial cost
print compute_cost(it, T, V)

V, J_history = gradient_descent(it, T, V, eta, iterations)

print V
