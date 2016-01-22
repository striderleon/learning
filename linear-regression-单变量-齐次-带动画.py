# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
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
    v_history = zeros(shape=(num_iters, 1)) 

    v = 0

    for i in range(num_iters):

        predictions = Z * v
        
        #这里用到了导数
        errors = (predictions - T) * Z

        v = v - eta * (1.0 / m) * errors.sum()

        J_history[i, 0] = compute_cost(Z, T, v)
        v_history[i, 0] = v

    return v, J_history, v_history


#Load the dataset
data = loadtxt('profits-population.txt', delimiter=',')

Z = data[:, 0]
T = data[:, 1]


#number of training samples
m = T.size

#Initialize the parameter v
v = 0


#<iterations = 200, eta = 0.01>无震荡
#<iterations = 200, eta = 0.02>有震荡，但趋于稳定
#<iterations = 200, eta = 0.025>有震荡，但趋于发散

#Some gradient descent settings
iterations = 500
#eta= 0.01
eta = 0.02

#compute and display initial cost
print 'initial cost %f:' %compute_cost(Z, T, v)

v, J_history, v_history = gradient_descent(Z, T, v, eta, iterations)

print 'v:%f' %v

#print J_history

#print v_history

#compute and display final cost
print 'final cost %f:' %compute_cost(Z, T, v)

########################################################

#plotting the results

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/wps-office/FZFSK.TTF')

data = np.loadtxt('profits-population.txt', delimiter=',')

fig = plt.figure()

plt.scatter(data[:, 0], data[:, 1], s = 55, c = 'b')
plt.title(u'城市人口与盈利分布', fontproperties = zhfont)
plt.xlabel(u'城市人口（万）', fontproperties = zhfont)
plt.ylabel(u'盈利（万）', fontproperties = zhfont)
plt.ylim(-10, 30)
plt.xlim(0,30)

v = v_history
v_start = 0
#插入(0,0)点
Z = np.insert(Z, 0, 0)

line, = plt.plot(Z, Z * v_start, color = 'g')

def animate(v):
    line.set_ydata(Z * v)  # update the data
    return line,

ani = animation.FuncAnimation(fig, animate, v, interval=200, blit=True)
plt.show()