# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/wps-office/FZFSK.TTF')

data = np.loadtxt('profits-population.txt', delimiter=',')

fig, ax = plt.subplots()

plt.scatter(data[:, 0], data[:, 1], c = 'b')
plt.title(u'城市人口与盈利分布', fontproperties = zhfont)
plt.xlabel(u'城市人口（万）', fontproperties = zhfont)
plt.ylabel(u'盈利（万）', fontproperties = zhfont)
plt.ylim(-10, 30)

v = [x * 0.80252685 for x in range(30)]

z = np.arange(0, 30, 1)
#z = data[:,0]
#line, = ax.plot(x, 2 * x)
line, = ax.plot(z, z * v, color = 'g')

def animate(i):
    line.set_ydata(v[i % 30] * z)  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(z, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              interval=1200, blit=True)
plt.show()