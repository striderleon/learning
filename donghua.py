# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/wps-office/FZFSK.TTF')

data = np.loadtxt('profits-population.txt', delimiter=',')

fig = plt.figure()

plt.scatter(data[:, 0], data[:, 1], c = 'b')
plt.title(u'城市人口与盈利分布', fontproperties = zhfont)
plt.xlabel(u'城市人口（万）', fontproperties = zhfont)
plt.ylabel(u'盈利（万）', fontproperties = zhfont)
plt.ylim(-10, 30)
plt.xlim(0,30)

v = [x for x in range(30)]

z = np.arange(0, 30, 1)

line, = plt.plot(z, z * v, color = 'g')

def animate(v):
    line.set_ydata(v * z)  # update the data
    return line,

ani = animation.FuncAnimation(fig, animate, v, interval=1200, blit=True)
plt.show()