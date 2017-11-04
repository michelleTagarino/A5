#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import matplotlib.pyplot as plt
import numpy as np

plt.figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
x,y = np.loadtxt('plot50.txt', delimiter=',', unpack=True)
a,b = np.loadtxt('plot5.txt', delimiter=',', unpack=True)
c,d = np.loadtxt('plot0.txt', delimiter=',', unpack=True)

plt.plot(x, y, color="blue")
plt.plot(a, b, color="red")
plt.plot(c, d, color="green")

v = [0, 5, 100, 1200]
labels = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

plt.xticks(x, labels, rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)

plt.xlabel('Alpha\n')
plt.ylabel('Average steps\n')
plt.title('Average Learning Curves for Dyna-Q\nGreen: 0 Planning steps - Red: 5 Planning steps - Blue: 50 Planning steps')
plt.axis(v)
plt.show()