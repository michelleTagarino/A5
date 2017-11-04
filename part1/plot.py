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

plt.plot(x,y, color="blue")
plt.plot(a,b, color="red")
plt.plot(c,d, color="green")

x = np.linspace(0, 50, 6)
y = np.linspace(0,2000,5000)

a = np.linspace(0, 50, 6)
b = np.linspace(0,2000,5000)

c = np.linspace(0, 50, 6)
d = np.linspace(0,2000,000)

labels = [2, 10, 20, 30, 40, 50]
plt.xticks(x, labels)

plt.margins(0.2)

plt.xlabel('Episodes')
plt.ylabel('Steps per episode\n')
plt.title('Average Learning Curves for Dyna-Q\nGreen: 0 Planning steps - Red: 5 Planning steps - Blue: 50 Planning steps')
plt.axis([2,50,0,800])
plt.show()