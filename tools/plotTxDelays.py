#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mpldatacursor

map = np.loadtxt("txMap.dlm")
delays = np.loadtxt("delays.dlm")

fig, ax = plt.subplots()
ax.imshow(delays, interpolation='none')

mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
plt.show()

#plt.ion()
#plt.show()
#plt.imshow(delays)

