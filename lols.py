import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi


def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


side = 1

step = .1

x = np.arange(-side, side+step, step)
y = [gaussian(x_i, 0.5, 1) for x_i in x]
sum = 0
for y_i in y:
    sum = sum + y_i
sum = sum / len(x)
plt.plot(x,y)
plt.show()
print('sum = ', sum)