import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, pow, cos


def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def cosinusian(x, mu, length):
    return cos(pi/2 * (abs(x-mu)/length))


side = 4
disp = side/4
step = 1
mean = 0

x = np.arange(-side, side, step)
# y = [pow(-1, x_i) * gaussian(x_i, 0, side) for x_i in x]
y = [gaussian(x_i, 0, disp) for x_i in x]
sum = 0
for y_i in y:
    sum = sum + y_i
# sum = sum / len(x)

print('sum = ', sum)
plt.plot(x,y)
plt.show()

