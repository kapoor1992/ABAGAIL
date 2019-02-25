import matplotlib.pyplot as plt
import numpy as np

x_vals = [100, 200, 300, 400, 500]
y_vals_rhc = [0.5263855421686747, 0.6080722891566265, 0.6767469879518073, 0.6584337349397591, 0.7126506024096386]
y_vals_sa = [0.5131325301204819, 0.4932530120481927, 0.509277108433735, 0.5199999999999999, 0.49710843373493974]
y_vals_ga = [0.6406024096385542, 0.6619277108433735, 0.6981927710843374, 0.6842168674698794, 0.6508433734939759]

plt.title('Mammography Neural Network Weight Training')
plt.xlabel('Iterations')
plt.ylabel('Score')

plt.plot(x_vals, y_vals_rhc, label='Randomized Hill Climbing')
plt.plot(x_vals, y_vals_sa, label='Simulated Annealing')
plt.plot(x_vals, y_vals_ga, label='Genetic Algorithm')

plt.xticks(np.arange(100, 501, 100))
plt.legend()
plt.show()