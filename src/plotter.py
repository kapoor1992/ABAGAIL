import matplotlib.pyplot as plt
import numpy as np

# neural net
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
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.show()

#knapsack
x_vals = [1000, 2000, 3000, 4000, 5000]
y_vals_rhc = [3238.6406018118596, 3316.980762513968, 3401.895021218431, 3066.2902910544535, 3387.6841700561954]
y_vals_sa = [3367.3769873831384, 3341.526836258415, 3518.864329674126, 3152.7008678489155, 3500.490384494412]
y_vals_ga = [3652.0617108141314, 3597.687551192289, 3777.310594825394, 3421.860440328449, 3842.2405067697923]
y_vals_mm = [3727.1612679802465, 3687.425341047327, 3862.080944114035, 3469.5337103866077, 3919.1711174386132]

plt.title('Knapsack Fitness')
plt.xlabel('Iterations')
plt.ylabel('Score')

plt.plot(x_vals, y_vals_rhc, label='Randomized Hill Climbing')
plt.plot(x_vals, y_vals_sa, label='Simulated Annealing')
plt.plot(x_vals, y_vals_ga, label='Genetic Algorithm')
plt.plot(x_vals, y_vals_mm, label='MIMIC')

plt.xticks(np.arange(1000, 5001, 1000))
plt.yticks(np.arange(0, 4001, 200))
plt.legend()
plt.show()