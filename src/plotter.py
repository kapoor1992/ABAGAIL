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
x_vals = [100, 200, 300, 400, 500]
y_vals_rhc = [3250.989066845604, 3061.669874905297, 3155.8120383574974, 3291.841483454704, 3370.0104631776135]
y_vals_sa = [1820.8366296832123, 2054.346341018835, 1906.4074121693368, 2040.1769506851992, 2765.530703487632]
y_vals_ga = [3509.6109095803063, 3373.8348736323105, 3565.24098024238, 3642.71828090305, 3648.5742480798413]
y_vals_mm = [3815.739161902573, 3585.176284964021, 3707.217931724224, 3796.7560390742306, 3770.2366897444676]

plt.title('Knapsack Problem Fitness')
plt.xlabel('Iterations')
plt.ylabel('Score')

plt.plot(x_vals, y_vals_rhc, label='Randomized Hill Climbing')
plt.plot(x_vals, y_vals_sa, label='Simulated Annealing')
plt.plot(x_vals, y_vals_ga, label='Genetic Algorithm')
plt.plot(x_vals, y_vals_mm, label='MIMIC')

plt.xticks(np.arange(100, 501, 100))
plt.yticks(np.arange(0, 4001, 200))
plt.legend()
plt.show()

#tsp
x_vals = [100, 200, 300, 400, 500]
y_vals_rhc = [0.05337642074584015, 0.06257602517760222, 0.07213365101345817, 0.07258913757915782, 0.07694561939568231]
y_vals_sa = [0.03874067350868622, 0.03831807663854282, 0.039908764137717136, 0.038879685036048064, 0.03888242125863951]
y_vals_ga = [0.14703296722647605, 0.14717808740965074, 0.1493962200786588, 0.15180068673194816, 0.15861719147518055]
y_vals_mm = [0.06700777234547556, 0.07499800024339258, 0.08391089961897627, 0.0822516317041716, 0.08887177454423936]

plt.title('Travelling Salesman Problem Fitness')
plt.xlabel('Iterations')
plt.ylabel('Score')

plt.plot(x_vals, y_vals_rhc, label='Randomized Hill Climbing')
plt.plot(x_vals, y_vals_sa, label='Simulated Annealing')
plt.plot(x_vals, y_vals_ga, label='Genetic Algorithm')
plt.plot(x_vals, y_vals_mm, label='MIMIC')

plt.xticks(np.arange(100, 501, 100))
plt.yticks(np.arange(0, 0.21, 0.01))
plt.legend()
plt.show()

#nqueens
x_vals = [100, 200, 300, 400, 500]
y_vals_rhc = [2.2675736961451248E-4, 2.2675736961451248E-4, 2.2675736961451248E-4, 2.2831050228310502E-4, 2.277904328018223E-4]
y_vals_sa = [2.577319587628866E-4, 2.5188916876574307E-4, 2.551020408163265E-4, 2.487562189054726E-4, 2.506265664160401E-4]
y_vals_ga = [2.4390243902439024E-4, 2.4154589371980676E-4, 2.4271844660194174E-4, 2.4154589371980676E-4, 2.4096385542168676E-4]
y_vals_mm = [2.3640661938534278E-4, 2.3866348448687351E-4, 2.3752969121140142E-4, 2.398081534772182E-4, 2.3696682464454977E-4]

plt.title('N-Queens Problem Fitness')
plt.xlabel('Iterations')
plt.ylabel('Score')

plt.plot(x_vals, y_vals_rhc, label='Randomized Hill Climbing')
plt.plot(x_vals, y_vals_sa, label='Simulated Annealing')
plt.plot(x_vals, y_vals_ga, label='Genetic Algorithm')
plt.plot(x_vals, y_vals_mm, label='MIMIC')

plt.xticks(np.arange(100, 501, 100))
plt.yticks(np.arange(0.0002, 3E-4, 0.00001))
plt.legend()
plt.show()