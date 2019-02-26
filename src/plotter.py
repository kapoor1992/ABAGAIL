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
y_vals_rhc = [2840.063103933665, 3329.900749788191, 3331.1772475818543, 3050.9139658375125, 3202.0667094459304]
y_vals_sa = [2895.301876083266, 3369.3796622160385, 3350.588852710348, 3121.9202140227217, 3263.5966497987397]
y_vals_ga = [3266.807045348459, 3678.733621872754, 3548.442739961316, 3390.367975730762, 3593.071905029364]
y_vals_mm = [3548.843111592112, 3824.399821698222, 3668.5030717605187, 3499.101182260001, 3734.913967308523]

plt.title('Knapsack Fitness')
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