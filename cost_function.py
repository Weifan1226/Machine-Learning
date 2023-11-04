import numpy as np
# matplotlib widget
import matplotlib.pyplot as plt
# from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
# plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])  # (size in 1000 square feet)
y_train = np.array([300.0, 500.0])  # (price in 1000s of dollars)


def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost