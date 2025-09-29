import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

theoretical = {2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
               7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36}
theo_x = np.arange(2, 13)
theo_y = np.array([theoretical[k] for k in theo_x])

for n in n_values:
    d1 = np.random.randint(1, 7, size=n)
    d2 = np.random.randint(1, 7, size=n)
    s = d1 + d2
    h, h2 = np.histogram(s, range(2, 14))
    p = h / n

    plt.bar(h2[:-1], p, width=0.9, align='center', label=f'Empirical (n={n})')
    plt.title(f'Sum of Two Dice — Histogram (n={n})')
    plt.xlabel('Sum')
    plt.ylabel('Probability')
    plt.xticks(np.arange(2, 13))
    plt.ylim(0, 0.18)
    plt.legend()
    plt.show()



print("Exercise 1 finished. Histograms saved as ex1_hist_n*.png")
