# ------------------------------------------------------------------------------
# Creat on 2022.01.07
# Written by Liu Tao (LiuTaobbu@163.com)
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.figure(figsize=(14, 7))

x = np.linspace(0, 1, 100)
y = stats.beta.pdf(x, a=1.0, b=1.0)

plt.plot(x, y, color='b', label='PDF')
plt.fill_between(x, y, color='b', alpha=0.25)
plt.legend()
plt.show()

