import matplotlib.pyplot as plt
import numpy as np

mean=0
sigma=15
x=np.linspace(start=-100, stop=100, num=100)
y= (1 / sigma * np.sqrt(2 * 3.14)) * 2.71 ** ((-0.5) * ((x - mean) / sigma) ** 2)
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.title("Gaussian PDF")
plt.plot(x, y)
plt.show()