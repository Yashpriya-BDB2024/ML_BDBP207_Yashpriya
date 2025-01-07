import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(start=-100, stop=100, num=100)
y=x**2
deri_y=2*x
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.plot(x, y, label="y=x^2")
plt.plot(x, deri_y, label="y=2x (derivative)")
plt.legend()
plt.show()

