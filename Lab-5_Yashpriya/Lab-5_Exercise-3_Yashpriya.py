# Compute the derivative of a sigmoid function and visualize it.

import numpy as np
import matplotlib.pyplot as plt
from Lab_5_Exercise_2_Yashpriya import sigmoid

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)
z = np.linspace(-10, 10, 100)
y = sigmoid_derivative(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, label="Sigmoid Derivative", color='r')
plt.title("Derivative of sigmoid function")
plt.xlabel("z")
plt.ylabel("g'(z)")
plt.show()
