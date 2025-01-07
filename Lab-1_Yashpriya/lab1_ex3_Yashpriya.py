import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return 2*(x**2)+(3*x)+4

def main():
    x=np.linspace(start=-10, stop=10, num=100)
    y=func(x)
    plt.xlabel("x-values")
    plt.ylabel("y-values")
    plt.title("y=2x^2+3x+4")
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()