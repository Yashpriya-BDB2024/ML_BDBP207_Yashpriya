import matplotlib.pyplot as plt
import numpy as np

def linear_func(x):
    return 2*x+3

def main():
    x=np.linspace(start=-100, stop=100, num=100)
    y=linear_func(x)
    plt.title("y=2x+3 (linear function)")
    plt.xlabel("x-values")
    plt.ylabel("y-values")
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()