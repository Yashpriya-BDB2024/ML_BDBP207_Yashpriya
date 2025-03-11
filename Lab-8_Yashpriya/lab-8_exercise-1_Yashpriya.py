### Implement L2-norm and L1-norm from scratch.

# L2 Regularization / Ridge regression -
def l2_norm(thetas):
    return sum(i**2 for i in thetas)    # Squaring each value in vector 'thetas' & then, summing them up

# L1 Regularization / LASSO -
def l1_norm(thetas):
    return sum(abs(i) for i in thetas)    # Taking the absolute value of each element of vector 'thetas' and summing them up

def main():
    thetas = [-1.8, 2.8, 2.1, 3.1, 1.9, 4]     # model parameters
    print(f"l2 norm for {thetas} is: {l2_norm(thetas)}")
    print(f"l1 norm for {thetas} is: {l1_norm(thetas)}")

if __name__ == "__main__":
    main()
