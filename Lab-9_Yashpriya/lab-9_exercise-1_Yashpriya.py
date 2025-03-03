# Write a program to partition a dataset (simulated data for regression) into two parts, based on a feature (BP) and for a threshold, t = 80.
# Generate additional two partitioned datasets based on different threshold values of t = [78, 82].

import pandas as pd

def partition_space(data, thresholds):
    for t in thresholds:
        lower_partition = data[data["BP"] <= t]
        upper_partition = data[data["BP"] > t]
        print(f"\nFor threshold t = {t}:")
        print(f"Lower Partition (BP ≤ {t}):")
        print(lower_partition)
        print(f"Upper Partition (BP > {t}):")
        print(upper_partition)

def main():
    data =  pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    thresholds = [80, 78, 82]
    partition_space(data, thresholds)

if __name__ == "__main__":
    main()
