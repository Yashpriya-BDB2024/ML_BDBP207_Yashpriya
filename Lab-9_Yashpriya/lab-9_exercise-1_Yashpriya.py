### Write a program to partition a dataset (simulated data for regression) into two parts, based on a feature (BP) and for a threshold, t = 80.
### Generate additional two partitioned datasets based on different threshold values of t = [78, 82].

import pandas as pd

def partition_space(data, thresholds):
    for t in thresholds:     # Iterate through each threshold value
        lower_partition = data[data["BP"] <= t]      # Create lower partition where BP is less than or equal to the threshold
        upper_partition = data[data["BP"] > t]       # Create upper partition where BP is greater than the threshold
        print(f"\nFor threshold t = {t}:")
        print(f"Lower Partition (BP ≤ {t}):")
        print(lower_partition)
        print(f"Upper Partition (BP > {t}):")
        print(upper_partition)

def main():
    data =  pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")     # Read the dataset from a CSV file
    thresholds = [80, 78, 82]        # Define different threshold values for partitioning
    partition_space(data, thresholds)      # Call the partition function to divide data based on threshold values

if __name__ == "__main__":
    main()
