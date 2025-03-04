# Implement entropy measure using Python. The function should accept a set of data points and their class labels and return the entropy value.

import math
from collections import Counter

def entropy_measure(data_points, labels):
    total = len(labels)
    counts = Counter(labels)    # It counts the occurrences of each class - 'High' & 'Low' respectively
    return -sum((count/total) * math.log2(count/total) for count in counts.values())      # Entropy = - summation(prob.*log_base2*prob.)

def main():
   data_points = [[100, 79, 86], [30, 40, 70], [80, 85, 65], [90, 82, 45]]      # x1: IQ, x2: Social, x3: Verbal
   labels = ['Low', 'High', 'Low', 'High']       # Risk for autism (y)
   print(f"Entropy: {entropy_measure(data_points, labels)}")

if __name__ == "__main__":
    main()
