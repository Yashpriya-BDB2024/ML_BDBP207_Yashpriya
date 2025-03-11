### Implement entropy measure using Python. The function should accept a set of data points and their class labels and return the entropy value.

import math
from collections import Counter

def entropy_measure(data_points, labels):
    total = len(labels)     # Total number of data points
    counts = Counter(labels)      # Count occurrences of each class label (e.g., 'High' and 'Low')
    # Compute entropy using the formula: 
    # Entropy = - summation(P * log2(P)), where P is the probability of each class
    return -sum((count/total) * math.log2(count/total) for count in counts.values())      

def main():
   data_points = [[100, 79, 86], [30, 40, 70], [80, 85, 65], [90, 82, 45]]      # x1: IQ (e.g.: 100), x2: Social (e.g.: 79), x3: Verbal (e.g.: 86)
   labels = ['Low', 'High', 'Low', 'High']       # Corresponding risk levels (y) for autism 
   print(f"Entropy: {entropy_measure(data_points, labels)}")

if __name__ == "__main__":
    main()
