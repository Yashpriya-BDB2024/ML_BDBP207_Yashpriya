### Implement information gain measures. The function should accept data points for parents, data points for both children and return an information gain value.

from  lab_10_exercise_1_Yashpriya import entropy_measure

def information_gain(parent_labels, child1_labels, child2_labels):
    # Calculate entropy of the parent node (before the split)
    entropy = entropy_measure([], parent_labels)    # [] - No need to pass data points, only labels matter
    total = len(parent_labels)      # Total no. of data points in the parent node
    # Calculate weighted entropy of child nodes after the split (i.e., weight * entropy of a child)
    expected_entropy = len(child1_labels)/total * entropy_measure([], child1_labels) + len(child2_labels)/total * entropy_measure([], child2_labels)
    # Compute Information Gain: reduction in entropy after splitting
    info_gain = entropy - expected_entropy
    return info_gain

def main():
    parent_labels = ['Low', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High']    # Class labels for the parent node (before the split)
    child1_labels = ['Low', 'Low', 'Low', 'High']      # First subset after split
    child2_labels = ['High', 'High', 'Low', 'High']    # Second subset after split
    print(f"Information Gain: {information_gain(parent_labels, child1_labels, child2_labels)}")

if __name__ == "__main__":
    main()
