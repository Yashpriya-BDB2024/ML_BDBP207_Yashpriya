# Implement information gain measures. The function should accept data points for parents, data points for both children and return an information gain value.

from  lab_10_exercise_1_Yashpriya import entropy_measure

def information_gain(parent_labels, child1_labels, child2_labels):
    entropy = entropy_measure([], parent_labels)    # [] - no need of data points
    total = len(parent_labels)
    expected_entropy = len(child1_labels)/total * entropy_measure([], child1_labels) + len(child2_labels)/total * entropy_measure([], child2_labels)
    info_gain = entropy - expected_entropy
    return info_gain

def main():
    parent_labels = ['Low', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High']
    child1_labels = ['Low', 'Low', 'Low', 'High']
    child2_labels = ['High', 'High', 'Low', 'High']
    print(f"Information Gain: {information_gain(parent_labels, child1_labels, child2_labels)}")

if __name__ == "__main__":
    main()