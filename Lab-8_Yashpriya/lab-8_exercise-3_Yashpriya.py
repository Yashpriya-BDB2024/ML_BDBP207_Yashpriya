###  Implement ordinal encoding and one-hot encoding methods in Python from scratch.

def ordinal_encoding(data, categories=None):    # Input: list of categorical values, Output: encoded list
    if not data:
        raise ValueError("Input data cannot be empty.")
        # Determine unique categories (sorted alphabetically if no custom order provided)
    if categories is None:
        categories = sorted(set(data))     # removes duplicates & sorts in ascending order
    else:
        # Check for unseen categories in data
        unseen = set(data)-set(categories)
        if unseen:
            raise ValueError(f"Unseen categories in data: {unseen}")
    # enumerate(unique_categories) generates pairs of (index, category) for each category.
    # dictionary comprehension maps each category to its corresponding index
    category_to_int = {category: idx for idx, category in enumerate(categories)}
    encoded_data = [category_to_int[item] for item in data]    # each category replaced with its corresponding integer
    return category_to_int, encoded_data

def main():
    try:
        colors = ['Red', 'Green', 'Blue', 'Red', 'Yellow', 'Blue']    # Test Case 1: Default ordinal encoding (alphabetical order)
        mapping, encoded = ordinal_encoding(colors)
        print("Test 1 - Default Order:")
        print(f"Mapping: {mapping}")
        print(f"Encoded: {encoded}\n")
    except ValueError as e:
        print(f"Test 1 Error: {e}")
    except Exception as e:
        print(f"Test 1 Unexpected Error: {e}")
    try:
        size_data = ['medium', 'large', 'medium', 'small', 'large', 'small']    # Test Case 2: Custom ordinal encoding
        size_order = ['small', 'medium', 'large']    # Correct natural order
        mapping, encoded = ordinal_encoding(size_data, size_order)
        print("Test 2 - Custom Order:")
        print(f"Mapping: {mapping}")
        print(f"Encoded: {encoded}\n")
    except ValueError as e:
        print(f"Test 2 Error: {e}")
    except Exception as e:
        print(f"Test 2 Unexpected Error: {e}")
    try:
        mapping, encoded = ordinal_encoding([])     # Test Case 3: Negative case (empty input)
    except ValueError as e:
        print(f"Test 3 Error: {e}")
    except Exception as e:
        print(f"Test 3 Unexpected Error: {e}")
    try:
        mapping, encoded = ordinal_encoding(['small', 'xlarge'], ['small', 'medium', 'large'])     # Test Case 4: Negative case (unseen category)
    except ValueError as e:
        print(f"Test 4 Error: {e}")
    except Exception as e:
        print(f"Test 4 Unexpected Error: {e}")

if __name__ == "__main__":
    main()