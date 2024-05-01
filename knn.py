"""
KNN - Implementation 
By Sagana Ondande
Github: https://github.com/sondande
"""

import argparse
import csv
import heapq
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def hamming_distance(x: list, y: list) -> float:
    return sum(x_i != y_i for x_i, y_i in zip(x[1:], y[1:]))


def euclidean_distance(x: list, y: list) -> float:
    return sum((x_i - y_i) ** 2 for x_i, y_i in zip(x[1:], y[1:]))

def get_k_nearest(k_value: int, k_instances: dict) -> list:
    # Create heap queue to get the k smallest elements
    heap = []
    for distance, labels in k_instances.items():
        heapq.heappush(heap, (distance, labels))
    
    # Get a list of the kValue amount of nearest neighbors
    nearest = []
    while len(nearest) < k_value and heap:
        _, labels = heapq.heappop(heap)
        nearest.extend(labels[: k_value - len(nearest)])
    return nearest


def k_nearest_neighbors(k_value: int, training_set: list, testing_set, distance_choice) -> list:
    print("Running KNN Algorithm\n")
    for x_test_instance in testing_set:
        # Use defaultdict to provide default values for missing keys
        k_instances = defaultdict(list)
        # Calculate distance between each instance in x_train against x_test instance
        for x_train_instance in training_set:
            if distance_choice == "H":
                dist = hamming_distance(x_test_instance, x_train_instance)
            else:
                dist = euclidean_distance(x_test_instance, x_train_instance)
            # Create list of k_instances with key is distance and value is list of labels with that distance
            k_instances[dist].append(x_train_instance[0])

        # Retrieve the k nearest instances
        k_results = get_k_nearest(k_value=k_value, k_instances=k_instances)

        # Find the most frequent label
        predicted_label = max(set(k_results), key=k_results.count)
        
        # Add predicted value to x_test_instance 
        x_test_instance.append(predicted_label)

    print("Finished Running KNN Algorithm\n")
    print("-----------------")
    return testing_set


def confusion_matrix(
    testing_set: list, dsPath: str, k_value: int, random_seed: int, possible_labels: list
) -> list:
    # Construct the output file name based on given parameters
    output_file_name = f"results-{os.path.basename(dsPath)}-{k_value}-{random_seed}.csv"
    output_file_path = os.path.join("knn-results", output_file_name)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Create a dictionary to hold label results
    label_results_dict = {}

    # Create dictionary of test_set results for confusion matrix
    for y_test_instance in testing_set:
        true_label = y_test_instance[0]
        predicted_label = y_test_instance[-1]
        if true_label not in label_results_dict:
            label_results_dict[true_label] = []
        label_results_dict[true_label].append(predicted_label)

    # Generate the confusion matrix
    confusion_matrix_result = []

    # Write Confusion Matrix to output file
    with open(output_file_path, "w", newline="") as outputFile:
        writer = csv.writer(outputFile)

        # Write the header row with possible labels
        writer.writerow(possible_labels)

        # Create rows for the confusion matrix
        for true_label in possible_labels:
            row_counts = Counter(label_results_dict.get(true_label, []))
            row = [row_counts.get(lbl, 0) for lbl in possible_labels]
            # Append the true label at the end of its row
            row.append(true_label)
            confusion_matrix_result.append(row)
            writer.writerow(row)

    print("Confusion Matrix")
    print("-----------------")
    # Print contents from output file
    with open(output_file_path, 'r') as outputFile:
        # Read and print each line from the file
        for line in outputFile:
            print(line.strip())
    print("-----------------")
    return confusion_matrix_result


def calculate_stats(confusion_matrix: list) -> None:
    print("Stats From Run")
    print("--------------\n")
    # Convert input to a numpy array for easier manipulation
    matrix = np.array(confusion_matrix)

    # Extract numerical part of the confusion matrix and labels
    numeric_matrix = matrix[:, :-1].astype(int)
    labels = matrix[:, -1]

    # Calculate accuracy
    sum_diagonal = np.trace(numeric_matrix)
    sum_of_cells = np.sum(numeric_matrix)
    accuracy = sum_diagonal / sum_of_cells
    print(f"Accuracy: {accuracy}")

    # Calculate recall for each label
    for index in range(len(numeric_matrix)):
        true_positive = numeric_matrix[index][index]
        false_negative = np.sum(numeric_matrix[index]) - true_positive
        if (true_positive + false_negative) != 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        print(f"Recall for label '{labels[index]}': {recall}")

    testingSet = sum_of_cells 

    # Calculate the confidence interval for the accuracy
    se = np.sqrt((accuracy * (1 - accuracy)) / testingSet)
    confidence_interval_positive = accuracy + 1.96 * se
    confidence_interval_negative = accuracy - 1.96 * se
    print(
        f"Confidence Interval: [{confidence_interval_negative}, {confidence_interval_positive}]"
    )


def identify_and_encode(dataset: pd.DataFrame) -> pd.DataFrame:
    # Identify categorical columns: Assume non-numeric columns are categorical
    categorical_columns = dataset.select_dtypes(include=["object"]).columns.tolist()

    # Encode categorical data
    for column in categorical_columns:
        encoder = LabelEncoder()
        dataset.loc[:, column] = encoder.fit_transform(dataset.loc[:, column])

    return dataset


def main():
    start = time.time()

    parser = argparse.ArgumentParser(description="Run K-Nearest Neighbors Algorithm.")
    parser.add_argument("-f", "--filename", type=str, required=True, help="File Path to the dataset file.")
    parser.add_argument("-d","--distance_function",type=str,required=True,choices=["H", "E"],help="Distance function: 'H' for Hamming, 'E' for Euclidean.",)
    parser.add_argument("-k", "--k_value", type=int, required=True, help="The number of nearest neighbors.")
    parser.add_argument("-t","--train_percent",type=float,required=True,help="The percentage of the dataset to use for training as a decimal.",)
    parser.add_argument("-r", "--random_seed", default=1, type=int, help="Random seed for shuffling the data.")

    args = parser.parse_args()
        
    # Use absolute path for dataset
    ds_path = os.path.abspath(args.filename)
    dist_choice = args.distance_function.upper()
    k_value = args.k_value
    train_set_percent = args.train_percent
    random_seed = args.random_seed
        
    # Validate training set percentage
    if not (0 < train_set_percent < 1):
        print("Percentage for training set must be between 0 and 1.")
        exit()
    
    # Load dataset using pandas
    data = pd.read_csv(ds_path)
    data = shuffle(data, random_state=random_seed)

    # Identify if data provided contains categorical data
    data.iloc[:, 1:] = identify_and_encode(data.iloc[:, 1:])

    # Convert all columns to float type if distance choice is 'E'
    if dist_choice == "E":
        for col in data.columns[1:]:
            data[col] = data[col].astype(float)

    # Get unique labels
    possible_labels = sorted(data.iloc[:, 0].unique(), reverse=True)

    training_set, testing_set = train_test_split(
        data, train_size=train_set_percent, random_state=random_seed
    )

    # Convert dataframes to lists
    training_set = training_set.values.tolist()
    testing_set = testing_set.values.tolist()

    # Run KNN Algorithm 
    test_set_results = k_nearest_neighbors(k_value, training_set, testing_set, dist_choice)
    
    # Produce Confusion Matrix from results
    result_c = confusion_matrix(
        test_set_results, ds_path, k_value, random_seed, possible_labels
    )
    calculate_stats(result_c)
    end = time.time()
    print(f"Time Program Ran: {end - start}")


if __name__ == "__main__":
    main()
