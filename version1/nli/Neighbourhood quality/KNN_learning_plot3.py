import numpy as np
import argparse
import faiss
from collections import defaultdict

# from util import metrics


def read_data_func(dataset_name):
    tensor = np.loadtxt(dataset_name)
    return tensor


def normalize_embedding_func(tensor):
    # Calculate the sum of each row
    row_sq = np.square(tensor)
    row_sums = np.sum(row_sq, axis=1, keepdims=True)
    row_sums_sqrt = np.sqrt(row_sums)

    # Divide each element by its row's sum
    result_tensor = tensor / row_sums_sqrt

    # Saving the normalized embedding for future use
    # np.savetxt(f"{args.input_file[0:-4]}normalized.txt", result_tensor)
    return result_tensor


def find_distance_func(tensor1, tensor2):
    # getting length of the tensor
    n = len(tensor1)
    m = len(tensor2)
    distance_matrix = []

    # for every row we calculate the distance to every row
    for i in range(n):
        distance_row = []
        for j in range(m):
            distance_tensor = np.subtract(tensor1[i], tensor2[j])
            distance_tensor_sq = np.square(distance_tensor)
            dist_square = np.sum(distance_tensor_sq, axis=0)
            distance = np.sqrt(dist_square)
            distance_row.append(distance)
        distance_matrix.append(distance_row)

    distance_matrix = np.array(distance_matrix)
    return distance_matrix


# def sort_distance_matrix_func(distance_matrix):
#     distance_matrix.sort(axis=1)
#     return distance_matrix

# def pick_kth_value_func(distance_matrix, k):

#     # picks the kth column from 2-d matrix
#     tensor = distance_matrix[:, k]

#     return tensor

# def select_threshold_func(tensor, percent):
#     tensor = np.sort(tensor)
#     percent = percent/100
#     n = len(tensor)
#     percent = n*percent
#     percent = int(percent)
#     threshold = tensor[percent]
#     return threshold

# def get_ood_count_func(tensor, th, val):
#     # Create a boolean mask for elements greater than 'k'
#     mask = tensor > th

#     # Count the number of elements greater than 'k' in each row
#     count_per_row = np.sum(mask, axis=1)
#     count_per_row = count_per_row/val

#     return count_per_row


def read_file(filename):
    lines = []
    with open(filename, "r") as file:
        for line in file:
            key = line.rstrip()
            lines.append(key)
    return lines


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_train", type=str, required=True)
    parser.add_argument("--input_file_test", type=str, required=True)
    parser.add_argument("--test_groundtruth", type=str, required=True)
    parser.add_argument("--train_groundtruth", type=str, required=True)
    parser.add_argument("--test_pred_file", type=str, required=True)
    # parser.add_argument('--percent', type=float, required=True)

    args = parser.parse_args()

    # loading the embeddings
    tensor_train = read_data_func(args.input_file_train)
    # tensor1_val = read_data_func(args.input_file_val)
    tensor_test = read_data_func(args.input_file_test)
    train_groundtruth = read_file(args.train_groundtruth)
    test_groundtruth = read_file(args.test_groundtruth)
    print(len(tensor_train), len(train_groundtruth))
    assert len(tensor_train) == len(train_groundtruth)

    print(len(tensor_test), len(test_groundtruth))
    assert len(tensor_test) == len(test_groundtruth)

    tensor_train = normalize_embedding_func(tensor_train)
    # tensor1_val = normalize_embedding_func(tensor1_val)
    tensor_test = normalize_embedding_func(tensor_test)
    # tensor1 = np.concatenate(tensor1_train,tensor1_val, axis = 0)

    k_values = [10]

    print(f"tensor1 shape {tensor_train.shape}")
    print(f"tensor1 shape {tensor_train.shape[1]}")
    index = faiss.IndexFlatL2(tensor_train.shape[1])
    index.add(tensor_train)
    # distance, _ = index.search(tensor1_val, 1000)
    test_distance, test_indices = index.search(tensor_test, 1000)

    print(test_indices.shape)
    # print(test_indices)

    for k in k_values:
        quality = []
        print(f"k = {k}")

        for i in range(len(test_indices)):
            groundtruth_i = test_groundtruth[i]
            d = defaultdict(int)
            for j, index in enumerate(test_indices[i]):
                if j == k:
                    break
                if d.get(train_groundtruth[index]) is None:
                    d[train_groundtruth[index]] = 1
                else:
                    d[train_groundtruth[index]] += 1
            # print(d.keys())
            x = float(d[groundtruth_i]) / float(k)
            quality.append(x)

    preds = read_file(args.test_pred_file)
    d = {
        "0.0": [0, 0],
        "0.1": [0, 0],
        "0.2": [0, 0],
        "0.3": [0, 0],
        "0.4": [0, 0],
        "0.5": [0, 0],
        "0.6": [0, 0],
        "0.7": [0, 0],
        "0.8": [0, 0],
        "0.9": [0, 0],
        "1.0": [0, 0],
    }
    for i in range(len(test_indices)):
        groundtruth_i = test_groundtruth[i]
        pred_i = preds[i]
        if groundtruth_i == pred_i:
            d[str(quality[i])][0] += 1
        d[str(quality[i])][1] += 1
    print(d)


if __name__ == "__main__":
    main()
