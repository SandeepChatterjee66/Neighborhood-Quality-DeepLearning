import argparse
import time
import os
import faiss
import numpy as np
from sklearn.preprocessing import LabelEncoder

def read_data(filename):
    """Load embeddings in binary .npy format"""
    embeddings = np.load(filename).astype(np.float32)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings

def read_labels(filename):
    """Load and encode labels"""
    labels = np.load(filename, mmap_mode='r')
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder

def main():
    start_time = time.time()
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Ultra-Optimized GPU K-NN Evaluation")
    parser.add_argument("--input_file_train", required=True, help="Path to .npy training embeddings")
    parser.add_argument("--train_groundtruth", required=True, help="Path to .npy training labels")
    parser.add_argument("--test_model_name", required=True, help="Model name for results")
    parser.add_argument("--results", required=True, help="Where to save the results")
    parser.add_argument("--p", required=True, help="Base percentage of embedding to use", type=float)
    args = parser.parse_args()

    # Load data with memory mapping
    train_embeddings = read_data(args.input_file_train)
    train_labels, encoder = read_labels(args.train_groundtruth)
    
    result_dir = args.results
    os.makedirs(result_dir, exist_ok=True)

    # Normalize directly on GPU
    faiss.normalize_L2(train_embeddings)
    
    # GPU configuration
    res = faiss.StandardGpuResources()
    res.setTempMemory(4 * 1024 * 1024 * 1024)  # 4GB temp memory
    dimension = train_embeddings.shape[1]
    
    # Create optimized IVF index
    nlist = 100  # Number of Voronoi cells
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Train and add vectors
    gpu_index.train(train_embeddings)
    gpu_index.add(train_embeddings)
    
    # Batch processing parameters
    batch_size = 100000
    k_values = [5, 7, 10, 20, 50, 75]
    max_k = max(k_values)  # Compute for the largest k and reuse for smaller k values
    
    # Process in batches
    all_indices = []
    for i in range(0, len(train_embeddings), batch_size):
        batch = train_embeddings[i:i + batch_size]
        _, indices = gpu_index.search(batch, max_k)
        all_indices.append(indices)
    
    neighbor_indices = np.vstack(all_indices).astype(np.int32)

    # Ensure indices are within bounds
    neighbor_indices = np.clip(neighbor_indices, 0, len(train_labels) - 1)
    neighbor_labels = train_labels[neighbor_indices]

    # Define percentages to evaluate
    base_p = args.p
    percentages = [base_p, 2 * base_p, 3 * base_p, 5 * base_p, 6 * base_p, 10 * base_p, 100.0]

    # Hierarchical Sampling: Sample 8p first, then derive 6p, 4p, and 2p
    num_samples = len(train_embeddings)
    
    indices_8p = np.random.choice(num_samples, int((8 * base_p / 100) * num_samples), replace=False)
    indices_6p = np.random.choice(indices_8p, int((6 * base_p / 100) * num_samples), replace=False)
    indices_4p = np.random.choice(indices_6p, int((4 * base_p / 100) * num_samples), replace=False)
    indices_2p = np.random.choice(indices_4p, int((2 * base_p / 100) * num_samples), replace=False)

    # Map percentages to the corresponding indices
    percentage_to_indices = {
        base_p: np.random.choice(num_samples, int((base_p / 100) * num_samples), replace=False),
        2 * base_p: np.random.choice(num_samples, int((2 * base_p / 100) * num_samples), replace=False),
        4 * base_p: indices_4p,
        6 * base_p: indices_6p,
        8 * base_p: indices_8p,
        10 * base_p: np.random.choice(num_samples, int((10 * base_p / 100) * num_samples), replace=False),
        100.0: np.arange(num_samples)  # Full dataset
    }

    # Evaluate for each percentage
    results = {}
    for p, subsample_indices in percentage_to_indices.items():
        subsample_labels = train_labels[subsample_indices]
        subsample_neighbor_labels = neighbor_labels[subsample_indices]
        
        # Vectorized accuracy calculation for all k values
        accuracies = {}
        for k in k_values:
            top_k = subsample_neighbor_labels[:, :k]
            pred_labels = np.array([np.bincount(row).argmax() for row in top_k])
            accuracies[k] = np.mean(pred_labels == subsample_labels)
        
        results[p] = accuracies
    
    # Save results
    with open(f"{result_dir}/knn_accuracies_{args.test_model_name}.txt", "w") as f:
        for p, accuracies in results.items():
            f.write(f"Percentage: {p}%\n")
            for k, acc in accuracies.items():
                f.write(f"k={k}\tAccuracy: {acc:.4f}\n")
            f.write("\n")
        print(f"Execution completed in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()