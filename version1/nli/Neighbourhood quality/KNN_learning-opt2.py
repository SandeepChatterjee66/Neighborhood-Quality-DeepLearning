import argparse
import time
import os
import faiss
import numpy as np
from sklearn.preprocessing import LabelEncoder

def read_data(filename):
    """Load embeddings in binary .npy format"""
    return np.load(filename).astype(np.float32)

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
    parser.add_argument("--results", required=True, help="where to save the results")
    parser.add_argument("--p", required=True, help="percentage of embedding to use")
    args = parser.parse_args()

    # Load data with memory mapping
    train_embeddings = read_data(args.input_file_train)
    train_labels, encoder = read_labels(args.train_groundtruth)
    
    result_dir = args.results
    os.makedirs(result_dir, exist_ok=True)

    # Global percentage of embeddings to use (e.g., 10%)
    p = float(args.p)  # Set p to the desired percentage

    # Calculate number of embeddings to use
    num_embeddings = len(train_embeddings)
    num_to_use = int((p / 100) * num_embeddings)

    # Randomly sample p% of the embeddings and labels
    indices = np.random.choice(num_embeddings, num_to_use, replace=False)
    train_embeddings = train_embeddings[indices]
    train_labels = train_labels[indices]

    # Normalize directly on GPU
    faiss.normalize_L2(train_embeddings)
    
    # GPU configuration
    res = faiss.StandardGpuResources()
    res.setTempMemory(4*1024*1024*1024)  # 4GB temp memory
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
    k_values = [5, 10, 20, 50, 75, 100]
    max_k = max(k_values) + 1  # +1 to exclude self
    
    # Process in batches
    all_indices = []
    for i in range(0, len(train_embeddings), batch_size):
        batch = train_embeddings[i:i+batch_size]
        _, indices = gpu_index.search(batch, max_k)
        all_indices.append(indices[:, 1:])  # Exclude self
    
    neighbor_indices = np.vstack(all_indices).astype(np.int32)
    neighbor_labels = train_labels[neighbor_indices]

    # Vectorized accuracy calculation
    accuracies = {}
    for k in k_values:
        top_k = neighbor_labels[:, :k]
        # Custom mode calculation for efficiency
        pred_labels = np.array([np.bincount(row).argmax() for row in top_k])
        accuracies[k] = np.mean(pred_labels == train_labels)
    
    with open(f"{result_dir}knn_accuracies_{args.test_model_name}.txt", "w") as f:
        for k, acc in accuracies.items():
            f.write(f"k={k}\tAccuracy: {acc:.4f}\n")
    
    print(f"Execution completed in {time.time()-start_time:.2f}s")


if __name__ == "__main__":
    main()