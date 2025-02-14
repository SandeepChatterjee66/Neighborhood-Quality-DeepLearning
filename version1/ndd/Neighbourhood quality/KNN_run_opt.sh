#!/bin/bash

# # Convert text files to numeric labels and binary format
# python3 -c "import numpy as np; from sklearn.preprocessing import LabelEncoder;
# labels = np.loadtxt('/home/gpuuser1/gpuuser1_a/sandeep/sandeep/ndd/FFNN/FFNN_NDD/qqp_groundtruth.txt', dtype=str);
# encoder = LabelEncoder(); encoded_labels = encoder.fit_transform(labels);
# np.save('labels.npy', encoded_labels)"

# Main execution
export CUDA_VISIBLE_DEVICES=0

# Define the values of i to loop over
i_values=$(seq 1 15)  # 1 to 15

# Loop through each value of i
for i in $i_values; do
    echo "Processing model $i..."
    
    # Convert embeddings to binary format
    python3 -c "import numpy as np; np.save('embeddings.npy', np.loadtxt(f'/home/gpuuser1/gpuuser1_a/sandeep/sandeep/ndd/FFNN/Embeddings/Train/ffnn_model_${i}_ndd.txt'))"
    
    # Run KNN evaluation
    python3 KNN_learning-opt3.py \
        --input_file_train embeddings.npy \
        --train_groundtruth labels.npy \
        --test_model_name "model${i}" \
        --results "./results/FFNN-NDD/" \
        --p 5
    
    # Clean up temporary files (optional)
    if [ -f "embeddings.npy" ]; then
        rm embeddings.npy
    else
        echo "embeddings.npy not found for model $i. Skipping cleanup."
    fi
    
    echo "Completed processing for model $i"
    echo "------------------------------"
done

echo "All models processed!"