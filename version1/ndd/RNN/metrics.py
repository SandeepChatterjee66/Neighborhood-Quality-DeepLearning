def extract_metrics(metrics_file, epochs):
    """Extract validation and test accuracies from metrics.txt."""
    val_accuracies = []
    test_accuracies = []
    
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    for epoch in epochs:
        val_acc = None
        test_acc = None
        
        for line in lines:
            if f"test accuracy at epoch {epoch} is" in line:
                test_acc = float(line.strip().split()[-1]) / 100
            if f"val accuracy at epoch {epoch} is" in line:
                val_acc = float(line.strip().split()[-1]) / 100
        
        if val_acc is None or test_acc is None:
            raise ValueError(f"Could not find accuracies for epoch {epoch} in {metrics_file}")
        
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
    
    return val_accuracies, test_accuracies

file = '/home/gpuuser1/gpuuser1_a/sandeep/sandeep/ndd/RNN/output-train-23jan-1pm.txt'
epochs = list(range(1,13))+[15,18,21,24,27,30]
print(extract_metrics(file, epochs))