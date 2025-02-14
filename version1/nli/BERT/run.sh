# !/bin/bash
echo $(pwd)

python train.py --train_file_path ../datasets/snli_1.0/snli_1.0_train.txt \
                --dev_file_path ../datasets/snli_1.0/snli_1.0_dev.txt \
                --output_model_directory Models 

# python3 test.py --input_model_path MNLI_Model/BestModel \
                # --snli_test_path snli_1.0
# --dataset_name ../../datasets/snli_1.0