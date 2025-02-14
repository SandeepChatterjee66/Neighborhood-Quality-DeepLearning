# !/bin/bash
echo $(pwd)

python3 train.py --dataset_name qqp --output_model_directory Models --output_tokenizer_directory Tokenizer

# python3 test.py --input_model_path QQP_MODEL/BestModel \
                --paws_file_path ./PAWS/test.tsv
