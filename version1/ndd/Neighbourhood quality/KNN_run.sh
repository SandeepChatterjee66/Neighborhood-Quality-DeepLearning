# python3 KNN_learning.py --input_file_train ./Embeddings_transformer/SNLI_train_pre_train_embeddings.txt \
#                         --train_groundtruth ./RNN_embeds_KNN_learning/SNLI_train_groundtruth.txt \
#                         --test_dataset_name SNLI 





python3 KNN_learning.py --input_file_train ../FFNN/Embeddings/ffnn_nli_train_embeddings_model_1.txt \
                        --train_groundtruth ../../nli/datasets/snli_1.0/0_train_groundtruth.txt \
                        --test_dataset_name SNLI


