
                                
#!/bin/bash
python3 KNN_learning_plot3.py --input_file_train ./RNN_embeds_KNN_learning/rnn_model_5_embeddings_nli.txt \
                              --input_file_test ./RNN_embeds_KNN_learning/rnn_model_5_test_embeddings_nli.txt \
                              --train_groundtruth ./RNN_embeds_KNN_learning/SNLI_train_groundtruth.txt \
                              --test_groundtruth ./RNN_embeds_KNN_learning/SNLI_test_groundtruth.txt \
                              --test_pred_file ./RNN_embeds_KNN_learning/snli_test_preds.txt \
                              --percent 95





