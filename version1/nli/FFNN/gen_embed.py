# ffnn nli embed
from utils import *

EMBED_SIZE = 30


train_set_path = config['train_set_path']
validation_set_path = config['val_set_path']
test_set_path = config['test_set_path']

BATCH_SIZE = 1024

# #train_loader, l1 = load_data(train_set_path)
# test_loader, l3 = load_data(test_set_path)

# #gen_embeddings(train_loader=train_loader, num_epochs=30)
# gen_test_preds(test_loader=test_loader)

#train_loader, l1 = load_data(train_set_path)
test_loader, l3 = load_data(train_set_path)

#gen_embeddings(train_loader=train_loader, num_epochs=30)
gen_test_preds(test_loader=test_loader)




  

