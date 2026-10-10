from sympy import im
import torch
from torch.utils.data import DataLoader
from _01_dataset import CustomDataset
from _01_sample_dataset import *


torch.manual_seed(123)

# Dataset instance is the input to DataLoader class
train_ds = CustomDataset(X_train, y_train)
test_ds = CustomDataset(X_test, y_test)

# create DataLoader class instance to load the data in batches
"""
#1 The CustomDataset instance created earlier serves as input to the data loader.
#2 Whether or not to shuffle the data
#3 The number of background processes
#4 It is not necessary to shuffle a test dataset.
#5 Since we have total of 5 recrods in the training dataset, 
#  the last batch will have only 1 record since the batch size is 2,
#  In practice, having a substantially smaller batch as the last batch in a training epoch 
#  can disturb the convergence during training. Hence dropping the last batch 
"""

train_loader = DataLoader(dataset=train_ds, #1
                          batch_size=2, 
                          shuffle=True, #2
                          num_workers=0, #3
                          drop_last=True #5
                          )

test_loader = DataLoader(dataset=test_ds,
                         batch_size=2,
                         shuffle=False, #4
                         num_workers=0
)

# iterate through the DataLoader to get batches of data
for idx, (X, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", X, y) # observe that only batch 1 and batch 2 is printed in the console

