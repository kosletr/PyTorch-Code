"""
    Split dataset to train, validation, test sets.
    Folders must be named by the class name.
"""

# %%

import os


data_dir = 'data'

train_split = 0.50
valid_split = 0.25
# test_split is automatically set to  1 - (train + valid)

# %%
dirs = []

for c in os.listdir(data_dir):
    dirs.append(c)

# %%
for c in dirs:

    folder = os.listdir(data_dir+'//'+c)
    dir_size = len(folder)
    
    train_size = int(train_split*dir_size)
    valid_size = int(valid_split*dir_size)
    test_size = dir_size - train_size - valid_size
    
    for idx, im in enumerate(folder):
    
        if idx < train_size:
            os.renames(data_dir+'//'+c+'//'+im, 'train//'+c+'//'+im)
        elif idx < train_size + valid_size:
            os.renames(data_dir+'//'+c+'//'+im, 'valid//'+c+'//'+im)
        else:
            os.renames(data_dir+'//'+c+'//'+im, 'test//'+c+'//'+im)

# %%
