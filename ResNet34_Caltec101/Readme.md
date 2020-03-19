# Transfer Learning using PyTorch - Caltech101 Dataset + ResNet34 Pretrained Network

![Valid_Loss](https://github.com/kosletr/PyTorch-Code/blob/master/ResNet34_Caltec101/valid_loss.png)

## Setup Instructions

 - Download the Caltech101 Dataset using the link provided.

 - Split the Dataset to train, validation, test sets using the splitDatasets.py python script.

 - Run caltech101.py PyTorch classifier to train, evaluate and then test the dataset, using ResNet34 pretrained network (transfer learning).

## Results

After 20 epochs of training on a GPU - Nvidia GeForce RTX 2060 6GB:

Validation Loss: 0.4758
Test Accuracy: 83.84%

## Information

- Dataset Split:
  - Train: 0.50
  - Validation: 0.25
  - Test: 0.25

- Loss Function: Negative Log Likelihood Loss
- Optimizer: Adam
- Batch Size: 128
- Extras: Early Stopping

- Metrics: Test Accuracy, Validation Loss

- Transformations (such as Resize, Normalization etc.) are applied to each set (train, validation, test).

ResNet34's Final Fully Connected Layer is replaced by:

    Fully Connected Layer 1 | input_size: 512 - output_size: 256
    Activation Function   1 | ReLU
    Dropout                 | p = 0.4
    Fully Connected Layer 2 | input_size: 256 - output_size: 102
    Activation Function   2 | Log Softmax

All the weights of the Layers apart from those listed above, are being frozen during the training process.
