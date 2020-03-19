# CNN Classification using PyTorch - MNIST Dataset

A Simple CNN classifier using PyTorch.

## Information

- Validation Size: 20% of the Training Set
- Batch Size: 320
- Loss Function: Cross Entropy Loss
- Optimizer: Adam (learning rate: 1e-3)
- Metrics: train loss, valid loss, train acc, valid acc, test acc

Convolutional Neural Network:

	Convolutional Layer   1 | input_size: 1x28x28, output_size: 16x28x28, kernel: (3, 3), padding : True
	Activation Function   1 | ReLU
	Max Pooling Layer     1 | input_size: 16x28x28, output_size: 16x14x14
	
	Convolutional Layer   2 | input_size: 16x14x14, output_size: 32x14x14, kernel: (3, 3), padding : True
	Activation Function   2 | ReLU
	Max Pooling Layer     2 | input_size: 32x14x14, output_size: 32x7x7
	
	Flatten Layer			| input_size: 32x7x7, output_size: 1568
	
	Fully Connected Layer 3 | input_size: 1568 - output_size: 1024
    Activation Function   3 | ReLU
    Dropout               3 | p = 0.5
	
    Fully Connected Layer 4 | input_size: 1024 - output_size: 128
    Activation Function   4 | ReLU
    Dropout               4 | p = 0.5
	
	Fully Connected Layer 5 | input_size: 128 - output_size: 10
    Activation Function   5 | Softmax
