# %%

import torch
from torch import optim, nn, functional as F
from torch.utils.data import dataloader
from torchvision import transforms, datasets
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# %%

transforms = {
'train': transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'valid': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
}

# %%
train_dir  = 'datadir//train'
valid_dir = 'datadir//valid'
test_dir = 'datadir//test'

dataset = {
    'train': datasets.ImageFolder(train_dir, transform=transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=transforms['test'])
}

batch_size = 128

data_loader = {
    'train': dataloader.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True),
    'valid': dataloader.DataLoader(dataset['valid'], batch_size=batch_size, shuffle=True),
    'test': dataloader.DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)  
}

"""

train_iter = iter(data_loader['train'])
features, labels = next(train_iter)
features.shape, labels.shape

"""

# %%

# import famous models
from torchvision import models

# import pretrained model
model = models.resnet34(pretrained=True)

# Show layers
print(model)

# %%
# Add extra layers in the end
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(256, 102),
    nn.LogSoftmax(dim=1)
)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last parameters
for param in model.fc.parameters():
    param.requires_grad = True

print(model)

# %%
"""

total_params = sum(params.numel() for params in model.parameters())
print(f'Total Parameters: {total_params:,}')

trainable_params = sum(params.numel() for params in model.parameters() if params.requires_grad)
print(f'Parameters to be trained: {trainable_params:,}')

"""
# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters())
    
# %%

num_epochs = 20
epochs_stop = 3

epochs_no_improve = 0
min_valid_loss = np.inf

writer = SummaryWriter()

for epoch in list(range(num_epochs)):

    valid_loss = 0.0

    for batch_idx, (data, target) in enumerate(data_loader['train']):
    
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        print('Train - Epoch: {} | {}/{} ({:.2f}%)'.format(epoch + 1, batch_idx*batch_size, len(data_loader['train'])*batch_size, 100*batch_idx/len(data_loader['train']) ))

    print('\n')
        
    for batch_idx, (data, target) in enumerate(data_loader['valid']):

        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = criterion(output, target)

        valid_loss += loss
    
    valid_loss /= len(data_loader['valid'])

    writer.add_scalar('Validation Loss', valid_loss, epoch + 1)
    print(f'Validation Loss: {valid_loss:.4f}\n')

    # Early Stopping
    if valid_loss < min_valid_loss:

        torch.save(model, 'model.pth')
        epochs_no_improve = 0
        valid_loss = min_valid_loss

    else:

        epochs_no_improve += 1

        if epochs_no_improve == epochs_stop:

            print('Early Stopping')

            torch.save(model, 'model.pth')

            break

writer.close()

# %%

test_accuracy = 0.0

for _, (data, target) in enumerate(data_loader['test']):
    
    data, target = data.to(device), target.to(device)

    output = model(data)

    _, out = torch.max(output, 1)

    test_accuracy += (out == target).sum().item()

test_accuracy /= batch_size*len(data_loader['test'])

print('Test Accuracy: {:.2f}%'.format(100*test_accuracy))

# %%

"""

model = torch.load('model1.pth')
model.eval()

"""
# %%
from matplotlib import pyplot as plt

def predict(N):
    
    t = dataset['test'][N]

    plt.imshow(t[0].permute([1, 2, 0]))

    dl = dataloader.DataLoader(torch.unsqueeze(t[0], 0), batch_size=batch_size)

    for (_, data) in enumerate(dl):

        data = data.to(device)

        y = model(data)

        _, out = torch.max(y, 1)
    
        plt.title('Prediction: {} | Real: {}'.format(
            dataset['test'].classes[out.item()], 
            dataset['test'].classes[t[1]] ))
        
    plt.show()

# %%
N = int(input("Enter a number between 0 and {}: ".format(len(dataset['test'])-1)))

while N > -1 and N < len(dataset['test']):
    predict(N)
    N = int(input("Enter a number between zero and {}: ".format(len(dataset['test'])-1)))

# %%