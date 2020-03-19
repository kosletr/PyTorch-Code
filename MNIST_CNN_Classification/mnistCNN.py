# %%
import torch
from torch import optim, nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# %%
writer = SummaryWriter('runs/mnist_experiment_1')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,),)
])

train_data = datasets.MNIST('C:/Users/Kostas/Desktop/',train=True, transform=transform, download=True)
test_data = datasets.MNIST('C:/Users/Kostas/Desktop',train=False, transform = transform, download=True)

# %%

validation_size = 0.2

train_size = len(train_data)
indices = list(range(train_size))
np.random.shuffle(indices)

valid_split = int(np.floor(train_size*validation_size))
train_indices, valid_indices = indices[valid_split:], indices[:valid_split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

batch_size = 320

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(test_data, batch_size=batch_size)

# %%

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding = 1)
        self.conv2 = nn.Conv2d(16, 32 , (3, 3), padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p = 0.5)
        self.linear1 = nn.Linear(1568, 1024)
        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 10)
        self.soft = nn.Softmax(1)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1,1568)

        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        x = F.relu(self.linear3(x))
        x = self.soft(x)

        return x

# %%

device = torch.device('cuda')
model = Net()
model.to(device)

# %%
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

epochs = 4

for epoch in list(range(epochs)):

    train_loss = 0.0
    valid_loss = 0.0

    train_acc = 0.0
    valid_acc = 0.0
    test_acc = 0.0

    model.train()
    
    for batch_idx , (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)

        opt.zero_grad()

        output = model(data)

        _, out = torch.max(output, 1)

        loss = criterion(output, target)

        loss.backward()

        opt.step()

        train_loss += loss.item()*batch_size
        train_acc += (out == target).sum().item()

        print('Train - Epoch: {} | Train Loss: {:.4f} | {}/{} ({:.2f}%) | Train Accuracy: {:.2f}%'.format(epoch + 1, train_loss/((batch_idx+1)*batch_size), (batch_idx + 1)*batch_size, train_size, 100*(batch_idx + 1)*batch_size/train_size, 100*train_acc/((batch_idx+1)*batch_size) ))

        if batch_idx % 100 == 0:    # every 100 mini-batches...

            # ...log the running loss
            writer.add_scalar('Training Accuracy (%)',
                            100*train_acc/((batch_idx+1)*batch_size),
                            epoch * len(train_loader) + batch_idx)

    print('\n')

    model.eval()

    for batch_idx, (data, target) in enumerate(valid_loader):

        data, target = data.to(device), target.to(device)

        output = model(data)

        _, out = torch.max(output, 1)

        loss = criterion(output, target)

        valid_loss += loss.item()*batch_size
        valid_acc += (out == target).sum().item()
        
        print('Validation - Epoch: {} | Valid Loss: {:.4f} | {}/{} ({:.2f}%) | Valid Accuracy: {:.2f}%'.format(epoch + 1, valid_loss/((batch_idx+1)*batch_size), (batch_idx + 1)*batch_size, len(valid_sampler), 100*(batch_idx + 1)*batch_size/len(valid_sampler), 100*valid_acc/((batch_idx+1)*batch_size) ))

    print('\n')

    train_loss /= len(train_loader.sampler)
    valid_loss /= len(valid_loader.sampler)

    train_acc /= len(train_loader.sampler)
    valid_acc /= len(valid_loader.sampler)

    print('Epoch: {} | Train Loss: {:.4f} | Valid Loss: {:.4f} | Train Accuracy: {:.2f}% | Valid Accuracy: {:.2f}% \n'.format(epoch + 1, train_loss, valid_loss, 100*train_acc, 100*valid_acc))

# %%

for _, (data, target) in enumerate(test_loader):
    
    data, target = data.to(device), target.to(device)

    output = model(data)

    _, out = torch.max(output, 1)

    test_acc += (out == target).sum().item()
        
print('Test Accuracy: {:.2f}%'.format(100*valid_acc))

# %%
def predict(N):

    plt.imshow(np.squeeze(test_data[N][0], axis=0))

    x = [test_data[N]]
    x1 = DataLoader(x, batch_size=batch_size)

    for _, (data, _) in enumerate(x1):

        data = data.to(device)

        y = model(data)
        _, out = torch.max(y, 1)
        plt.title(test_data.classes[out.item()])

    plt.show()

# %%

N = int(input("Enter a number between zero and {}: ".format(len(test_data)-1)))

while N > -1 and N < len(test_data):
    predict(N)
    N = int(input("Enter a number between zero and {}: ".format(len(test_data)-1)))

# %%
