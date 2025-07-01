import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision import models

#Creating basic tensors
'''x =  torch.tensor([1,2,3])
print(x.dtype)'''

'''y = torch.zeros(2,3) #matrix of zeroes with specified dimension
print(y)'''

'''z = torch.rand((2,2)) #matrix of random numbers with specified dimension
print(z)'''

#Operations on tensors
'''a = torch.tensor([1,2,3])
b = torch.tensor([5,6,7])'''

'''print(a+b) # Addition'''
'''print(a*b) # Element-wise multiplication'''

#Automatic differentiation(autograd)
'''a = torch.tensor(5.0, requires_grad = True)
y = a**5
y.backward()
print(a.grad)'''

#Neural Networks (torch.nn)
'''class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2,1) # Fully connected layer with 2 inputs and 1 output
    def forward(self, x):
        return self.fc(x)'''
    

#Training a model
'''model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)'''

'''x = torch.tensor([[1.0,2.0]], requires_grad=True)'''
'''y = torch.tensor([[1.0]])+

#Training loop
for epoch in range(100):
    pred = model(x)
    loss = criterion(pred,y)
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update the weights

    print(f'Epoch {epoch}, Loss: {loss.item()}')'''

#Datasets and DataLoaders
'''transform = transforms.ToTensor()
train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
train_loader = DataLoader(train_data, batch_size =32, shuffle = True)'''

#Computer vision model(Torchvision)
'''model = models.resnet18(weights=True)  # Load a pre-trained ResNet-18 model'''

#Saving nd Loading models
'''x = torch.save(model.state_dict(), 'model.pth') # Save the model state
print(x)'''

'''d = model.load_state_dict(torch.load('model.pth'))  # Load the model state
print(d)'''

#Usig GPU(CUDA)
'''device  = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model.to(device)  # Move the model to GPU
x = x.to(device)'''

#Creation of custom datasets
'''class Cdataset(Dataset):
    def __init__(self):
        self.data = [i for i in range(100)]
    def __index__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)'''
