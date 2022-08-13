import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

##Create the fully connected neural network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):#input size here will be 784 since mnist has 28x28 images
        super(NN, self).__init__()#Super calls the initialisation method of the parent class
        #here it is nn.Module

        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NN(784,10)
x = torch.randn(64,784)
print(model(x).shape)

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#Load the data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#initialise the network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train network
for epoch in range(num_epochs):#Goes over each epoch
    for batch_idx, (data, targets) in enumerate(train_loader):
        #Goes over every item in the batch
        #which batch index do we have is found from above
        data = data.to(device=device)
        targets = targets.to(device=device)
        #Output for data.shape is 64,1,28,28
        #We want it to be 64,784
        data = data.reshape(data.shape[0], -1)
        #print(data.shape)

        #Forward part
        scores = model(data)
        loss = criterion(scores, targets)

        #Backward part
        optimizer.zero_grad()
        loss.backward()

        #Gradient Descent or Adam Step
        optimizer.step()
        #update weights computed in loss.backward()


#Check accuracy on training and testing
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    if loader.dataset.train:
        print("Checking accuracy on training")
    else:
        print("Checking accuracy on testing")

    with torch.no_grad():
        #we dont need gradients for calculating acccuracy
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            #Shape of scores is 64x10
            #We want to know which one is the max of those 10 digits

            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {num_correct/num_samples * 100:.2f}")

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
