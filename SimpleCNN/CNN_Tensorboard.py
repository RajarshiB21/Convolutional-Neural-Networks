#Importing the libraries
import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#Making the NN structure
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))#28 becomes 14 for MNIST
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes) ##For the size of the images in MNIST, it is passed through pool twice before this

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

##Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

#Load the Data
train_dataset = datasets.MNIST(root="C:/Users/rajar/PycharmProjects/CNN Architectures/dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="C:/Users/rajar/PycharmProjects/CNN Architectures/dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Initialise the Network
model = CNN(in_channels= in_channels, num_classes=num_classes).to(device)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter(f'runs/MNIST/tryingout_tensorboard')

#Train the Network
#initialise tqdm in separately

step = 0
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for bath_idx, (data,targets) in loop:
        #Get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        #Forward
        scores = model(data)
        loss = criterion(scores, targets)

        #Backward
        optimizer.zero_grad()
        loss.backward()

        #Gradient Descent or adam step
        optimizer.step()

        # Calculate running training acuracy
        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct) / float(data.shape[0])

        # Update tqdm
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item(), acc=running_train_acc, device=device)

        #Update Tensorboard
        writer.add_scalar('Training Loss', loss, global_step= step)
        writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)
        step += 1


#Check accuracy on training and test
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")