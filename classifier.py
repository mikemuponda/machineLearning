import torch
import torch.nn as nn
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torch.nn import functional as F
from PIL import Image
class MultinomialLogisticRegression(nn.Module):
    def __init__(self,input_size,hidden_size,classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size[0])
        self.linear2=nn.Linear(hidden_size[0],hidden_size[1])
        self.linear3=nn.Linear(hidden_size[1],classes)

    def forward(self, feature):
        output=torch.relu(self.linear1(feature))
        output=torch.relu(self.linear2(output))
        output=self.linear3(output)
        output=F.log_softmax(output,dim=1)
        return output
    
def ConvertImageToTensor(image_path):
    img = Image.open(image_path)
    img_2dvector = np.array(img)
    flattened_img=img_2dvector.flatten()
    img_tensor=torch.tensor(flattened_img,dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

DATA_DIR=""
if('MNIST data' in os.listdir()):
    DATA_DIR="./MNIST data"
else:
    DATA_DIR = "."
download_dataset = False

train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset)

X_train = train_mnist.data.float()
y_train = train_mnist.targets
X_test = test_mnist.data.float()
y_test = test_mnist.targets
# Sample random indices for validation
test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

# Create validation set
X_valid = X_train[indices]
y_valid = y_train[indices]

# Remove validation set from training set
X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

batch_size = 48
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

num_epochs=10
learning_rate=0.001
input_size=28*28
hidden_size=[128,64]
classes=10

model = MultinomialLogisticRegression(input_size,hidden_size,classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        labels = labels.long()
        
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    model.eval()
    correct = 0
    total = 0
    for images, labels in val_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs,dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct.item() / total

print('Validation Accuracy: {} %'.format(accuracy))
print('Training complete')
image_path = input("Please enter image path: ")
image_tensor=ConvertImageToTensor(image_path)
with torch.no_grad():
    model.eval()
    outputs = model(image_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()
print("Predicted Class: ", predicted_class)
