import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # first layer
        self.fc1 = nn.Linear(1 * 28 * 28, 1 * 28 * 28)
        # residual layer
        self.conv1 = nn.Conv2d(1, 1, 5, padding=2)
        # last layer
        self.fc2 = nn.Linear(1 * 28 * 28, 10)

    def forward(self, x):
        x_shape = x.shape
        x = f.relu(self.fc1(x.view(-1, 1 * 28 * 28))).view(x_shape)
        x = f.relu(self.conv1(x)) + x
        x = self.fc2(x.view(-1, 1 * 28 * 28))
        return x


net = Net()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True,  download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=64, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# put on gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

for epoch in range(1):
    for batch_idx, (input_batch, label) in enumerate(train_loader, 0):
        input_batch = input_batch.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        output = net(input_batch)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        # print statistics
        if batch_idx % 50 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, loss.item()))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,  download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=64, shuffle=True)

net.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for _, (input_batch, label) in enumerate(test_loader, 0):
        input_batch = input_batch.to(device)
        label = label.to(device)
        output = net(input_batch)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    print(correct)
    print('Accuracy: %.3f' % (correct / len(test_loader.dataset)))
