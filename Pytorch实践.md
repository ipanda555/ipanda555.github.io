# Pytorch实践

## 第一章：线性模型

```python
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list= []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print('w =', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_data)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('MSE=', l_sum / 3)
        w_list.append(w)
        mse_list.append(l_sum / 3)
```
## 第二章：梯度下降

```python
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forward(4))

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w =', w, 'loss=', cost_val)
print('Predient (after training)', 4, forward(4))

```

## 第四章：反向传播

```python
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print('predict (before training)', 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

    print('progress:', epoch, l.item())

print('predict (after training)', 4, forward(4).item())

```

## 第五章：线性模型完整

```python
import torch.nn as nn


# 构造模型的时候把其构造为一个类，继承于Module
# Modulel里面有很多方法
# init是构造函数，初始化时使用
# forward是前馈时执行的计算
# 为何没有backword?是因为在Module里自动完成
# Linear继承自Module，可以自动进行反向传播

class LinearModle(nn.Module):
    def __init__(self):
        super(LinearModle, self).__init__()  # 调用父类的构造
        self.linear = nn.Linear(1, 1)  # nn.Linear是pytorch里面的一个类，加括号是构造对象,包含了权重和偏置

    def forward(self, x):
        y_pred = self.linear(x)  # 对象后面加括号，意味着实现一个可调用的对象
        return y_pred


model = LinearModle()

```
## 第六章：逻辑斯蒂回归

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogistRegressionModel(nn.Module):
    def __init__(self):
        super(LogistRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogistRegressionModel()

criterion = nn.BCELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

## 第七章：多尺度输入

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
model = Model()

criterion = nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


```
## 第八章：数据集加载

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()  # 无参数;

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 第九章：Softmax_Classifier

```python
#第一种
# import numpy as np
#
# y = np.array([1, 0, 0])
# z = np.array([0.2, 0.1, -0.1])
#
# y_pred = np.exp(z) /np.exp(z).sum()
# loss = (- y * np.log(y_pred)).sum()
# print(loss)



#第二种
# import torch
# import torch.nn as nn
#
# y = torch.LongTensor([0])
# z = torch.Tensor([[0.2, 0.1, -0.1]])
#
# criterion = nn.CrossEntropyLoss()
# loss = criterion(z, y)
# print(loss)



#第三种
# import  torch
# import torch.nn as nn
# criterion = nn.CrossEntropyLoss()
# Y = torch.LongTensor([2, 0, 1])
#
# Y_pred1 = torch.Tensor([
#     [0.1, 0.2, 0.9],
#     [1.1, 0.1, 0.2],
#     [0.2, 2.1, 0.1]
# ])
# Y_pred2 = torch.Tensor([
#     [0.8, 0.2, 0.3],
#     [0.2, 0.3, 0.5],
#     [0.2, 0.2, 0.5]
# ])
#
# l1 = criterion(Y_pred1, Y)
# l2 = criterion(Y_pred2, Y)
# print('Batch Loss1 = ', l1.data, '\nBatch Loss2= ', l2.data)


import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='../dataset/minist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/minist', train=False, download=False,transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train():
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy om test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```
