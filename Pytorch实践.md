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

