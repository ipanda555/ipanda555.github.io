# offcial_demo更新ing

## **model.py**

1. 先定义类LeNet，继承nn.Moudule这个父类，这里面实现两个方法，一个是写出网络每层的结构，另外一个是写出正向传播的过程。这样才能够使得对象实例化的时候按正向传播的过程走下来。
2. 在最后fc3的时候不加softmax函数是因为在计算损失函数的时候，CELoss已经加过softmax过程了

## **train.py**

1. 首先引入所需要的包

2. 下载训练数据，使用`torchvision.datasets()`函数，如果本地未下载，则将download设置成True，设置数据保存路径，数据转换格式（需要自己写转换格式的方法）等等

3. 载入训练数据，使用`torch.utils.data.Dataloader()`,设置载入的图片路径，batch_size，是否打乱顺序num_works等等

4. 下载验证数据

5. 载入验证数据val_loader

6. 载入之后，使用`val_data_iter =  iter(val_loader)`,将生成的val_loader转化成可迭代的迭代器

7. 使用next方法获取一批数据，包含测试图像以及图像对应的标签值

   `val_image, val_label = val_data_iter.next()`

8. 导入官方含有的标签值

9. 实例化模型net = LeNet()

10. 定义损失函数，`loss_function = nn.CrossEntropyLoss()`
    其中CELoss中包含了softmax函数，所以最后一层不需要加softmax

11. 定义优化器，optimizer = optim.Adam(所有可训练的参数，学习率)

12. 开始训练过程，循环设置多少次，

    定义训练过程中的累加损失running_loss

13. 遍历训练集样本，for step,data in enumerate(train_laoder,start=0)

    enumerate不仅可以返回每批数据的data，还可以返回data对应的的步数index

14. 所有输入的图像以及对应的标签inputs, labels = data

15. optimizer.zero_grad(),将历史梯度清零，若不清零则会将历史梯度进行累加

16. 正向传播output = net(inputs)

17. 使用loss = loss_functional(outputs, labels)进行损失计算

18. 接着对loss进行反向传播loss.backward()

19. 进行参数更新optimizer.step()

### 打印信息

1. 每隔多少步打印一次数据信息

   `if step % 500 == 499:`

2. `with  torch.no_grad():`意思是在接下来的过程中不要计算每个结点的误差损失梯度
   否则会占用很多内存资源，在训练和测试在都不要计算每个结点的误差梯度
   with是一个上下文管理器
   
3. `outputs = net(val_name)`进行正向传播

4. `pre_y = torch.max(outputs, dim=1)[1]`
   dim=1的意思是，只想获取第一维度下的index。因为第0维度是batch,
   [1]的意思是，只需要只需要对应的index的值，不需要知道他的最大值是多少，这样就可以得到最大值对应的标签类别

5. torch.eq(predict_y, val_label).sum()得到的是一个tensor类别的，并不是一个数值，因此加上item()方法来得到数值，最后除以总样本得到准确率

6. 打印信息

   ```python
   print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
         (epoch + 1, step + 1, running_loss / 500, accuracy))
   ```

​       其中，[%d]-----打印轮次，[%5d]-----打印步数，%.3f打印准确率

### 设置保存路径

1. 打印信息：全部训练完毕
2. 设置保存路径`save_path = './Lenet.pth'`
3. 将所有模型的参数保存至设置的路径`torch.save(net.state_dict(), save_path)`

## **predict.py**

1. 引入所需要的包
   重点是：`from model import LeNet`

2. 定义transfrom,导入标签值

3. 实例化LeNet:
   `net = LeNet()`

4. 载入我们保存的权重文件
   `net.load_state_dict(torch.load('Lenet.pth'))`

5. 利用torch.unsqueeze方法增加新的维度

   ```python
   im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
   ```

   6.寻找最大输出对应的index

```python
with torch.no_grad():
    outputs = net(im)
    #寻找最大输出对应的index
    predict = torch.max(outputs, dim=1)[1].data.numpy()

print(classes[int(predict)])
```