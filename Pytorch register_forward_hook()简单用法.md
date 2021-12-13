# Pytorch register_forward_hook()简单用法

[PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection](https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection)使用pytorch复现原论文，其中出现register_forward_hook()，相关代码如下

```python
        # Extract and register intermediate features of VGG-16
        self.vgg16[3].register_forward_hook(conv_1_2_hook)
        self.vgg16[8].register_forward_hook(conv_2_2_hook)
        self.vgg16[15].register_forward_hook(conv_3_3_hook)
        self.vgg16[22].register_forward_hook(conv_4_3_hook)
        self.vgg16[29].register_forward_hook(conv_5_3_hook)
```

查阅文档发现其用法是在不改动网络结构的情况下获取网络中间层输出

很多时候，我们无法直接修改网络的源代码，比如在pytorch中已经封装好的网络，那么这个时候就可以利用hook从外部获取Module的中间输出结果了。

```python
features = []
def hook(module, input, output): 
    features.append(output.clone().detach())

net = XXXNet() 
x = torch.randn(2, 3, 32, 32)  
handle = net.conv2.register_forward_hook(hook)
y = net(x)
print(features[0])
handle.remove()

```

取出网络的相应层后，对该层调用register_forward_hook方法。这个方法需要传入一个hook方法

```python
hook(module, input, output) -> None or modified output

```

- module：表示该层网络

- input：该层网络的输入

- output：该层网络的输出

  从这里可以发现hook甚至可以更改输入输出(不过并不会影响网络forward的实际结果)，不过在这里我们只是简单地将output给保存下来。

  需要注意的是hook函数在使用后应及时删除，以避免每次都运行增加运行负载。



参考：https://blog.csdn.net/qq_40714949/article/details/114702690