

# Intelligent-Computing-Exercise 

### 智能计算训练

此项目为大三上学期《智能计算》课程的项目汇总，其中包括 基于朴素贝叶斯方法识别手写数字、基于遗传算法优化特征实现的电影评论情感分类，以及基于神经网络的手写数字识别。

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [部署](#部署)
- [文件目录说明](#文件目录说明)
- [版本控制](#版本控制)
- [作者](#作者)
- [鸣谢](#鸣谢)



### 上手指南

###### 开发前的配置要求

1. Keras
1. Numpy
3. Python
2. Anaconda

###### **安装步骤**

```sh
git clone https://github.com/Son4ta/Intelligent-Computing-Exercise.git
```



### 部署

看起来可执行的python都可以如下方式运行：

```sh
python ./文件名.py
```

但network 和 mnist_loader 应在python终端中使用：

```python
import mnist_loader

training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 1.0, test_data=test_data)
```

### 文件目录说明

根目录下有各个版本的代码与预处理过的先验概率文件

```
filetree 
├─.idea
│  └─inspectionProfiles
├─data(影评测试样本)
├─image(手写数字测试样本)
└─result(训练结果)

```



### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。



### 作者

Son4ta@qq.com

知乎:Son4ta&ensp; QQ:1152670339

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*



### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/Son4ta/Intelligent-Computing-Exercise/blob/master/LICENSE.txt)



### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)

<!-- links -->

[your-project-path]:Son4ta/Intelligent-Computing-Exercise
[contributors-shield]: https://img.shields.io/github/contributors/Son4ta/Intelligent-Computing-Exercise.svg?style=flat-square
[contributors-url]: https://github.com/Son4ta/Intelligent-Computing-Exercise/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Son4ta/Intelligent-Computing-Exercise.svg?style=flat-square
[forks-url]: https://github.com/Son4ta/Intelligent-Computing-Exercise/network/members
[stars-shield]: https://img.shields.io/github/stars/Son4ta/Intelligent-Computing-Exercise.svg?style=flat-square
[stars-url]: https://github.com/Son4ta/Intelligent-Computing-Exercise/stargazers
[issues-shield]: https://img.shields.io/github/issues/Son4ta/Intelligent-Computing-Exercise.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/Son4ta/Intelligent-Computing-Exercise.svg
[license-shield]: https://img.shields.io/github/license/Son4ta/Intelligent-Computing-Exercise.svg?style=flat-square
[license-url]: https://github.com/Son4ta/Intelligent-Computing-Exercise/blob/master/LICENSE.txt



