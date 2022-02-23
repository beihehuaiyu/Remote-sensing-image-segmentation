# 基于paddleseg实现遥感图像分割
本项目利用paddleseg实现对遥感图像的分割。
## 一、准备工作
这里主要包括
### 1、数据集的解压、对数据进行强化及构建数据集
### 2、安装paddleseg
## 1.1、解压数据集
### ①关于数据集
UDD数据集是北京大学图形与交互实验室采集并标注的，面向航拍场景理解、重建的数据集。本项目所使用的是UDD5，该数据集将图分为5类：Vegetation、Building、
Road、Vehicle、Other。其中训练集中有120张图片，验证集中有40张图片。
![](https://ai-studio-static-online.cdn.bcebos.com/6d7992ca287648759ca7ea7b7535b5a893a4aeeb53324aff81961963b46683f1)
![](https://ai-studio-static-online.cdn.bcebos.com/3b8f55fbba8642e48c1690babadccd7c0b4cdab2af014ac095222f8850dcd256)
### ②关于PaddleSeg
>PaddleSeg是基于飞桨PaddlePaddle开发的端到端图像分割开发套件，涵盖了高精度和轻量级等不同方向的大量高质量分割模型。通过模块化的设计，提供了配置化驱动和API调用等两种应用方式，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用

>Github链接：https://github.com/PaddlePaddle/PaddleSeg

>Gitee链接：https://gitee.com/paddlepaddle/PaddleSeg

```python
#解压数据集
!unzip data/data75675/UDD5.zip -d UDD
```
```python
#安装paddleseg
!pip install paddleseg
```
## 1.2、数据强化及数据集的构建
在该步中对图片进行一定概率的水平、竖直翻转和旋转并将图片放缩为256x256大小并进行标准化。
```python
# 构建训练用的数据增强和预处理
import paddleseg.transforms as T
transforms = [
    T.Resize(target_size=(512, 512)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.Normalize(),
    T.RandomDistort()
]
# 构建训练集
from paddleseg.datasets import Dataset
train_dataset = Dataset(
    transforms=transforms,
    dataset_root='UDD',
    num_classes=5,
    mode='train',
    train_path='UDD/metadata/train.txt',
    )
```
# 二、模型训练
## 2.1 搭建模型
<ul>
<li>模型：UNet++
<li>损失函数：BCELoss
<li>优化器：Adam
<li>学习率变化：CosineAnnealingDecay
 </ul>
 ```python
from paddleseg.models import UNetPlusPlus

model = UNetPlusPlus(in_channels=3,
                    num_classes=5,
                    use_deconv = False,
                    align_corners = False,
                    pretrained = None,
                    is_ds = True
                    )
 ```
 ```python
 import paddle
# 设置学习率
iters=1000
base_lr = 1e-4
lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  # 余弦衰减
# 设置优化器
optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters())  # Adam优化器
```
```python
from paddleseg.models.losses import CrossEntropyLoss
losses = {}
losses['types'] = [CrossEntropyLoss()]
losses['coef'] = [1]
```
## 2.2开始训练
```python
from paddleseg.core import train
train(
    model=model,
    train_dataset=train_dataset,
    optimizer=optimizer,
    save_dir='output',
    iters=1000,
    batch_size=4,
    save_interval=200,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)
 ```
 # 三、模型评估
 ## 3.1、加载模型参数
 ```python
 model_path = 'output/best_model/model.pdparams'
if model_path:
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    print('Loaded trained params of model successfully')
else: 
    raise ValueError('The model_path is wrong: {}'.format(model_path))
 ```
 ## 3.2、加载验证数据
 ```python
 # 构建验证用的transforms
import paddleseg.transforms as T
transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]

# 构建验证集
from paddleseg.datasets import Dataset
val_dataset = = Dataset(
    transforms=transforms,
    dataset_root='UDD',
    mode='val',
    val_path='UDD/metadata/val.txt'
)
```
## 3.3、评估
```python
from paddleseg.core import evaluate
evaluate(
        model,
        val_dataset)
 ```
 # 四、效果可视化
 ```python
 image_path = 'UDD/val/src/DJI_0532.JPG'
 
from paddleseg.core import predict
predict(
        model,
        model_path='output/best_model/model.pdparams',
        transforms=transforms,
        image_list=image_path,
        save_dir='output/results'
    )
 ```
    
  # 五、自我介绍
>太原理工大学 软件学院 软件工程专业 2020级 本科生 王志洲

>AIstudio地址链接：https://aistudio.baidu.com/aistudio/personalcenter/thirdview/559770

>码云地址链接：https://gitee.com/wang-zhizhou_1
