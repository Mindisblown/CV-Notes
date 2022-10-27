CV领域的数据增强技术可以分为以下几种：

​		数据增强到底是增加了训练数据还是控制模型复杂度？

​		数据增强这两种效果都能兼顾，图像增强是在原图上进行空间会像素级别的改变，并不是来源于数据的分布，也不是独立同分布的抽样。具体而言，这些增强数据的作用是增强模型对某种变换的invariance(减少模型估计的方差)，也就是控制了模型的复杂度。

​		

1.基于原始图像

几何变换：翻转、裁剪、旋转、平移...

色彩空间变换：对比度、锐化、白平衡、色彩抖动...

RandomErase/CutOut：随机擦除、随机从图片上丢弃部分区域图像

Mixup：两张图按一定比例融合，融合后的图带有两张图的标签

CutMix：与CutOut类似，但是将矩形区域填充为另一张图片的像素

AugMix：对一张图进行3个并行增强分支处理，然后按照一定比例融合

CopyPaste：依据Mask将一张图的内容贴到另一张图中

GridMask：在图像上生成网格状的mask

Mosaic：对四张图像进行随机裁剪，再拼接到一张图上，丰富了图片背景

常用增强模块：torchvision与albumentations，其中数据格式不同，分别为RGB与BGR



KeepAugment - KeepAugment: A Simple Information-Preserving Data Augmentation Approach，使用saliency map计算得到图像中的关键区域，保留关键区域或将关键区域贴到其他图上

MaxUp - MaxUp: Lightweight Adversarial Training with Data Augmentation Improves Neural Network Training，使用随机扰动或转换方法生成一系列增强数据，从中挑选损失最大的用于更新网络，最小化损失最大的增强样本替代最小化 增强样本的平均损失



~~~python
"""
	mixup分支
	1.mixup - mixup: Beyond Empirical Risk Minimization，相当于在两个输入x1与x2之间线性插值，label进行标签平滑。数据的离散空间进行连续化，提高数据空间的平滑性
	2.mainfold mixup - Manifold Mixup: Better Representations by Interpolating Hidden States，将mixup扩展到特征空间，在更高维度空间进行mixup提供了额外的训练信号，平滑决策边界、拉开各类别高置信度空间的间距并且展平隐层输出的数值
	3.adaptive mixup - Mixup as locally linear out-of-manifold regularization，当进行mixup操作时，混合样本会发生数据之间的相撞(变成另一种标签的数据)，这会让模型效果变差，引入新的分类器，判断输入的是mixup样本还是原样本，再训练一个自适应参数来保证mixup不与原数据相撞
	4.cutmix - Cutmix: Regularization strategy to train strong classifiers with localizable features，以往的mixup可以看做一种信息的混合，但是对于图像而言，信息往往是连续的，用一张图像的部分连续信息去代替另一张图像的连续信息。但是，一旦裁剪到背景区域，并没有改变图像，标签却会改变，这将提供错误的信息
	5.patchup - PatchUp: A Regularization Technique for Convolutional Neural Networks，mainfold与cutmix的结合体，在特征空间进行cutmix
	6.puzzlemix - Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup，cutmix的改进，使用传统方法计算显著性，对信息密度最高的地方进行cutmix，避免裁剪到背景
	7.stylemix - StyleMix: Separating Content and Style for Enhanced Data Augmentation，使用图像的风格和内容两个角度来改善图像插值过程 
	8.mixstyle - Domain generalization with mixstyle，混合图像的style信息
	9.supermix - SuperMix: Supervising the Mixing Data Augmentation，输入图片计算显著性区域，根据两张图片的显著区域进行mixup
"""
~~~



2.基于特征空间

MoEx(Moment Exchange)

​		On Feature Normalization and Data Augmentation CVPR2021 
​		在特征空间进行数据增强，因此不受输入数据类型限制。

​		x1作为已经归一化后的特征，再使用x2的均值方差对x1进行扰动

~~~python
"""
Aug实现过程
x1 ---> feature extrct ---> [(x-mean)/std] * gamma + beta ---> x1_input
x2 ---> feature extrct ---> [(x-mean)/std] * gamma + beta

利用x2的mean和std，实际的x2_input并没有使用
	x1_input * x2_mean + x2_std
	
训练Loss计算
随机从输入的批次中选择一个x2与x1进行处理
output = model(x1, x2)
loss = lam * CE_Loss(output, x1) + (1-lam) * CE_Loss(output, x2)
lam为超参数
"""
~~~



3.基于GAN网络

生成对抗网络目的在于学习数据的分布，而不是数据类别间的边界。其核心组件分为生成器和辨别器，生成器负责生成具有noise的假数据，辨别其接受GT图像与生成图像，以此辨别图像的真假。生成器的优化目标是生成的数据能够欺骗辨别器，而辨别器的目标是不能被假数据欺骗，随着两个网络的对抗将生成难以分辨的真实假数据。



4.基于NAS策略

AutoAugment

使用强化学习使网络在数据增强的搜索空间中自动寻找最佳的图像增强策略。验证集上的acc作为reward

RandomAugment

减小了搜索空间，从大的增强列表中随机(概率相等)取几种增强方式

Fast AutoAugment/DADA

在AutoAugment基础上进行优化，提高搜索效率

