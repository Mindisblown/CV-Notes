CV领域的数据增强技术可以分为以下几种：

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

