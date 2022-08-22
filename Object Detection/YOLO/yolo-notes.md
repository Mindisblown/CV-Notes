YOLOV4

Backbone ：CSPDarkNet53，含有52个卷积及最后一个全连接层共计53层



Neck：SPP、PAN



Bag of freebies(不增加推理成本)

​		Backbone阶段：CutMix、Mosaic、DropBlock、Label Smoothing

​		Detector阶段：CIOU-Loss、CmBN、DropBlock、Self-Adversarial Training、Cosine Annealing Scheduler、Random Training Shapes、Optimal Hyper-Parameters



Bag of specials(增加推理成本)

​		Backbone阶段：Mish、CSP、MiWRC

​		Detector阶段：Mish、SPP、Modified SAM、Modified PAN、DIOU-NMS



**Label Assignment**

​		Anchor box与目标框的IOU大于阈值，就作为正样本，其他为负样本，无忽略样本概念。

**DropBlock**

​		与CutOut类似，CutOut用于图像数据增强，而DropBlock作用于CNN的特征层中。DropOut随机屏蔽一部分特征，而DropBlock随机屏蔽一部分连续区域。

**CutMix**

​		Mixup与CutOut结合，区域删除操作变成截取一张图片同样大小填充在原图像中删除的区域，并且改变新图片的标签。

**Label Smoothing**

​		Softmax+CE在做分类任务时，当logits趋向负无穷时，Softmax出来的值趋近于0，当logits越大那么Softmax出来的值越趋近于1。这种极端的趋向性，使得网络只关注positive，这会导致泛化能力下降，并且CNN很难输出负无穷的情况。而label smoothing通过修改label的编码形式，原来为1的位置=1-a，原来为0的位置为a / (class_num - 1)，a一般取0.1。

​		因此对于一个6分类任务，标签编码形式(1, 0, 0, 0, 0, 0)经过label smoothing变成(0.9, 0.02, 0.02, 0.02, 0.02)，从hard label变为soft label。

​		个人理解：1.hard label的形式并没有考虑类内、类间的关联。2.正确类与错误类的logits相差一个常数，并且从熵的角度来说，标签平滑提供了更多信息，3.并且可以看做给数据添加了部分噪音，使网络更加鲁棒。4.从优化角度来说softmax得到的值永远不可能为1，当优化达到一定程度就会达到饱和区，优化效率就会很低。而soft label可以避免进入饱和区。

**CmBN**

​		BN是对当前mini-batch数据进行归一化，CBN则是Cross BN，对当前以及之前3个mini-batch的数据进行归一化，而CmBN可以看做Cross mini-BN，在mini-batch之间不做更新计算，而在单个batch中4个mini-batch之间做完才去更新参数。

**SAT**

​		自对抗训练同样可以视为一种数据增强方法，分为两个阶段。第一阶段使用CNN去改变图片数据，而不更新权重数据(类似图像数据生成)；第二阶段CNN以正常方式在扩充后的图像数据集上。

**CosineAnnealingLR**

​		在训练开始时首先缓慢降低学习率，在训练中途加速下降，最后再缓慢下降
$$
n_{t}=n_{min}+\dfrac{1}{2}(n_{max}^{i}-n_{min}^{i})(1+cos\dfrac{T_{cur}}{T_i}\pi)
$$
​		最大学习率、最小学习率、当前迭代次数、最大迭代次数

**MiWRC**

​		多输入加权残差链接，在FPN中每一层取得的feature视为平等，而MiWRC则认为不同层的feature具有不同的重要性，针对不同尺度特征给予不同的权重比例。

**Mish**
$$
f(x)=xtanh(ln(1+e^{x}))
$$
​		Mish连续可微的非单调激活函数，与ReLU相比，Mish的梯度更加平滑，并且在负值时允许有较小的负梯度，稳定网络梯度流，具有更好的泛化性。

**Modified SAM**

​		在SAM的基础上使用(7,7)卷积替代Max pooling与Average pooling。