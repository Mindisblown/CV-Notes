# YOLOV4

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

# YOLOV5

**Label Assignment**

​		不同于以往使用IOU的匹配方式，V5采用GT box与anchor的宽高比来分配：

​		1.抽取GT box的宽高，将宽高与anchor相除。只需要满足anchor与当前特征层中GT box的宽高比例在阈值范围内，太大表示当前特征图无anchor能够很好的表征GT box，这一步完成不同尺度的特征层处理不同大小的物体；

​		2.满足条件1，取GT box的xy(中心点)，一个block区域分为四个象限，左上，右上，左下，右下，判断xy落在哪个象限内，最后通过offset来扩张，讲1个grid扩张到3个grid。

**Letter Box**

~~~python
"""
原图输入尺寸 w1 h1
期望输入尺寸 w

scale = min(w1/w, h1/w)

新的输入尺寸
	w2 = w1 * scale
	h2 = h1 * scale
  
此时计算填充黑边数量，一边必定满足期望的输入
	h2 - h1 = 填充高度
	被32整数,取余数
	np.mod(number, 32) = number
	余数除2，表示两边填充
	number / 2

因此最后的高度h2 = h2 + number
"""
~~~

# YOLOX

基于yolov3-spp结构改进

**Decoupled head**

​		现有的YOLO检测算法框架中使用一个预测头完成Cls+Reg+Obj的预测，多任务的confilct必然会导致问题。因此，作者使用解耦头，分别预测Cls、Reg、Obj。

**Strong data augmentation**

​		Mosaic+Mixup

**Anchor-free**

​		之前的yolo算法需要计算对应数据的anchor，但是这样是基于某个数据集的，不具有泛化性。anchor机制也增加了检测头的复杂性，对一幅图像存在多个预测。

**SimOTA-Label Assignment**

~~~python
"""
三个解耦头输出(假设输出尺度20*20 40*40 80*80)
cls 80dim数据集coco类别，reg 4dim回归坐标，obj 1dim前景背景  85dim
解耦头
	20*20*80	40*40*80	80*80*80
	20*20*4		40*40*4		80*80*4
	20*20*1		40*40*1		80*80*1
Concat
	20*20*85	40*40*85	80*80*85
	400*85		1600*85		6400*85

三个输出尺度Concat
	8400*85

Anchor-free未设置anchor box如何分配正样本
	使用下采样信息
	20*20下采样32倍，因此使用32*32的anchor box 400个
	40*40下采样16倍，因此使用16*16的anchor box 1600个
	80*80下采样8倍，因此使用8*8的anchor box 6400个

构造正样本anchor box

1.anchor box初步筛选，已知拥有8400个anchor box：
	a.根据GT box标注信息[x_center, y_center, width, height]，计算左上-右下位置		坐标[x1, y1, x2, y2]，计算每个anchor中心[x_center, y_center]，需满足：
	x_center - x1 > 0
	y_center - y1 > 0
	x2 - x_center > 0
	y2 - y_center > 0
	四个条件满足时可知anchor box的中心点位于GT box内
	
	b.以GT box中心为基准，构造5*5矩形，计算anchor box中心点落在矩形内的anchor box
	
2.精细筛选
	经过步骤1，假设得到1000个正样本anchor box，这些anchor box的位置与之前8400个位置一一对应，依据anchor box的位置信息获取预测信息：
	Reg [1000, 4]
	Obj [1000, 1]
	Cls [1000, 80]
	假设GT box为3，那么将1000预测框与3真实框计算IOU与Cls：
	iou = [gt_boxes, pred_boxes]
	iou_loss = -torch.log(iou + 1e-8)
	
	pred_cls = pred_obj * pred_cls
	cls_loss = binary_cross_entropu(pred_cls, gt_cls)
	维度为[3, 1000]
	
3.cost计算
	cost = cls_loss + 3 * iou_loss + 10000.0 * (~is_in_boxes_and_center)
	不在中心区域的box增加了cost，相当于过滤操作
	
4.SimOTA
	4.1 动态候选框数量
		新建全为0的matching_matrix，维度与cost一致[3, 1000]
		设置候选框数量为10
		从iou中取出值最大的前10个候选框[3, 10]，每个gt box对应10个候选框，iou从大到小
		axis=1上求和，得到该gt box10个候选框iou的总和[3, 1]
		
		假设为[3.67, 4.15, 3.55].T
		直接取整为[3, 4, 5].T
		因此可知3个gt box分别划分3、4、3个候选框
		matching_matrix依据cost与候选框数量将对应位置=1，其余=0
	4.2 过滤共用的候选框
		当出现1和2的gt box共用一个候选框，需要进行过滤
		axis=0上求和，得到[1000, 1]，当出现共用时，对应列数值>1
		取出>1对应列的数值，比较二者cost，大者置为0，丢弃该候选框
		

is_in_boxes_and_center是精细过滤的样本
is_in_box_anchor是初步筛选的样本

前者用于构造正样本anchor box，而最终计算loss使用初筛样本与构造的正样本anchor box
anchor用于构造正样本，Reg分支实际在预测这些正样本anchor box相对于GT box的偏移量
"""
~~~

# YOLOF

只需要看一层特征，一个特征层完成目标检测

**Dilated Encoder**

​		仅使用输出的C5特征图，但是其感受野是单一的，这对于不同尺度的物体是不友好的。如何具有多种尺度的感受野是这个模块的出发点。

​		输出的C5经过4层膨胀卷积，每层膨胀率不同，通过膨胀率获取感受野。

​		注意：感受野并不能完全覆盖多尺度，C5下采样丢失信息较多，小物体会丢失大部分信息。

**Uniform Matching**

​		所有anchor box与gt box计算L1距离，保留topk结果，选择最近的K个anchor box作为正样本，在此基础上进一步使用iou来筛选K个结果，阈值为0.15。C5尺度太小，anchor box在图像上分布很稀疏，iou应取较小值。

​		注意：分配策略过于暴力，未考虑相邻物体共用box。