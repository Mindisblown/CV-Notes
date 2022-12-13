所有代码源自\# https://github.com/xmu-xiaoma666/External-Attention-pytorch，仅供方便查阅

# Self-Attention

~~~python
"""
你有一个问题Q，然后去搜索引擎里面搜，搜索引擎里面有好多文章，每个文章V有一个能代表其正文内容的标题K，然后搜索引擎用你的问题Q和那些文章V的标题K进行一个匹配，看看相关度（QK --->attention值），然后你想用这些检索到的不同相关度的文章V来表示你的问题，就用这些相关度将检索的文章V做一个加权和，那么你就得到了一个新的Q'，这个Q'融合了相关性强的文章V更多信息，而融合了相关性弱的文章V较少的信息。这就是注意力机制，注意力度不同，重点关注（权值大）与你想要的东西相关性强的部分，稍微关注（权值小）相关性弱的部分。
https://www.zhihu.com/question/427629601/answer/1558216827

QKV本质上在做向量之间的内积运算，而内积表征了一个向量在另一个向量上的投影，也就是两个向量之间的夹角。投影的值越大，表示两个向量相关度高。
"""
~~~

**1.Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks arXiv21**

​		使用两个线性层替代Self-Attention，减少计算量；自注意力机制仅仅利用自身样本内的信息，在单个样本内捕获长距离依赖关系，忽略了不同样本之间同一类别的潜在联系，引入两个MLP(M_k, M_v作为K, V，且作为整个数据的memory)，隐式的学习整个数据集的特征。

**2.Attention Is All You Need NIPS17**

​		Transformer的开山之作。QK相乘得到attention map，再与V相乘得到加权后特征，最终经过FC进行特征映射。Scale：除以d_k的原因在于保持QK的方差，QK向量的维度越大，点积往往越大，而softmax在归一化时会被大值主导，分配给较大值更接近于1。假设QK服从均值0方差1的分布，那么在QK点击相乘后方差=q1k1+q2k2+...=d。

**3.Dual Attention Network for Scene Segmentation CVPR19**

​		DANet，Self-attention + self-channel-attention，QKV的生成不再由self-attention中的Linear映射得到。最后直接sum两个attention map。

# Channel Attention

**1.Squeeze and Excitation Network CVPR18**

​		最经典的通道注意力SENet，学习channel之间的相关性。对于卷积输出后的特征F1，得到一个和通道数一样的1D向量[1, 1, C]作为每个通道得分，最终将得分图与特征图相乘。

​		Squeeze：[C, H, W]特征图进行全局平均池化得到[1, 1, C]特征图，这个特征图具有全局感受野。Excitation：使用FC层，对Squeeze之后的结果进行非线性变换。

​		分类模型一般添加到一个block结束的位置，对一个block的信息进行refine。检测模型一般添加到backbone的stage、block等结束位置。

**2.Selective Kernel Networks CVPR19**

​		SENet加强版本，SKNet主要分为Split、Fuse、Select三个模块。

​		Split：使用不同大小卷积核得到多个feature map；Fuse：将多个feature map进行element-wise得到总的feature map U，对[H, W]维度进行全局平均池化，使用FC层进行降维得到Z；Select：使用FC层计算每个通道的attention weight，经过softmax后与原特征图相乘得到新的特征图，对所有特征图再进行element-wise得到最终的feature map V。

**3.ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks CVPR202**

​		SENet的轻量版，使用感受野为K的1D卷积替代SE中的FC层。

# Channel + Spatial Attention

**1.CBAM: Convolutional Block Attention Module ECCV18**

​		Channel attention与SENet相似，在[H, W]维度分别使用了全局平均池化与全局最大值池化，然后add两个特征图(MLP由Conv层替代)。

​		Spatial attention现在channel维度进行最大值与平均池化并进行cat，随后使用7x7卷积核来提取空间的注意力，经过sigmoid归一化后得到最终的feature map。

**2.BAM: Bottleneck Attention Module BMVC18**

​		与CBAM相似，BAM直接add Channel与Spatial维度attention矩阵。

**3.EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network arXiv21**

​		PSA模块，特征Split成多组子特征图，每组子特征图使用不同大小的卷积核提取新特征，每组新特征经过SENet的channel attention，最后concat所有attention map。不同尺度空间的通道信息来丰富特征空间。

