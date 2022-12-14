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

**4.ResT: An Efficient Transformer for Visual Recognition arXiv21**

​		针对self-attention计算复杂度高，并且在multi-head中每个head只包含QKV的部分信息。如果QKV维度较小，那么获取的信息就会不连续。因此作者在FC之前引入卷积来降低空间复杂度。

**5.MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning arXiv19**

​		Self-attention仅能捕获全局的依赖，当维度较大时，全局捕获的能力也会变弱。在SA的基础上使用多个不同尺寸大小的深度可分离卷积来捕获局部依赖。

**6.An Attention Free Transformer ICLR21**

​		减少计算量，KV相乘求和除K输出out，out与Q相乘(不同SA求点积，直接对应位置相乘)。

**7.VOLO: Vision Outlooker for Visual Recognition arXiv21**

​		获取QKV时都需要进行feature embedding，embedding图片大则计算量大，图片小则信息损失较多。作者引入Outlooker获取更加细粒度的特征表示(flod unflod滑窗操作)。

**8.Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition arXiv21**

​		Permutator编码空间与通道信息。

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

**4.Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks arXiv19**

​		SKNet同一作者，轻量化attention。特征分组后与它的global pooling计算相似性得到attention mask，减均值除方差经过sigmoid后与原始feature map相乘。

# Spatial Attention

**1.A2-Nets: Double Attention Networks NIPS18**

​		1x1卷积获得ABV(类似SA中的QKV)，AB点乘得到全局信息的attention G，GV点乘得到最终结果。

# Channel + Spatial Attention

**1.CBAM: Convolutional Block Attention Module ECCV18**

​		Channel attention与SENet相似，在[H, W]维度分别使用了全局平均池化与全局最大值池化，然后add两个特征图(MLP由Conv层替代)。

​		Spatial attention现在channel维度进行最大值与平均池化并进行cat，随后使用7x7卷积核来提取空间的注意力，经过sigmoid归一化后得到最终的feature map。

**2.BAM: Bottleneck Attention Module BMVC18**

​		与CBAM相似，BAM直接add Channel与Spatial维度attention矩阵。

**3.EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network arXiv21**

​		PSA模块，特征Split成多组子特征图，每组子特征图使用不同大小的卷积核提取新特征，每组新特征经过SENet的channel attention，最后concat所有attention map。不同尺度空间的通道信息来丰富特征空间。

**4.SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS ICASSP21**

​		主要降低了计算复杂度。输入特征分组，每一组再split成两个分支，每个分支计算分别计算channel与spatial attention，并且两种attention都使用可训练的参数；将两个分支结果concat并且进行channel shuffle(首先对通道进行拆分[group, c/group]，将两个维度转置[c/group, group]，最终reshape[c/group * group])。

# Coordinate Attention

**1.Coordinate Attention for Efficient Mobile Network Design CVPR21**

​		Coordinate Attention基于coordinate information embedding和coordinate attention generation两个步骤来编码通道关系和长距离关系。

​		通道注意力中经常使用GAP对空间信息进行全局编码，但是这样就丢失了位置信息。coordinate information embedding将GAP分解为沿着H和W方向分别池化，获取HW相关的位置信息。

​		coordinate attention generation将两个池化结果concat，经过1x1的conv再将输出按照H和W方向划分成两组，分别经过1x1卷积后再相乘。

