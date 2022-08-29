**ResNet**

​		最核心的在于identity mapping，它保证了可以训练更深的网络。学到的内容与真实输入的残差F(x)=H(x)-x，x表示真实输入，H(x)表示学到的内容，网络去学习F(x)这个残差项。

​		当无差距时，残差项=0，H(x)=x就达到了恒等映射的功能。另一方面通过residual，越来的微分公式由F(G(x))变为F(G(x))+G(x)
$$
\dfrac{\partial{F(G(x))}}{\partial{x}}=\dfrac{\partial{F(G(x))}}{\partial{G(x)}}\dfrac{\partial{G(x)}}{\partial{x}}
$$

$$
\dfrac{\partial({F(G(x))}+G(x))}{\partial{x}}=\dfrac{\partial{F(G(x))}}{\partial{G(x)}}\dfrac{\partial{G(x)}}{\partial{x}}+\dfrac{\partial{G(x)}}{\partial{x}}
$$

​		梯度一般在0附近的正态分布，越深的网络由于连乘，导致梯度越来越小，直至梯度消失。一个较小值乘较大值仍为很小的值，而在第二项由于残差项的存在，仍然有一个较大的梯度提供给网络。

​		由于残差项的存在促使网络可以自主去选择合适的复杂度，难学习的样本可以在更深的地方学习到residual，容易学习的样本，更深层的residual可以为0，导致后面深层全部为恒等映射。

**RepVGG**

​		kernel_size(3, 3)卷积在GPU上的计算密度(理论运算量除以处理时间)比kernel_size(1, 1)与kernel_size(5, 5)快四倍左右。

​		单路架构推理时很快，因为并行度告，大而整运算效率远超小而碎，且单路结构非常节省内存，灵活性也更加优秀。

​		RepVGG在主体部分只有一种算子：kernel_size(3, 3)卷积+ReLU

重参数化

​		训练时给每一个3x3卷积添加一个1x1卷积与恒等映射分支，构成一个RepVGG Block，训练完成后，最终使用这些之路与3x3卷积合并构成新的3x3卷积。

​		

文章的核心在于多路模型到单路模型的转换：

1.Conv与BN层合并

​		卷积计算公式
$$
Conv(x)=Wx+b
$$
​		BN计算公式
$$
BN(x)=\gamma*\dfrac{x-mean}{\sqrt{var}}+\beta
$$

$$
BN(Conv(x))=\gamma*\dfrac{W(x)+b-mean}{\sqrt{var}}+\beta
$$

$$
BN(Conv(x))=\dfrac{\gamma*W(x)}{\sqrt{var}}+(\dfrac{\gamma*(b-mean)}{\sqrt{var}}+\beta)
$$

$$
W_{fused}=\dfrac{\gamma*W}{\sqrt{var}}
$$

$$
B_{fused}=\dfrac{\gamma*(b-mean)}{\sqrt{var}}+\beta
$$

​		BN本来就是对输入x进行操作，训练后得到的x的均值方差以及gamma和beta，直接代入卷积中的x

2.conv3x3与conv1x1合并

​		1x1卷积可保留当前权重到3x3中心，其余位置为0的卷积核

3.恒等映射层合并

​		恒等映射层输入等于输出，因此只需要将对应通道的权重置为1，其余通道的卷积权重置为0，类似步骤2，当前通道中心为1，其余位置0，其余通道全0

