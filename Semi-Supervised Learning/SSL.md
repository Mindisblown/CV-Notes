基础知识：

**信息量**

​		任何事件都会承载着一定的信息量，无论是已发生还是未发生的事件。信息量与事件发生的概率相关，事件发生的概率越小，其携带的信息量越大(太阳从东方升起，大家都知道的信息，因此信息量不大)。

​		因此很容易推导出对事件的概率求log来反应信息量，但是log是单调递增函数，-log的函数与log都经过(1,0)点且关于x轴对称，并且概率通常在0-1之间，经过-log映射到(正无穷,0)，即满足概率越大，信息量越小

**熵-Entropy**

​		如果将事件的所有可能性列出来，那么可以得到这个事件的期望(均值)，而熵=信息量的期望，则熵可定义为
$$
H(x)=-\sum_{i=1}^{n}{p(x_{i})log(p(x_{i}))}
$$
​		可知事件的熵与其发生概率与信息量有关，信息量再大，发生的概率小，其熵也会小。对于目标分类中是与不是的判断熵可扩展为
$$
H(x)=-p(x)log(p(x))-(1-p(x))log(1-p(x))
$$
**相对熵(KL散度)**

​		相对熵用于衡量两个概率分布的差异，其定义为
$$
D_{KL}(p||q)=\sum_{i=1}^{n}p(x_{i})log\dfrac{p(x_{i})}{q(x_{i})}
$$
​		由公式可知，两个分布越近时，KL散度趋向于0。Jensen不等式证明KL散度是大于0的非负数。P(x)为真实分布，Q(x)为近似分布

**交叉熵-CrossEntropy**

​		扩展KL散度公式得到
$$
DK(p||q)=\sum_{i=1}^{n}p(x_{i})log(p(x_{i}))-\sum_{i=1}^{n}p(x_{i})log(q(x_{i}))
$$
​		回顾熵定义中，扩展的前半部分恰好就是p的熵，而后半部分为交叉熵
$$
H(p,q)=-\sum_{i=1}^{n}p(x_{i})log(q(x_{i}))
$$
​		在网络训练时，通常度量GT Label与Predict之间的差距，而GT Label的熵必定是恒定的，因此只需关注后半部分即可
$$
D(p||q)=H(p,q)-H(p)
$$
**半监督主流思想**

​		Consistency-Based：一致性原则，输入x与经过噪音扰动的x+noise输入同一个网络，它们的输出应该是Invariant

1.Unsupervised Data Augmentation for Consistency Training-UDA

​		使用高质量的数据增强来替代噪音嵌入的方式，通过这种方式使模型对噪音不敏感，那么模型不会过度拟合部分噪音数据，使得模型隐藏空间更加平滑，并且一致性Loss的约束下，也会将标签信息从已标注数据传播到未标记数据

~~~python
"""
Part A:
label data ---> x ---> model ---> x_output
		|-----> GT Label y 

supervised loss = CE_Loss(x_output,y) 

Part B:
ublabel data ---> x ---> model ---> x_output
	|---> augment ---> x_aug ---> model ---> x_aug_output
    
unsupervised loss = KL(x_output, x_aug_output)
利用KL散度求原x和经过数据增强后x_aug的分布，根据Consistency原则可知两分布应一致

Final Loss = supervised loss + unsupervised loss
"""
~~~

