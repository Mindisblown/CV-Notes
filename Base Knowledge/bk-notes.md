**正则化与权重衰减**

​		正则化的目的就是为了让权重衰减到更小的值，从而减少过拟合。因此权重衰减也叫正则化。具体实现就是再损失函数的后面加上一个正则化项
$$
L=L+\dfrac{\lambda}{2n}\sum_{w}w^{2}
$$
​		依据链式法则
$$
\dfrac{\partial{L}}{\partial{w}}=\dfrac{\partial{L}}{\partial{w}}+\dfrac{\lambda}{n}w
$$
​		BP时
$$
w\rightarrow{n}-\eta{\dfrac{\partial{L}}{\partial{w}}}-\dfrac{\eta\lambda}{n}w=(1-\dfrac{\eta\lambda}{n})w-\eta{\dfrac{\partial{L}}{\partial{w}}}
$$
​		w的前系数小于1，在更新时先减小w的值再去更新。

​		权重衰减通过控制w在较小的范围波动，使得决策边界更加平滑(越大的取值范围，越容易造成极端的拟合情况)。

**交叉熵**

​		nn.CrossEntropyLoss=nn.functional.cross_entropy()

​		常用于C类别的分类问题，定义为：
$$
l(x,y)=L=(l_1,...,l_N)^T,l_n=-w_{y_n}log\dfrac{exp(x_n, y_n)}{\sum_{c=1}^{C}exp(x_n,c)}
$$
​		存在两种形式，mean与sum，mean在sum的基础上除以batch数，默认为mean。

​		CE=LogSoftmax+NLLLoss，由定义可知对softmax取log后，softmax后的值位于[0-1]之间，取Log后位于[负无穷-0]之间，因此需要加负号转为正值。

**BCELoss**

​		nn.BCEloss=nn.functional.binary_cross_entropy()

​		用于二分类问题，定义为：
$$
l(x,y)=L=(l_1,...,l_N)^T,l_n=-w_n[y_nlogx_n+(1-y_n)log(1-x_n)]
$$
​		存在mean与sum两种形式，默认为mean。

​		由定义可知，x_n必须位于[0-1]之间，假设target=[0,1,0]，input=[0.1, 0.8, 0.8]，对于index=0时，loss=-(1-0)log(1-0.1)=-log(0.9)，log(1)=0，loss趋向较小值。index=1，loss=-log(0.8)，同样较小。index=2，loss=-(1-0)log(1-0.8)=-log(0.2)，此时loss较大。

**BCEWithLogitsLoss**

​		nn.BCEWithLogitsLoss=nn.functional.binary_cross_entropy_with_logits()

​		将Sigmoid加入到BCELoss中，BCELoss中的x变为Sigmoid(x)