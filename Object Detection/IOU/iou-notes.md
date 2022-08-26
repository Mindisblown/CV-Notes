# IOU

IOULoss = 1 - IOU

​	当预测框与真实框不相交，IOU=0，无法衡量两个box的位置关系(距离、相交方式、重叠方式)

# GIOU

​		当GT box与Predict box无重叠，IOU始终为0且无法优化

​		GIOU在此基础上增加惩罚项，GIOU = IOU - (两box并集外接矩形空隙面积  / 并集外接矩形面积)，GIOULoss=1-GIOU

​		当两box无重叠时，IOU=0，但GIOU不为0

​		GIOU取值在[-1, 1]之间

# DIOU

​		当GT box与Predict box无限接近，GIOU退化成IOU。并且GIOU的训练过程中，GIOU首先倾向于增大Predict box的大小来增大与GT box之间的交集，最后通过IOU来引导最大化重叠区域，GIOU很大程度上依赖IOU项，GIOU理论上来说优化的是面积，因此需要更多的迭代次数完成收敛。

​		DIOU直接最小化预测框与目标框之间的归一化距离，DIOU不仅考虑了重叠面积还考虑了中心店距离，并没有考虑两box的长宽比。

​		DIOU=IOU-(中心点距离 / 并集外接矩形对角线距离)^2，DIOULoss=1-DIOU

# CIOU

​		CIOU在DIOU的惩罚性基础上增加了一个影响因子av，影响因子考虑了长宽比，a权重系数，v度量长宽比的相似性。
$$
v=\dfrac{4}{\pi^{2}}(arctan\dfrac{w^{gt}}{h^{gt}}-arctan\dfrac{w}{h})^{2}
$$

$$
a=\dfrac{v}{(1-IOU)+v}
$$

CIOU=DIOU-av，CIOULoss=1-CIOU

# EIOU

​		CIOU仅将宽高比作为影响因子，当预测框与真实框宽高比呈现线性比例使，那么惩罚项就无作用，宽和高无法同时增加。

​		在CIOU的基础上将宽高比拆开，提出EIOU，并且加入Focal loss的思想，优化边界框回归任务中的样本不平衡问题，减少与目标框重叠较少的box对回归的优化贡献，使回归专注于高质量box的回归。

​		C_w与C_h是覆盖预测框真实框最小外接矩形的宽度和高度
$$
L_{EIOU}=L_{IOU}+L_{dis}+L{asp}=1-IOU+\dfrac{\rho^{2}(b,b^{gt})}{c^{2}}+\dfrac{\rho^{2}(w,w^{gt})}{C_{w}^{2}}+\dfrac{\rho^{2}(h,h^{gt})}{C_{h}^{2}}
$$
EIOU Loss = IOULoss+中心店损失+宽度损失+高度损失
$$
L_{Focal-EIOU}=IOU^{\gamma}L_{EIOU}
$$

# SIOU

​		前面工作的基础上引入角度，box的角度能够影响回归

​		SIOU = 距离损失(角度+距离)+形状损失+IOU损失

​		**Angle Cost**


$$
\Lambda=1-2*sin^{2}(arcsin(x)-\dfrac{\pi}{4})
$$

$$
x=\dfrac{c_{h}}{\sigma}=sin(\alpha)
$$

$$
\sigma=\sqrt{(b_{c_{x}^{gt}}-b_{c_{x}})^{2}+(b_{c_{y}^{gt}}-b_{c_{y}})^{2}}
$$

$$
c_{h}=max(b_{cy}^{gt},b_{c_{y}})-min(b_{cy}^{gt}-b_{c_{y}})
$$

$$
\Lambda=1-2sin^{2}(arcsin(sin(\alpha))-\dfrac{\pi}{4})
=1-2sin^{2}(\alpha-\dfrac{\pi}{4})
=cos^{2}(\alpha-\dfrac{\pi}{4})-sin^{2}(\alpha-\dfrac{\pi}{4})
=cos(2a-\dfrac{\pi}{4})=sin(2\alpha)
$$

​		当alpha=0，Lambda=0，alpha=pi/4，Lambda=1。两中心点连线构成的矩形，必定有alpha与beta两个角度。当alpha<=pi/4收敛过程最小化alpha，否则优化beta

​		**Distance Cost**
$$
\Delta=\sum_{t=xy}(1-e^{-\gamma\rho_{t}})
$$

$$
p_{x}=(\dfrac{b_{cx}^{gt}-b_{cx}}{c_{w}})^{2}
$$

$$
p_{y}=(\dfrac{b_{cy}^{gt}-b_{cy}}{c_{h}})^{2}
$$

$$
\gamma=2-\Lambda
$$

​		当alpha=0，gamma=2，delta减小，distance cost降低；alpha=pi/4，gamma=1，distance cost增大。角度与距离结合。

​		**Shape Cost**
$$
\Omega=\sum_{t=w,h}(1-e^{-w_{t}})^{\theta}
$$

$$
W_{w}=\dfrac{|w-w^{gt}|}{max(w,w^{gt})}
$$

$$
W_{h}=\dfrac{|h-h^{gt}|}{max(h,h^{gt})}
$$

​		theta控制着对shape cost的关注程度，作者将遗传算法用于每个数据集来计算theta。


$$
SIOULoss=1-IOU+\dfrac{\Delta+\Omega}{2}
$$
