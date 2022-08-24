# IOU

IOULoss = 1 - IOU

# GIOU

​		当GT box与Predict box无重叠，IOU始终为0且无法优化

​		GIOU在此基础上增加惩罚项，GIOU = IOU - (两box并集外接矩形空隙面积  / 并集外接矩形面积)，GIOULoss=1-GIOU

​		当两box无重叠时，IOU=0，但GIOU不为0

# DIOU

​		当GT box与Predict box无限接近，GIOU退化成IOU。并且GIOU的训练过程中，GIOU首先倾向于增大Predict box的大小来增大与GT box之间的交集，最后通过IOU来引导最大化重叠区域，GIOU很大程度上依赖IOU项，因此需要更多的迭代次数完成收敛。

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