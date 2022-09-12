# FCOS

**Label Assignment**

​		anchor point替代anchor box，如果一个location(x, y)落在GT box中，那么它就为正样本点，该点需要回归(l, t, r, b)，即该点相比GT box的左、上、右、下得距离。

​		当两个GT box重叠时，重叠区域的location必定会被共享。直接使用最小区域作为回归目标。并且为了进一步减少尺度差异大的物体重叠，每一个特征层引入最大距离参数m_i，对于一个location(x, y)满足条件max(l, t, r, b)>m_i或者max(l, t, r, b)<m_i-1，在该层将其视为负样本不进行回归。

**Centerness**

​		anchor point的方式把中心点的区域扩充到整个物体的边框，会带来很多低质量的样本框，而centerness则是对回归框的约束，使其与GT box重叠。
$$
centerness=\sqrt{\dfrac{min(l,r)}{max(l,r)}*\dfrac{min(t,b)}{max(t,b)}}
$$
​		centerness越接近于1，表明越靠近GT box中心。

​		训练时，centerness采用BCELoss，在实际推理时，centerness的值乘上类别分数做为nms的排序参考。