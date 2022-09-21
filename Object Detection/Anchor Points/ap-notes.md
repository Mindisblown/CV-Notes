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

# CenterNet-Object as points

​		无需nms，其anchor为一个点，这个点只有location的信息，不存在依据iou划分正负样本。

​		1.输入图片转换到heatmap，根据GT box的信息得到中心点坐标(余数时向下取整)；

​		2.依据box的大小来计算高斯圆半径，分为三种情况，预测框包含标注框，标注框包含真实框，标注框真实框重叠但不包含。依据标注框与预测框的两个角点以r为半径的圆，将iou计算公式展开，得到二元一次方程，可得a、b、c，r=(-b+\sqrt(b^2-4ac))/2a得到半径；

​		3.根据中心点与r半径来计算高斯值，半径r的目的在于保证预测points在这个半径内。

​		最终的loss构成为关键点heatmap loss + 中心点偏移offset loss + 长宽尺寸size loss。

​		在推理时判断当前点周围8个点的大小，找到大于周围8个点的点作为points，相当于一种nms。