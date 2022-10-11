# ATSS(Adaptive Training Sample Selection)

1. 计算每个GT Box和特征层上Anchor Box之间的中心点L2距离；
2. 筛选出距离最近的K个Anchor Box，并与GT Box计算IOU；
3. 计算IOU的均值和方差；
4. 保留IOU大于均值和方差，且中心点位于GT Box内的Anchor Box作为正样本。



# Soft Anchor-Point Object Detection

soft-weighted anchor points

​		针对遮挡、背景混乱的干扰情况。Anchor-point检测器在处理这类场景是会发生注意力偏差，