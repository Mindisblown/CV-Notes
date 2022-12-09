1.Squeeze and Excitation Network CVPR18

​		最经典的通道注意力SENet，学习channel之间的相关性。对于卷积输出后的特征F1，得到一个和通道数一样的1D向量[1, 1, C]作为每个通道得分，最终将得分图与特征图相乘。

​		Squeeze：[C, H, W]特征图进行全局平均池化得到[1, 1, C]特征图，这个特征图具有全局感受野。Excitation：使用FC层，对Squeeze之后的结果进行非线性变换。

​		分类模型一般添加到一个block结束的位置，对一个block的信息进行refine。检测模型一般添加到backbone的stage、block等结束位置。

