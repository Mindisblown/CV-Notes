# Decoupled Mixup for Generalized Visual Recognition

​		不同于mixup直接对图片进行操作，通过decoupled操作从style、context、frequency三个角度思考mixup，提出Style-based Decoupled-Mixup(与MixStyle类似)，Context-aware Decoupled-Mixup(通过C2AM方法来分离前景背景)，Frequency-aware Decoupled-Mixup(傅里叶变换去除高频中的texture信息，高频是噪音多的区域)

​		https://github.com/HaozheLiu-ST/NICOChallenge-OOD-Classification

