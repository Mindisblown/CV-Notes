from contextlib import asynccontextmanager
import torch
import torch.nn as nn


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        # 初始化
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                # 仅计算所有正样本的回归loss
                # sigmoid使值在0-1之间 *2-0.5使得取值在[-0.5, 1.5]之间
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # wh sigmoid的output位于负无穷与正无穷之间, 通过该公式变为0-4之间sigmoid(0)*2^x = 1 regardless of x
                # 相对于anchor的倍数
                # 不同于v3, 没有采用exp操作, 而是直接乘上anchors[i], 这种做法使得输出没有边界exp - (ps[;, 2:4] * anchors[i])
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # 给定target bbox标签信息
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            # 正负样本一起计算
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        
        """
        p是一个列表，长度为3，保存所有anchor层的输出
        p[0].shape [1, 3, x, x, 6]
        p[1].shape [1, 3, x, x, 6]
        p[2].shape [1, 3, x, x, 6]
        [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
        """
        # input targets(image_index,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # anchor数量, targets数量
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        # 生成[1,3]列数据，通过view变换成[3,1]，最后通过repeat转换成[3, nt]
        # 第一行nt个0，第二行nt个1，第三行nt个2
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        # targets原本维度[标签个数, 6]，通过repeat扩张为[3, 标签个数, 6]，与ai在axis=2上进行cat [3, 标签个数, 7]
        # 用ai来标记targets是属于哪个anchor的targets，添加anchor的索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # 中心偏移，衡量targets的中心点距离哪个grid更近
        # grid偏移，自身，右，上，左，下
        # ----------|--------|--------|
        # |         | (0, -1)|        |
        # ----------|--------|--------|
        # | (-1, 0) | (0, 0) | (1, 0) |
        # ----------|--------|--------|
        # |         | (0, 1) |        |
        # ----------|--------|--------|
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 遍历处理anchor
        for i in range(self.nl):
            anchors = self.anchors[i]
            # 2:6 = pi[i].shape [[3, 2, 3, 2]] 取特征图的whwh
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            # targets是归一化后的坐标，gain[2:6]=gain[1, 1, w, h, w, h, 1]
            # targets*gain表示将坐标映射到对应特征图的尺寸
            t = targets * gain
            
            if nt:
                # 开始匹配
                # t[:, :, 4:6]取wh / 当前组anchor
                r = t[:, :, 4:6] / anchors[:, None]  # wh的比例
                # GT bbox与anchor的wh比例超过阈值则过滤，注意倒数，则有w1/w2, w2/w1, h1/h2, h2/h1
                # max返回最大值与其索引，[0]表示取wh比例的最大值
                """
                v5不同点
                yolov3 v4的筛选方法: wh_iou  GT与anchor的wh_iou超过一定的阈值就是正样本
                """
                # 当宽高比大于4时，表明该targets的box与当前anchor尺寸不对,则为负样本,V5中没有忽略样本 
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 舍弃大于anchor_t的targets

                # Offsets
                gxy = t[:, 2:4]  # targets - grid的中心点，以图像左上角为原点
                gxi = gain[[2, 3]] - gxy  # 得到target中心点相对于右下角的坐标，相当于以右下角为坐标原点  gain[[2, 3]]为当前feature map的wh
                
                """
                %1的含义
                    假设中心点坐标为(1.84134, 62.55986)，那么是以其左上角坐标为原点即(1, 62)
                    代码中直接%1取坐标的余数0.84134, 0.55986就能得到中心点相对左上角的偏移
                >1的含义
                    如果小于1的只有坐标原点为左上角，无法找到左上右下，并且在扩张时会溢出边界(超出跨网格预测的偏移值)
                """
                # 中心点xy大于1，y距离网格上面和x距离网格左边距离小于0.5
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # 中心点xy大于1，y距离网格下面和x距离网格右边距离小于0.5
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # 默认选取的anchor数量<=3，左上和右下+本身
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j] # *3，加入附近的两个
                # 对于前面判断得到距离上下左右哪个近，再加上对应的偏移，取每个框的偏移
                # 创建维度与gxy一致的全0矩阵，并加上off
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 跨网格预测，故xy预测输出不再是0-1，而是-1～1，加上off*g=offset偏移，则为-0.5-1.5

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # 依据offsets向周围扩张两个grid，因为offsets中有+0.5， -0.5， 0
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch