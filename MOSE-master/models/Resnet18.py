# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # optional early fusion inside feature extraction (hierarchical within-stage)
        self.enable_early_fusion = True
        if self.enable_early_fusion:
            self.layer1 = BlockSeqWithFusion(self.layer1, embed_dim=nf * 1 * block.expansion, num_heads=4)
            self.layer2 = BlockSeqWithFusion(self.layer2, embed_dim=nf * 2 * block.expansion, num_heads=4)
            self.layer3 = BlockSeqWithFusion(self.layer3, embed_dim=nf * 4 * block.expansion, num_heads=4)
            self.layer4 = BlockSeqWithFusion(self.layer4, embed_dim=nf * 8 * block.expansion, num_heads=4)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.simclr = nn.Linear(nf * 8 * block.expansion, 128)
        self.classifier = self.linear

        self.pool = nn.AdaptiveAvgPool2d((1, 1))


class BlockSeqWithFusion(nn.Module):
    """
    Wraps a Sequential of blocks and applies intra-stage progressive attention fusion:
    y_0 = block_0(x)
    y_1 = block_1(y_0 + attn(y_0 -> y_1))
    y_2 = block_2(y_1 + attn([y_0,y_1] -> y_2))
    ...
    """
    def __init__(self, seq: nn.Sequential, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.seq = seq
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def _to_seq(self, fmap: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fmap.shape
        x = fmap.flatten(2).transpose(1, 2)  # (B, HW, C)
        return x

    def _from_seq(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        B, C, H, W = ref.shape
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_feats = []  # list of (B,C,H,W)
        out = x
        for block in self.seq:
            out = block(out)
            if len(prev_feats) > 0:
                # align spatial size by interpolation if needed
                kv_list = []
                for pf in prev_feats:
                    if pf.shape[2:] != out.shape[2:]:
                        pf_resized = F.interpolate(pf, size=out.shape[2:], mode='bilinear', align_corners=False)
                    else:
                        pf_resized = pf
                    kv_list.append(pf_resized)
                kv = torch.stack(kv_list, dim=1)  # (B,Lp,C,H,W)
                B, Lp, C, H, W = kv.shape
                q = self._to_seq(out)                  # (B, HW, C)
                kv = kv.view(B * Lp, C, H, W)
                kv = self._to_seq(kv).view(B, Lp * (H * W), C)  # (B, Lp*HW, C)

                # LayerNorm then project
                qn = self.norm_q(q)
                kvn = self.norm_kv(kv)
                Q = self.proj_q(qn)
                K = self.proj_k(kvn)
                V = self.proj_v(kvn)

                # multi-head split
                h = max(1, int(self.num_heads))
                if (self.embed_dim % h) != 0:
                    h = 1
                def split(x):
                    Bx, Nx, Cx = x.shape
                    d = Cx // h
                    return x.view(Bx, Nx, h, d).permute(0, 2, 1, 3)  # (B,h,N,d)
                Qh, Kh, Vh = split(Q), split(K), split(V)
                scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (Qh.size(-1) ** 0.5)  # (B,h,HW,Lp*HW)
                attn = torch.softmax(scores, dim=-1)
                fused = torch.matmul(attn, Vh)  # (B,h,HW,d)
                fused = fused.permute(0, 2, 1, 3).contiguous().view(out.size(0), -1, self.embed_dim)  # (B,HW,C)
                fused = self.proj_o(fused)
                fused = self._from_seq(fused, out)
                out = out + fused
            prev_feats.append(out.detach())
        return out

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f_train(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)  
        return out

    def forward(self, x: torch.Tensor, use_proj=False):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x)

        if use_proj:
            feature = out
            out = self.simclr(out)
            return feature, out
        else:
            out = self.linear(out)
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.f_train(x)
        return out

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    @property
    def n_params(self):
        return sum(np.prod(p.size()) for p in self.parameters())


def resnet18(nclasses: int, nf: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)


def init_weights(model, std=0.01):
    print("Initialize weights of %s with normal dist: mean=0, std=%0.2f" % (type(model), std))
    for m in model.modules():
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()
