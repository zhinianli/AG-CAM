import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y, attn_weight = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weight


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=1, res_conv=True,
                                          groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t, attn_weight = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t, attn_weight


class Net(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, num_classes=81, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,  # todo
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, self.num_classes - 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Conv2d(1024, self.num_classes - 1, kernel_size=3, stride=1, padding=1)

        # AGF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_channel = 1024
        self.key_channel, self.query_channel, self.value_channel = 512, 512, 1024
        self.f_key = nn.Conv2d(self.in_channel, self.key_channel, 1)
        self.f_query = nn.Conv2d(self.in_channel, self.query_channel, 1)

        # Stem stage
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0], )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)

        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)

        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.fin_stage = fin_stage

        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(256, 128, 1, bias=False)
        self.side3 = nn.Conv2d(512, 256, 1, bias=False)
        self.side4 = nn.Conv2d(1024, 512, 1, bias=False)

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def agf(self, sem_feature, attention):
        batch_size = sem_feature.size(0)
        value = sem_feature.view(batch_size, 1024, -1)
        value = value.permute(0, 2, 1)

        query = self.f_query(sem_feature).view(batch_size, 512, -1)
        query = query.permute(0, 2, 1)

        key = self.f_key(sem_feature).view(batch_size, 512, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channel ** -.5) * sim_map

        sim_map = sim_map * attention

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channel, *sem_feature.size()[2:])

        fuse = self.gamma * context + sem_feature

        return fuse

    def get_seed(self, norm_cam, feature):
        n, c, h, w = norm_cam.shape

        # iou evalution
        seeds = torch.zeros((n, h, w, c)).cuda()
        feature_s = feature.view(n, -1, h * w)
        feature_s = feature_s / (torch.norm(feature_s, dim=1, keepdim=True) + 1e-5)
        correlation = F.relu(torch.matmul(feature_s.transpose(2, 1), feature_s), inplace=True).unsqueeze(1)

        cam_flatten = norm_cam.view(n, -1, h * w).unsqueeze(2)  # [n,21,1,h*w]
        inter = (correlation * cam_flatten).sum(-1)
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter / union).view(n, self.num_classes, h, w)  # [n,21,h,w]
        miou[:, 0] = miou[:, 0] * 0.5
        belonging = miou.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n, h, w, 1), 1).permute(0, 3, 1, 2).contiguous()
        return seeds, correlation

    def get_prototype(self, seeds, feature):
        n, c, h, w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1)
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1, c, h, w), (1, 1)).view(n, self.num_classes, c, 1, 1)
        return prototype

    def reactivate(self, prototype, feature):
        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2))
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam

    def forward(self, x):
        B, C, H, W = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        attn_weights = []

        # state 1
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t, attn_weight = self.trans_1(x_t)
        attn_weights.append(attn_weight)
        state1 = x

        for i in range(2, 13):
            x, x_t, attn_weight = eval('self.conv_trans_' + str(i))(x, x_t)
            attn_weights.append(attn_weight)
            if i == 4:
                state2 = x
            if i == 8:
                state3 = x
            if i == 12:
                state4 = x

        sem_feature = x
        h, w = sem_feature.size(2), sem_feature.size(3)

        # F0 ~ F3
        side1 = self.side1(state1.detach())
        side2 = self.side2(state2.detach())
        side3 = self.side3(state3.detach())
        side4 = self.side4(state4.detach())

        x_patch = x_t[:, 1:]
        n, _, _ = x_patch.shape

        # DSIE
        # AM
        attn_mid = attn_weights[4:10]
        attn_mid = torch.stack(attn_mid)
        attn_mid = torch.mean(attn_mid, dim=2)
        attn_no_cls = attn_mid.sum(0)[:, 1:, 1:]
        attention_mid = attn_no_cls.detach()

        # AS
        attn_shallow = attn_weights[0:5]
        attn_shallow = torch.stack(attn_shallow)
        attn_shallow = torch.mean(attn_shallow, dim=2)
        attention_shallow = attn_shallow.sum(0)[:, 1:, 1:]

        # AD
        attn_deep = attn_weights[10:12]
        attn_deep = torch.stack(attn_deep)
        attn_deep = torch.mean(attn_deep, dim=2)
        attention_deep = attn_deep.sum(0)[:, 1:, 1:]

        # Interaction
        side1 = F.interpolate(side1 / (torch.norm(side1, dim=1, keepdim=True) + 1e-5), side4.shape[2:], mode='bilinear')
        side1_flat = side1.flatten(start_dim=2)
        side1_att = torch.einsum('nhw,ncw->nch', attention_deep, side1_flat).reshape([n, -1, 32, 32])
        side4_flat = side4.flatten(start_dim=2)
        side4_att = torch.einsum('nhw,ncw->nch', attention_shallow, side4_flat).reshape([n, -1, 32, 32])

        hie_fea_att = torch.cat(
            [side1_att,
             F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side4.shape[2:], mode='bilinear'),
             F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side4.shape[2:], mode='bilinear'),
             side4_att],
            dim=1)

        # AGF
        fuse = self.agf(sem_feature, attention_mid)

        # trans classification
        x_t = self.trans_norm(x_t)
        trans_logits = self.trans_cls_head(x_t[:, 0])

        # conv classification
        cam = self.conv_cls_head(fuse)
        conv_logits = self.pooling(cam).flatten(1)

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side4.shape[2:], mode='bilinear', align_corners=True)

        cam_flat = norm_cam.flatten(start_dim=2)

        # SRE
        seeds, _ = self.get_seed(norm_cam.clone(), fuse.clone())
        prototypes = self.get_prototype(seeds, hie_fea_att)
        aux_cam = self.reactivate(prototypes, hie_fea_att)

        final_cam = torch.einsum('nhw,ncw->nch', attention_mid, cam_flat).reshape([n, self.num_classes, h, w])

        return {"score_cnn": conv_logits, "score_trans": trans_logits, 'aux_cam': aux_cam,
                "cam": norm_cam, "final_cam": final_cam}
