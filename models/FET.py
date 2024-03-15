import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quadtree_attention import QuadtreeAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops
from .module import PatchEmbed


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
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
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dwconv(x, H, W)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class QuadtreeBlock(nn.Module):

    def __init__(self, dim, num_heads, topks, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, scale=1, attn_type='B'):

        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = QuadtreeAttention(
            dim,
            num_heads=num_heads, topks=topks, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, scale=scale, attn_type=attn_type)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'init'):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(target), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, source, source,
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y)


class FET(nn.Module):
    def __init__(self,block_type, topks, nhead, d_model, cross_attention):
        super(FET, self).__init__()

        self.block_type = block_type
        self.d_model = d_model
        self.nhead = nhead
        self.cross_attention = cross_attention
        if block_type == 'quadtree':
            encoder_layer = QuadtreeBlock(d_model, nhead, topks, scale=3)
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.cross_attention))])
        elif block_type == 'linear':
            encoder_layer = EncoderLayer(d_model, nhead)
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.cross_attention))])
            self._reset_parameters()

        self.patch_embed = PatchEmbed(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, feat="ref"):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W]
            src_feature(torch.Tensor): [N, C, H, W]
        """

        assert ref_feature is not None

        if feat == "ref":  # only self attention layer

            assert self.d_model == ref_feature.size(1)
            _, _, H, W = ref_feature.shape

            # ref_feature = einops.rearrange(self.pos_encoding(ref_feature), 'n c h w -> n (h w) c')
            ref_feature = self.patch_embed(ref_feature)
            ref_feature_list = []
            for layer, name in zip(self.layers, self.cross_attention):  # every self attention layer
                if self.block_type == "quadtree":
                    if name == 'False':
                        ref_feature = layer(ref_feature, ref_feature, H, W)
                        ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))
                elif self.block_type == 'linear':
                    if name == 'False':
                        ref_feature = layer(ref_feature, ref_feature)
                        ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))
                else:
                    raise ValueError("Wrong block type")
            return ref_feature_list

        elif feat == "src":

            assert self.d_model == ref_feature[0].size(1)
            _, _, H, W = ref_feature[0].shape

            ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c') for _ in ref_feature]

            # src_feature = einops.rearrange(self.pos_encoding(src_feature), 'n c h w -> n (h w) c')
            src_feature = self.patch_embed(src_feature)
            for i, (layer, name) in enumerate(zip(self.layers, self.cross_attention)):
                if self.block_type == "quadtree":
                    if name == 'False':
                        src_feature = layer(src_feature, src_feature, H, W)
                    elif name == 'True':
                        src_feature = layer(src_feature, ref_feature[i // 2], H, W)
                    else:
                        raise KeyError
                elif self.block_type == "linear":
                    if name == 'False':
                        src_feature = layer(src_feature, src_feature)
                    elif name == 'True':
                        src_feature = layer(src_feature, ref_feature[i // 2])
                    else:
                        raise KeyError
                else:
                    raise ValueError("Wrong block type")
            return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H)
        else:
            raise ValueError("Wrong feature name")


class GFF(nn.Module):
    def __init__(self,base_channels, block_type, topks, nhead, d_model, cross_attention):

        super(GFF, self).__init__()

        self.FET = FET(block_type, topks, nhead, d_model, cross_attention)

        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Sequential(nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1))
        self.smooth_2 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 1 ,kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(base_channels * 1, base_channels * 1, kernel_size=3, stride=1, padding=1))

    def up_concat(self, x, y):
        _, _, H, W = y.size()
        return torch.cat([F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True), y], dim=1)

    def forward(self, features):
        for nview_idx, feature_multi_stages in enumerate(features):
            if nview_idx == 0:  # ref view
                ref_fea_t_list = self.FET(feature_multi_stages["stage_3"].clone(), feat="ref")
                feature_multi_stages["stage_3"] = ref_fea_t_list[-1]
                feature_multi_stages["stage_2"] = self.smooth_1(
                    self.up_concat(self.dim_reduction_1(feature_multi_stages["stage_3"]),
                                       feature_multi_stages["stage_2"]))
                feature_multi_stages["stage_1"] = self.smooth_2(
                    self.up_concat(self.dim_reduction_2(feature_multi_stages["stage_2"]),
                                       feature_multi_stages["stage_1"]))

            else:  # src view
                feature_multi_stages["stage_3"] = self.FET([_.clone() for _ in ref_fea_t_list],
                                                           feature_multi_stages["stage_3"].clone(), feat="src")
                feature_multi_stages["stage_2"] = self.smooth_1(
                    self.up_concat(self.dim_reduction_1(feature_multi_stages["stage_3"]),
                                       feature_multi_stages["stage_2"]))
                feature_multi_stages["stage_1"] = self.smooth_2(
                    self.up_concat(self.dim_reduction_2(feature_multi_stages["stage_2"]),
                                       feature_multi_stages["stage_1"]))

        return features