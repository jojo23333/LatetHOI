from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr, scatter

import einops
import pointops

DEBUG = False

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError

def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().to(offset.device)

def batch2offset(batch):
    assert (batch[1:] >= batch[:-1]).all()
    return torch.cumsum(batch.bincount(), dim=0).long()

class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True
                 ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        self.linear_p_bias = nn.Sequential(
            nn.Linear(3, embed_channels),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        
        peb = self.linear_p_bias(pos)
        relation_qk = relation_qk + peb
        value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index) \
            if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + feat
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint
            )
            self.blocks.append(block)

    def forward(self, points, reference_index=None):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        if reference_index is None:
            reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points, reference_index


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_size,
                 bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 groups,
                 bias=True,
                 skip=True,
                 backend="map"
                 ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        self.groups = groups
        assert self.backend in ["map", "interp", "ca"]

        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels, bias=bias),
                                       PointBatchNorm(out_channels),
                                       nn.ReLU(inplace=True))
        if self.backend == 'ca':
            self.key = nn.Sequential(nn.Linear(out_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True))
            self.query = nn.Sequential(nn.Linear(out_channels, out_channels, bias=bias),
                                    PointBatchNorm(out_channels),
                                    nn.ReLU(inplace=True))
            self.linear_p_bias = nn.Sequential(
                    nn.Linear(3, out_channels),
                    PointBatchNorm(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_channels, out_channels),
            )
            self.weight_encoding = nn.Sequential(
                nn.Linear(out_channels, groups),
                PointBatchNorm(groups),
                nn.ReLU(inplace=True),
                nn.Linear(groups, groups)
            )
            self.softmax = nn.Softmax(dim=1)

    def forward(self, points, skip_points, cluster, reference_index=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "ca":
            skip_feat = self.proj_skip(skip_feat)
            lres_feat = self.proj(feat)
            hres_feat = lres_feat[cluster] + skip_feat
            reference_index = reference_index[cluster, :]
            
            key, query, value = self.key(lres_feat), self.query(hres_feat), lres_feat
            key = pointops.grouping(reference_index, key, coord, new_xyz=torch.zeros_like(skip_coord), with_xyz=True)
            value = pointops.grouping(reference_index, value, coord, with_xyz=False)
            pos, key = key[:, :, 0:3], key[:, :, 3:]
            pos = pos - skip_coord.unsqueeze(1)
            relation_qk = key - query.unsqueeze(1)

            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

            weight = self.weight_encoding(relation_qk)
            weight = self.softmax(weight)
            mask = torch.sign(reference_index + 1)
            weight = torch.einsum("n s g, n s -> n s g", weight, mask)
            value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
            feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
            feat = einops.rearrange(feat, "n g i -> n (g i)")
            if self.skip:
                feat = feat + skip_feat
        elif self.backend == "map":
            feat = self.proj(feat)[cluster] 
            if self.skip:
                feat = feat + self.proj_skip(skip_feat)
        else:
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
            if self.skip:
                feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 embed_channels,
                 groups,
                 depth,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 unpool_backend="ca",
                 ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
            groups=groups
        )
        

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points, skip_points, cluster, reference_index):
        lr_reference_index, hr_reference_index = reference_index
        points = self.up(points, skip_points, cluster, lr_reference_index)
        return self.blocks(points, hr_reference_index)


class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


class  PointTransformerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 patch_embed_depth=1,
                 patch_embed_channels=32,
                 patch_embed_groups=4,
                 patch_embed_neighbours=4,
                 enc_depths=(1, 2, 2),
                 enc_channels=(64, 128, 256),
                 enc_groups=(8, 16, 32),
                 enc_neighbours=(16, 16, 16),
                 dec_depths=(1, 1, 1),
                 dec_channels=(64, 128, 256),
                 dec_groups=(8, 16, 32),
                 dec_neighbours=(8, 8, 8),
                 grid_sizes=(0.02, 0.06, 0.15),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], in_channels)
        )

    def forward_encoder(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        points, reference_index = points
        skips = [[points, reference_index]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            points, reference_index = points
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points, reference_index])  # record points info of current stage
        coord, feat, offset = points
        batch = offset2batch(offset)
        max_pooled_x = scatter(feat, batch.to(torch.int64), dim=0, reduce="max")
        return max_pooled_x, skips
        
    
    def forward_decoder(self, new_feat, skips):
        points, lr_reference_index = skips.pop(-1)  # unpooling points info in the last enc stage
        batch = offset2batch(points[2])
        points[1] = points[1] + new_feat[batch, :]
        for i in reversed(range(self.num_stages)):
            skip_points, hr_reference_index, cluster = skips.pop(-1)
            points, _ = self.dec_stages[i](points, skip_points, cluster, (lr_reference_index, hr_reference_index))
            lr_reference_index = hr_reference_index
        coord, feat, offset = points
        seg_logits = self.seg_head(feat)
        return seg_logits

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        points, reference_index = points
        skips = [[points, reference_index]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            points, reference_index = points
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points, reference_index])  # record points info of current stage

        points, lr_reference_index = skips.pop(-1)  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, hr_reference_index, cluster = skips.pop(-1)
            points, _ = self.dec_stages[i](points, skip_points, cluster, (lr_reference_index, hr_reference_index))
            lr_reference_index = hr_reference_index
        coord, feat, offset = points
        seg_logits = self.seg_head(feat)
        return seg_logits