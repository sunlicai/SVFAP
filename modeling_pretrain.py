import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table, SpatialAttention, CSBlock
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange, repeat


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_svfap_base_patch16_160',
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False,
                 stage_depths=(12, 6, 3),  # number of blocks in each stage
                 num_bottleneck_tokens=(8,),  # number of bottleneck tokens in spatial bottleneck transformer
                 temporal_kernel_size=2,  # kernel size for 1D temporal conv
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # model architecture: TPSBT
        depth = sum(stage_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.stage_depths = stage_depths

        ## Temporal Pyramid: kernel size of 1D conv
        self.temporal_kernel_size = temporal_kernel_size
        ## Spatial Bottleneck Transformer: number of bottleneck tokens for summarizing important information in spatial attention
        if len(num_bottleneck_tokens) == 1:
            self.num_bottleneck_tokens = num_bottleneck_tokens * (len(stage_depths) - 1)  # repeat
        else:
            assert len(num_bottleneck_tokens) == (len(stage_depths) - 1)
            # support different numbers of bottleneck tokens in different stages
            self.num_bottleneck_tokens = num_bottleneck_tokens

        self.blocks = nn.ModuleList([])
        self.temporal_downsamples = nn.ModuleList([])
        self.spatial_downsamples = nn.ModuleList([])

        block_idx = 0
        for stage_idx, stage_depth in enumerate(stage_depths):
            # 1st stage (standard Transformer block)
            if stage_idx == 0:
                for _ in range(stage_depth):
                    self.blocks.append(
                        Block(
                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], norm_layer=norm_layer,
                            init_values=init_values)
                    )
                    block_idx += 1
            # 2nd, 3rd,... stage (TPSBT block)
            else:
                # Temporal Pyramid (TP)
                self.temporal_downsamples.append(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=temporal_kernel_size, stride=temporal_kernel_size)
                )
                # Spatial Bottleneck Transformer (SBT)
                cur_temporal_seq_len = self.patch_embed.temporal_seq_len // (temporal_kernel_size ** stage_idx)
                ## spatial attention
                self.spatial_downsamples.append(
                    SpatialAttention(
                        dim=embed_dim, num_tokens=self.num_bottleneck_tokens[stage_idx - 1],
                        bottleneck_dim=128, drop=drop_rate, norm_layer=norm_layer, temporal_seq_len=cur_temporal_seq_len
                    )
                )
                ## cross self block
                for _ in range(stage_depth):
                    self.blocks.append(
                        CSBlock(dim=embed_dim, context_dim=embed_dim, num_heads=num_heads,
                                num_cross_heads=num_heads,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[block_idx], norm_layer=norm_layer, init_values=init_values)
                    )
                    block_idx += 1


        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        # model architecture
        block_idx = 0
        multiscale_feature = None
        for stage_idx, stage_depth in enumerate(self.stage_depths):
            # 1st stage (standard Transformer block)
            if stage_idx == 0:
                for _ in range(stage_depth):
                    x_vis = self.blocks[block_idx](x_vis)
                    block_idx += 1
                # store
                multiscale_feature = x_vis
            # 2nd, 3rd,... stage (TPSBT)
            else:
                # Temporal Pyramid (TP)
                cur_temporal_seq_len = self.patch_embed.temporal_seq_len // (self.temporal_kernel_size ** (stage_idx - 1))
                x_vis = rearrange(x_vis, 'b (t hw) c -> (b hw) c t', t=cur_temporal_seq_len)
                x_vis = self.temporal_downsamples[stage_idx - 1](x_vis)
                x_vis = rearrange(x_vis, '(b hw) c t -> b (t hw) c', b=B)

                # Spatial Bottleneck Transformer (SBT)
                x_vis_before_spatial_downsample = x_vis
                # spatial attention
                x_vis = self.spatial_downsamples[stage_idx - 1](x_vis)
                # cross self block
                for _ in range(stage_depth - 1):
                    x_vis = self.blocks[block_idx](x_vis, context=x_vis_before_spatial_downsample)
                    block_idx += 1
                # reverse cross self block for spatial resolution recovering
                x_vis = self.blocks[block_idx](x_vis_before_spatial_downsample, context=x)
                block_idx += 1

                # temporal resolution recovering (i.e, temporal upsample): use simple interpolation
                x_vis_up = rearrange(x_vis, 'b (t hw) c -> b t hw c', t=cur_temporal_seq_len)
                x_vis_up = x_vis_up.repeat_interleave(self.temporal_kernel_size ** stage_idx, dim=1)
                x_vis_up = rearrange(x_vis_up, 'b t hw c -> b (t hw) c')
                # multiscale feature fusion
                multiscale_feature = multiscale_feature + x_vis_up

        # re-assign
        x_vis = multiscale_feature

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_num_heads=12,
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 stage_depths=(12, 6, 3),  # number of blocks in each stage
                 num_bottleneck_tokens=(8,),  # number of bottleneck tokens in spatial bottleneck transformer
                 temporal_kernel_size=2,  # kernel size for 1D temporal conv
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            stage_depths=stage_depths,
            num_bottleneck_tokens=num_bottleneck_tokens,
            temporal_kernel_size=temporal_kernel_size,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token', 'lg_region_tokens'}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x



@register_model
def pretrain_svfap_base_patch16_160(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
