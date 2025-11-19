import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import to_2tuple
from .feature_fusion import DAF, AFF, iAFF

def get_linear_layer(method='', *args, **kwargs):
    if 'lora' in method:
        from .model_utilities_adapt import Linear as LinearLoRA
        kwargs.update(kwargs.get('linear_kwargs', {}))
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return LinearLoRA(*args, **kwargs)
    else:
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return nn.Linear(*args, **kwargs)
    

def get_conv2d_layer(method='', **kwargs):
    if 'lora' in method:
        from .model_utilities_adapt import Conv2d as Conv2dLoRA
        kwargs.update(kwargs.get('conv_kwargs', {}))
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return Conv2dLoRA(**kwargs)
    else:
        kwargs = {k: v for k, v in kwargs.items() if '_kwargs' not in k}
        return nn.Conv2d(**kwargs)


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X
    

class CrossStitch(nn.Module):
    def __init__(self, feat_dim):

        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(feat_dim, 2, 2).uniform_(0.1, 0.9)
            )
    
    def forward(self, x, y):
        if x.dim() == 4:
            equation = 'c, nctf -> nctf'
        elif x.dim() == 3:
            equation = 'c, ntc -> ntc'
        else:
            raise ValueError('x must be 3D or 4D tensor')
        x = torch.einsum(equation, self.weight[:, 0, 0], x) + \
            torch.einsum(equation, self.weight[:, 0, 1], y)
        y = torch.einsum(equation, self.weight[:, 1, 0], x) + \
            torch.einsum(equation, self.weight[:, 1, 1], y)
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False,
                pool_size=(2,2), pool_type='avg'):
        
        super(ConvBlock, self).__init__()

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_size)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        else:
            raise Exception('pool_type must be avg or max')
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=bias)
                            
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=bias)
                            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.,
                 **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = get_linear_layer(in_features=in_features, 
        #                             out_features=hidden_features,
        #                             **kwargs)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.fc2 = get_linear_layer(in_features=hidden_features, 
        #                             out_features=out_features,
        #                             **kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x):

        x = self.fc2(self.drop(self.act(self.fc1(x))))

        x = self.drop(x)
        return x
    

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True, patch_stride=16, padding=True,
                 cfg_adapt={}, cfg_fusion={}):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        if padding:
            padding = ((patch_size[0] - patch_stride[0]) // 2, 
                       (patch_size[1] - patch_stride[1]) // 2)
        else:
            padding = 0

        # self.proj = get_conv2d_layer(in_channels=in_chans, out_channels=embed_dim, 
        #                              kernel_size=patch_size, stride=patch_stride, 
        #                              padding=padding, **cfg_adapt)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.enable_fusion = cfg_fusion.get('enable', False)
        self.fusion_type = cfg_fusion.get('type')
        if self.enable_fusion and self.fusion_type in ['daf_2d','aff_2d','iaff_2d']:
            self.mel_conv2d = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size[0], patch_size[1]*3), 
                                        stride=(patch_stride[0], patch_stride[1] * 3), padding=padding)
            if self.fusion_type == 'daf_2d': self.fusion_model = DAF()
            elif self.fusion_type == 'aff_2d': self.fusion_model = AFF(channels=embed_dim, type='2D')
            elif self.fusion_type == 'iaff_2d': self.fusion_model = iAFF(channels=embed_dim, type='2D')    

    def forward(self, x, longer_list_idx=[]):

        H, W = x.shape[2], x.shape[3]
        assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.enable_fusion and self.fusion_type in ['daf_2d','aff_2d','iaff_2d']:
            global_x = x[:, :1]
            # global processing
            B, C, H, W = global_x.shape # (B, 1, F', T')
            global_x = self.proj(global_x) # (B, embed_dim, F'', T'')
            TW = global_x.shape[-1]
            if len(longer_list_idx) > 0:
                local_x = x[longer_list_idx, 1:].contiguous()
                B, C, H, W = local_x.shape # (B, 3, F', T')
                local_x = local_x.view(B*C, 1, H, W) # (B*3, 1, F', T')
                local_x = self.mel_conv2d(local_x) # (B*3, embed_dim, F'', T''/3)
                local_x = local_x.view(B, C, *local_x.shape[1:]) # (B, 3, embed_dim, F'', T''/3)
                local_x = local_x.permute(0, 2, 3, 1, 4).contiguous().flatten(3) # (B, embed_dim, F'', T'')
                TB, TC, TH, _ = local_x.shape
                if local_x.size(-1) < TW:
                    local_x = torch.cat([local_x, 
                                         torch.zeros((TB,TC,TH,TW-local_x.size(-1)), 
                                                     device=global_x.device)
                                         ], dim=-1)
                else:
                    local_x = local_x[..., :TW]
                local_x = local_x.to(global_x.dtype)
                global_x[longer_list_idx] = self.fusion_model(global_x[longer_list_idx], local_x)
            x = global_x
        else:
            x = self.proj(x)
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

