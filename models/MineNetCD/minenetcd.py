# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch UperNet model. Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation."""

from typing import List, Optional, Tuple, Union

from timm.models.layers import DropPath

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

# decline this line for resnet, due to imcompatible for transformers 4.40
# from transformers.utils.backbone_utils import load_backbone

from transformers import UperNetConfig
import torch.nn.functional as F
from transformers import AutoBackbone, AutoConfig
import numpy as np
import torchvision.transforms as tfs

# General docstring
_CONFIG_FOR_DOC = "UperNetConfig"

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            UperNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class UperNetPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.add_module(str(i), block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UperNetHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """

    def __init__(self, config, in_channels):
        super().__init__()

        self.config = config
        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = in_channels
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = UperNetConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = UperNetConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class UperNetFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config:
            Configuration.
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self, config, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()

        self.config = config
        self.in_channels = config.auxiliary_in_channels
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            UperNetConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim #768
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        x=x.permute(0,2,3,1)
        # print(x.shape, self.hidden_size)
        B, H, W, C = x.shape
        x = x.view(B, H, W, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1,2), norm='ortho') # FFT on N dimension

        x_real_1 = F.relu(self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) + self.complex_bias_1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) + self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1, self.complex_weight_2[1]) + self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1, self.complex_weight_2[0]) + self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1,2), norm="ortho")
        
        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        # x: complex 64?
        x = x.to(torch.float32)
        x = x.reshape(B, H, W, C).permute(0,3,1,2)
        return x

class UperNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UperNetConfig
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, UperNetPreTrainedModel):
            # module.backbone.init_weights()
            module.decode_head.init_weights()
            if module.auxiliary_head is not None:
                module.auxiliary_head.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        # self.backbone.init_weights()
        self.decode_head.init_weights()
        if self.auxiliary_head is not None:
            self.auxiliary_head.init_weights()


UPERNET_START_DOCSTRING = r"""
    Parameters:
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

UPERNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SegformerImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers in case the backbone has them. See
            `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
            returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    UPERNET_START_DOCSTRING,
)
class UperNetForSemanticSegmentation(UperNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.Backbone_type=="Swin":
            self.backbone = AutoBackbone.from_config(config)
            self.backbone_type="swin"
            config.auxiliary_in_channels=self.backbone.channels[2]*2
            config.auxiliary_channels=self.backbone.channels[2]
        elif "Swin_Diff" in config.Backbone_type:
            self.backbone = AutoBackbone.from_config(config.backbone_config)
            self.backbone_type="swindiff"
            config.auxiliary_in_channels=self.backbone.channels[2]
            config.auxiliary_channels=256
        elif "ResNet_Diff" in config.Backbone_type:
            if config.Backbone_type=="ResNet_Diff_18":
                self.backbone = AutoBackbone.from_pretrained("microsoft/resnet-18", out_features=["stage1", "stage2", "stage3", "stage4"])
                self.backbone_type="swindiff"
            elif config.Backbone_type=="ResNet_Diff_50":
                self.backbone = AutoBackbone.from_pretrained("microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"])
                self.backbone_type="swindiff"
            elif config.Backbone_type=="ResNet_Diff_101":
                self.backbone = AutoBackbone.from_pretrained("microsoft/resnet-101", out_features=["stage1", "stage2", "stage3", "stage4"])
                self.backbone_type="swindiff"
            config.auxiliary_in_channels=self.backbone.channels[2]
            config.auxiliary_channels=256
        if hasattr(config, 'channel_mixing'):
            self.channel_mixing=config.channel_mixing 
        else:
            self.channel_mixing=False
        # Semantic segmentation head(s)
        # print("self.backbone.dims",self.backbone.dims)
        if config.Backbone_type=="Swin":
            self.decode_head = UperNetHead(config, in_channels=[i*2 for i in self.backbone.channels])
        elif self.backbone_type=="concat":
            in_channels=[i*2 for i in self.backbone.dims]
            self.decode_head = UperNetHead(config, in_channels=[i*2 for i in self.backbone.dims])

            self.channel_mixing=nn.ModuleList([EinFFT(dim) for dim in in_channels]) if self.channel_mixing else None
        elif self.backbone_type=="swindiff":
            self.decode_head = UperNetHead(config, in_channels=self.backbone.channels) 
            self.channel_mixing=nn.ModuleList([EinFFT(dim) for dim in self.backbone.channels]) if self.channel_mixing else None
            self.norm=nn.ModuleList([nn.LayerNorm(dim) for dim in self.backbone.channels]) if self.channel_mixing else None
            self.drop_path=DropPath(0.1)
        elif self.backbone_type=="diff":
            self.decode_head = UperNetHead(config, in_channels=self.backbone.dims) 
            self.channel_mixing=nn.ModuleList([EinFFT(dim) for dim in self.backbone.dims]) if self.channel_mixing else None
            self.norm=nn.ModuleList([nn.LayerNorm(dim) for dim in self.backbone.dims]) if self.channel_mixing else None
            self.drop_path=DropPath(0.1)
        else:
            self.decode_head = UperNetHead(config, in_channels=self.backbone.dims)
            self.channel_mixing=nn.ModuleList([EinFFT(dim) for dim in self.backbone.dims]) if self.channel_mixing else None
        self.auxiliary_head = UperNetFCNHead(config) if config.use_auxiliary_head else None
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(UPERNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        x1: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
        >>> model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits  # shape (batch_size, num_labels, height, width)
        >>> list(logits.shape)
        [1, 150, 512, 512]
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if self.backbone_type=="swin":
            outputs=[]
            outputs_A = self.backbone.forward_with_filtered_kwargs(
            x2, output_hidden_states=False, output_attentions=False).feature_maps
            outputs_B = self.backbone.forward_with_filtered_kwargs(
            x1, output_hidden_states=False, output_attentions=False).feature_maps
            for i in range(4):
                outputs.append(torch.cat([outputs_A[i],outputs_B[i]],dim=1))
            features = tuple(outputs)
        elif self.backbone_type=="concat":
            outputs=[]
            outputs_A = self.backbone(x2)
            outputs_B = self.backbone(x1)
            for i in range(4):
                output=torch.cat([outputs_A[i],outputs_B[i]],dim=1)
                if self.channel_mixing:
                    output=self.channel_mixing[i](output)

                # outputs.append(output)
                outputs.append(torch.cat([outputs_A[i],outputs_B[i]],dim=1))
            features = tuple(outputs)
        elif self.backbone_type=="diff":
            outputs=[]
            outputs_A = self.backbone(x2)
            outputs_B = self.backbone(x1)
            for i in range(4):
                # output=torch.cat([outputs_A[i],outputs_B[i]],dim=1)
                output=outputs_A[i]-outputs_B[i]
                if self.channel_mixing:
                    output=output+self.drop_path(self.channel_mixing[i](self.norm[i](output.permute(0,2,3,1)).permute(0,3,1,2)))

                outputs.append(output)
                # outputs.append(torch.cat([outputs_A[i],outputs_B[i]],dim=1))
                features = tuple(outputs)
        elif self.backbone_type=="swindiff":
            outputs=[]
            outputs_A = self.backbone(x2).feature_maps
            outputs_B = self.backbone(x1).feature_maps
            for i in range(4):
                # output=torch.cat([outputs_A[i],outputs_B[i]],dim=1)
                output=outputs_A[i]-outputs_B[i]
                if self.channel_mixing:
                    output=self.channel_mixing[i](output)
                    output=output+self.drop_path(self.channel_mixing[i](self.norm[i](output.permute(0,2,3,1)).permute(0,3,1,2)))

                outputs.append(output)
                # outputs.append(torch.cat([outputs_A[i],outputs_B[i]],dim=1))
                features = tuple(outputs)
        else:
            outputs = self.backbone(x2, x1)
            features = outputs

        logits = self.decode_head(features)
        logits = nn.functional.interpolate(logits, size=x1.shape[2:], mode="bilinear", align_corners=False)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)
            auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # compute weighted loss
                loss_fct = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index)
                loss = loss_fct(logits, labels)
                if auxiliary_logits is not None:
                    auxiliary_loss = loss_fct(auxiliary_logits, labels)
                    loss += self.config.auxiliary_loss_weight * auxiliary_loss

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
def get_model_minenetcd(backbone_type="Swin_Diff_T", channel_mixing=True, num_classes=2):

    # backbone_type="ResNet_Diff_50"
    backbone_type="Swin_Diff_T"
    # backbone_type="VSSM_T_ST_Diff"

    # VMamba here
    # if "VSSM" in backbone_type:
    #     pretrained_model_name ="openmmlab/upernet-swin-tiny"
    #     config = AutoConfig.from_pretrained(pretrained_model_name)
    #     config.update({"num_labels":2,"Backbone_type":backbone_type, "channel_mixing":channel_mixing})
    #     model=UperNetForSemanticSegmentation._from_config(config)
    # Swin Transformer here
    if "Swin" in backbone_type:
        if backbone_type=="Swin_Diff_T":
            pretrained_model_name ="openmmlab/upernet-swin-tiny"
        elif backbone_type=="Swin_Diff_S":
            pretrained_model_name ="openmmlab/upernet-swin-small"
        elif backbone_type=="Swin_Diff_B":
            pretrained_model_name ="openmmlab/upernet-swin-base"
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.update({"num_labels":num_classes,"Backbone_type":backbone_type, "channel_mixing":channel_mixing})
        model=UperNetForSemanticSegmentation.from_pretrained(pretrained_model_name, config=config,ignore_mismatched_sizes=True)
    elif "ResNet" in backbone_type:
        pretrained_model_name ="openmmlab/upernet-swin-base"
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.update({"num_labels":num_classes,"Backbone_type":backbone_type, "channel_mixing":channel_mixing})
        model=UperNetForSemanticSegmentation._from_config(config)
    else:
        raise ValueError("We support Swin or ResNet, please make sure the config is correct.")
    return model
if __name__=="__main__":
    # config = AutoConfig.from_pretrained("ericyu/minenetcd-upernet-Swin-Diff-T-Pretrained-ChannelMixing-Dropout")
    # print(config)
    # model=UperNetForSemanticSegmentation._from_config(config).cuda()
    model=get_model_minenetcd(backbone_type="Res_Diff_50",channel_mixing=True,num_classes=2).cuda()
    # model=UperNetForSemanticSegmentation.from_pretrained("ericyu/minenetcd-upernet-Swin-Diff-T-Pretrained-ChannelMixing-Dropout", ignore_mismatched_sizes=True).cuda()
    output=model(torch.randn(1,3,256,256).cuda(),torch.randn(1,3,256,256).cuda(),labels=torch.zeros(1,256,256).cuda().long())
    print(output.logits)