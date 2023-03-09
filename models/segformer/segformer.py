import torch
from torch import nn
from torch.nn.functional import interpolate

from models.segformer.backbones import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, MixTransformer
from models.segformer.heads import SegFormerHead


class SegFormer(nn.Module):
    def __init__(self, backbone: MixTransformer, decode_head: SegFormerHead):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    @property
    def align_corners(self):
        return self.decode_head.align_corners

    @property
    def num_classes(self):
        return self.decode_head.num_classes

    def forward(self, img, flow):
        x = torch.cat([img, flow], dim=1)
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decode_head(x)
        x = interpolate(x, size=image_hw, mode='bilinear', align_corners=self.align_corners)
        return x


def create_segformer_b0(num_classes, uncertainty_probability):
    backbone = mit_b0(uncertainty_probability=uncertainty_probability, in_c=6)
    head = SegFormerHead(
        in_channels=(32, 64, 160, 256),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)


def create_segformer_b1(num_classes, uncertainty_probability):
    backbone = mit_b1(uncertainty_probability=uncertainty_probability, in_c=6)
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)


def create_segformer_b2(num_classes, uncertainty_probability):
    backbone = mit_b2(uncertainty_probability=uncertainty_probability, in_c=6)
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b3(num_classes, uncertainty_probability):
    backbone = mit_b3(uncertainty_probability=uncertainty_probability, in_c=6)
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b4(num_classes, uncertainty_probability):
    backbone = mit_b4(uncertainty_probability=uncertainty_probability, in_c=6)
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b5(num_classes, uncertainty_probability):
    backbone = mit_b5(uncertainty_probability=uncertainty_probability, in_c=6)
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def _load_pretrained_weights_(model, pretrained, model_name):
    state_dict = torch.load(pretrained)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decode_head'):
            if k.endswith('.proj.weight'):
                k = k.replace('.proj.weight', '.weight')
                v = v[..., None, None]
            elif k.endswith('.proj.bias'):
                k = k.replace('.proj.bias', '.bias')
            elif '.linear_fuse.conv.' in k:
                k = k.replace('.linear_fuse.conv.', '.linear_fuse.')
            elif '.linear_fuse.bn.' in k:
                k = k.replace('.linear_fuse.bn.', '.bn.')

            if '.linear_c4.' in k:
                k = k.replace('.linear_c4.', '.layers.0.')
            elif '.linear_c3.' in k:
                k = k.replace('.linear_c3.', '.layers.1.')
            elif '.linear_c2.' in k:
                k = k.replace('.linear_c2.', '.layers.2.')
            elif '.linear_c1.' in k:
                k = k.replace('.linear_c1.', '.layers.3.')
        else:
            if 'patch_embed1.' in k:
                k = k.replace('patch_embed1.', 'stages.0.patch_embed.')
            elif 'patch_embed2.' in k:
                k = k.replace('patch_embed2.', 'stages.1.patch_embed.')
            elif 'patch_embed3.' in k:
                k = k.replace('patch_embed3.', 'stages.2.patch_embed.')
            elif 'patch_embed4.' in k:
                k = k.replace('patch_embed4.', 'stages.3.patch_embed.')
            elif 'block1.' in k:
                k = k.replace('block1.', 'stages.0.blocks.')
            elif 'block2.' in k:
                k = k.replace('block2.', 'stages.1.blocks.')
            elif 'block3.' in k:
                k = k.replace('block3.', 'stages.2.blocks.')
            elif 'block4.' in k:
                k = k.replace('block4.', 'stages.3.blocks.')
            elif 'norm1.' in k:
                k = k.replace('norm1.', 'stages.0.norm.')
            elif 'norm2.' in k:
                k = k.replace('norm2.', 'stages.1.norm.')
            elif 'norm3.' in k:
                k = k.replace('norm3.', 'stages.2.norm.')
            elif 'norm4.' in k:
                k = k.replace('norm4.', 'stages.3.norm.')

            if '.mlp.dwconv.dwconv.' in k:
                k = k.replace('.mlp.dwconv.dwconv.', '.mlp.conv.')

            if '.mlp.' in k:
                k = k.replace('.mlp.', '.ffn.')
        new_state_dict[k] = v

    if model_name in ["segformer_b0", "segformer_b1", "segformer_b2", "segformer_b3", "segformer_b4", "segformer_b5"]:
        pop_keys = ["stages.0.patch_embed.proj.weight"]
    else:
        pop_keys = ["backbone.stages.0.patch_embed.proj.weight", "decode_head.linear_pred.weight",
                    "decode_head.linear_pred.bias"]

    for pop_key in pop_keys:
        del new_state_dict[pop_key]

    model.load_state_dict(new_state_dict, strict=False)


def segformer_b0_ade(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B0 model for the ADE20K segmentation task.
    """
    model = create_segformer_b0(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b0_ade")
    return model


def segformer_b1_ade(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B1 model for the ADE20K segmentation task.
    """
    model = create_segformer_b1(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b1_ade")
    return model


def segformer_b2_ade(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B2 model for the ADE20K segmentation task.
    """
    model = create_segformer_b2(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b2_ade")
    return model


def segformer_b3_ade(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B3 model for the ADE20K segmentation task.
    """
    model = create_segformer_b3(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b3_ade")
    return model


def segformer_b4_ade(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B4 model for the ADE20K segmentation task.
    """
    model = create_segformer_b4(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b4_ade")
    return model


def segformer_b5_ade(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B5 model for the ADE20K segmentation task.
    """
    model = create_segformer_b5(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b5_ade")
    return model


def segformer_b0_city(pretrained="", num_classes=19, uncertainty_probability=0.5):
    """Create a SegFormer-B0 model for the CityScapes segmentation task.
    """
    model = create_segformer_b0(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b0_city")
    return model


def segformer_b1_city(pretrained="", num_classes=19, uncertainty_probability=0.5):
    """Create a SegFormer-B1 model for the CityScapes segmentation task.
    """
    model = create_segformer_b1(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b1_city")
    return model


def segformer_b2_city(pretrained="", num_classes=19, uncertainty_probability=0.5):
    """Create a SegFormer-B2 model for the CityScapes segmentation task.
    """
    model = create_segformer_b2(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b2_city")
    return model


def segformer_b3_city(pretrained="", num_classes=19, uncertainty_probability=0.5):
    """Create a SegFormer-B3 model for the CityScapes segmentation task.
    """
    model = create_segformer_b3(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b3_city")
    return model


def segformer_b4_city(pretrained="", num_classes=19, uncertainty_probability=0.5):
    """Create a SegFormer-B4 model for the CityScapes segmentation task.
    """
    model = create_segformer_b4(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b4_city")
    return model


def segformer_b5_city(pretrained="", num_classes=19, uncertainty_probability=0.5):
    """Create a SegFormer-B5 model for the CityScapes segmentation task.
    """
    model = create_segformer_b5(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model, pretrained, model_name="segformer_b5_city")
    return model


def segformer_b0(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B0 model.
    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b0(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model.backbone, pretrained, model_name="segformer_b0")
    return model


def segformer_b1(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B1 model.
    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b1(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model.backbone, pretrained, model_name="segformer_b1")
    return model


def segformer_b2(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B2 model.
    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b2(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model.backbone, pretrained, model_name="segformer_b2")
    return model


def segformer_b3(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B3 model.
    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b3(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model.backbone, pretrained, model_name="segformer_b3")
    return model


def segformer_b4(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B4 model.
    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b4(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model.backbone, pretrained, model_name="segformer_b4")
    return model


def segformer_b5(pretrained="", num_classes=150, uncertainty_probability=0.5):
    """Create a SegFormer-B5 model.
    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b5(num_classes=num_classes, uncertainty_probability=uncertainty_probability)
    if pretrained:
        _load_pretrained_weights_(model.backbone, pretrained, model_name="segformer_b5")
    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512)
    pretrained = ""
    model = segformer_b0_ade(pretrained, num_classes=1)
    print(model(x, x).shape)