from .transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from .transforms.color_transforms import GammaTransform
from .transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from .transforms.resample_transforms import SimulateLowResolutionTransform
from .transforms.spatial_transforms import SpatialTransform, MirrorTransform
from .params import default_3D_augmentation_params as params
import torchvision

medklip_trans = torchvision.transforms.Compose([
        MirrorTransform(params.get("mirror_axes")),
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        BrightnessTransform(params.get("additive_brightness_mu"),params.get("additive_brightness_sigma"),True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                         p_per_channel=params.get("additive_brightness_p_per_channel")),
        ContrastAugmentationTransform(p_per_sample=0.15),
        GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),p_per_sample=params["p_gamma"])
    ])

kad_trans = torchvision.transforms.Compose([
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        BrightnessTransform(params.get("additive_brightness_mu"),params.get("additive_brightness_sigma"),True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                         p_per_channel=params.get("additive_brightness_p_per_channel")),
        ContrastAugmentationTransform(p_per_sample=0.15),
        GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),p_per_sample=params["p_gamma"])
    ])

# tr_transforms = []

# 空间变化
# tr_transforms.append()
# flip
# rotate

# 高斯噪声
# tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

# 高斯模糊
# tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))

# 镜像
# tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

# 亮度变换
# tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
     
# 亮度变换
# tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),params.get("additive_brightness_sigma"),True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                         # p_per_channel=params.get("additive_brightness_p_per_channel")))
# 对比度变换
# tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))

# 分辨率变换
# tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
#                                                         p_per_channel=0.5,
#                                                         order_downsample=0, order_upsample=3, p_per_sample=0.25,
#                                                         ignore_axes=None))

# Gamma变换，对输入图像灰度值进行非线性操作，使输出图像灰度值与输入图像灰度值呈指数关系

# tr_transforms.append(GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),p_per_sample=params["p_gamma"]))