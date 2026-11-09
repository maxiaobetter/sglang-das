# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for GLM-5-Next."""

import math
from typing import Literal, Optional, Union

import numpy as np

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, logging
from transformers.video_utils import VideoInput
from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from transformers.image_processing_utils import (
    BatchFeature,
)

try:
    # transformers < 5.3
    from transformers.image_processing_utils_fast import DefaultFastImageProcessorKwargs
except ImportError:
    # transformers >= 5.3 renamed the base fast-processor kwargs to ImagesKwargs
    from transformers.processing_utils import ImagesKwargs as DefaultFastImageProcessorKwargs

from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TensorType,
    auto_docstring,
    logging,
)

logger = logging.get_logger(__name__)

ResizeMode = Literal["resize", "pad"]


def _ceil_to_factor(value: int, factor: int) -> int:
    """Round a positive integer upward to the nearest multiple of factor."""
    return math.ceil(value / factor) * factor


def _fit_aligned_size_within_budget(
    t: int,
    h: int,
    w: int,
    h_factor: int,
    w_factor: int,
    max_pixels: int,
) -> tuple[int, int]:
    """
    Find the largest proportional size whose upward-aligned canvas fits the
    pixel budget.

    The binary search is performed on the unaligned content height. Each
    candidate content size is then rounded upward to h_factor/w_factor.
    Consequently, the returned aligned dimensions always satisfy:

        t * aligned_h * aligned_w <= max_pixels
    """
    minimum_pixels = t * h_factor * w_factor
    if max_pixels < minimum_pixels:
        raise ValueError(
            f"max_pixels={max_pixels} is too small. At least "
            f"{minimum_pixels} pixels are required for one aligned patch."
        )

    low, high = 1, h
    best_h, best_w = h_factor, w_factor

    while low <= high:
        content_h = (low + high) // 2
        content_w = max(1, math.floor(w * content_h / h))
        aligned_h = _ceil_to_factor(content_h, h_factor)
        aligned_w = _ceil_to_factor(content_w, w_factor)

        if t * aligned_h * aligned_w <= max_pixels:
            best_h, best_w = aligned_h, aligned_w
            low = content_h + 1
        else:
            high = content_h - 1

    return best_h, best_w


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """
    Calculate the aligned output canvas size.

    Height and width are always rounded upward to their respective factors.
    If the upward-aligned canvas exceeds max_pixels, the largest proportional
    size whose upward-aligned canvas fits max_pixels is selected.
    """
    if min(t, h, w, t_factor, h_factor, w_factor) <= 0:
        raise ValueError("Image dimensions and alignment factors must be positive.")
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("min_pixels and max_pixels must be positive.")
    if min_pixels > max_pixels:
        raise ValueError("min_pixels must be less than or equal to max_pixels.")

    t_bar = max(t_factor, round(t / t_factor) * t_factor)
    h_bar = _ceil_to_factor(h, h_factor)
    w_bar = _ceil_to_factor(w, w_factor)

    if t_bar * h_bar * w_bar > max_pixels:
        h_bar, w_bar = _fit_aligned_size_within_budget(
            t=t_bar,
            h=h,
            w=w,
            h_factor=h_factor,
            w_factor=w_factor,
            max_pixels=max_pixels,
        )
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = _ceil_to_factor(max(1, math.ceil(h * beta)), h_factor)
        w_bar = _ceil_to_factor(max(1, math.ceil(w * beta)), w_factor)

        # Alignment can push a candidate slightly over a tight max_pixels
        # budget. Refit it when that happens.
        if t_bar * h_bar * w_bar > max_pixels:
            h_bar, w_bar = _fit_aligned_size_within_budget(
                t=t_bar,
                h=h,
                w=w,
                h_factor=h_factor,
                w_factor=w_factor,
                max_pixels=max_pixels,
            )

    return h_bar, w_bar


def _get_pad_content_size(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """
    Return an aspect-ratio-preserving content size that fits the canvas.

    Oversized images are shrunk proportionally. Small images are enlarged only
    when allow_upscale=True. Padding is applied after the proportional resize.
    """
    scale = min(
        canvas_height / image_height,
        canvas_width / image_width,
    )
    if not allow_upscale:
        scale = min(1.0, scale)
    content_height = max(1, min(canvas_height, math.floor(image_height * scale)))
    content_width = max(1, min(canvas_width, math.floor(image_width * scale)))
    return content_height, content_width



class Glm5nextImageProcessor(BaseImageProcessor):
    r"""
    Constructs a GLM-5-Next image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        min_image_tokens (`int`, *required*):
            Minimum number of tokens per image. The min pixel budget is derived as
            `min_image_tokens * patch_size ** 2 * merge_size ** 2`.
        max_image_tokens (`int`, *required*):
            Maximum number of tokens per image. The max pixel budget is derived as
            `max_image_tokens * patch_size ** 2 * merge_size ** 2`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        patch_expand_factor: Optional[int] = None,
        merge_size: Optional[int] = None,
        min_image_tokens: Optional[int] = None,
        max_image_tokens: Optional[int] = None,
        resize_mode: ResizeMode = "pad",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        # min/max pixels are derived from the token bounds where needed:
        # tokens * patch_size**2 * merge_size**2.
        assert (
            min_image_tokens is not None and max_image_tokens is not None
        ), "min_image_tokens and max_image_tokens must be provided."
        if resize_mode not in ("resize", "pad"):
            raise ValueError("resize_mode must be either 'resize' or 'pad'.")
        self.resize_mode = resize_mode

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

        self.temporal_patch_size = temporal_patch_size
        self.patch_expand_factor = patch_expand_factor
        self.do_convert_rgb = do_convert_rgb

    def _resize_or_pad(
        self,
        image: np.ndarray,
        target_height: int,
        target_width: int,
        resize_mode: ResizeMode,
        resample: PILImageResampling,
        input_data_format: Union[str, ChannelDimension],
        allow_upscale: bool = False,
    ) -> np.ndarray:
        if resize_mode == "resize":
            return resize(
                image,
                size=(target_height, target_width),
                resample=resample,
                input_data_format=input_data_format,
            )

        if resize_mode != "pad":
            raise ValueError("resize_mode must be either 'resize' or 'pad'.")

        image_height, image_width = get_image_size(image, channel_dim=input_data_format)
        content_height, content_width = _get_pad_content_size(
            image_height=image_height,
            image_width=image_width,
            canvas_height=target_height,
            canvas_width=target_width,
            allow_upscale=allow_upscale,
        )

        if (content_height, content_width) != (image_height, image_width):
            image = resize(
                image,
                size=(content_height, content_width),
                resample=resample,
                input_data_format=input_data_format,
            )

        pad_bottom = target_height - content_height
        pad_right = target_width - content_width
        pad_width = [(0, 0)] * image.ndim

        if input_data_format == ChannelDimension.FIRST:
            pad_width[-2] = (0, pad_bottom)
            pad_width[-1] = (0, pad_right)
        else:
            pad_width[0] = (0, pad_bottom)
            pad_width[1] = (0, pad_right)

        # pad only the right and bottom sides with zeros.
        return np.pad(image, pad_width=pad_width, mode="constant", constant_values=0)

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        resize_mode: ResizeMode = "pad",
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        min_image_tokens: Optional[int] = None,
        max_image_tokens: Optional[int] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`List[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_flat_list_of_images(images)

        min_image_tokens = (
            min_image_tokens if min_image_tokens is not None else self.min_image_tokens
        )
        max_image_tokens = (
            max_image_tokens if max_image_tokens is not None else self.max_image_tokens
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []

        for image in images:
            if do_resize:
                factor = patch_size ** 2 * merge_size ** 2 * temporal_patch_size
                min_pixels = self.min_image_tokens * factor
                max_pixels = self.max_image_tokens * factor
                resized_height, resized_width = smart_resize(
                    t=temporal_patch_size,
                    h=height,
                    w=width,
                    t_factor=temporal_patch_size,
                    h_factor=patch_size * merge_size * self.patch_expand_factor,
                    w_factor=patch_size * merge_size * self.patch_expand_factor,
                    min_pixels=min_image_tokens * factor,
                    max_pixels=max_image_tokens * factor,
                )
                image = self._resize_or_pad(
                    image=image,
                    target_height=resized_height,
                    target_width=resized_width,
                    resize_mode=resize_mode,
                    resample=resample,
                    input_data_format=input_data_format,
                    allow_upscale=(temporal_patch_size * height * width < min_pixels),
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[VideoInput] = None,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        patch_expand_factor: Optional[int] = None,
        merge_size: Optional[int] = None,
        resize_mode: Optional[ResizeMode] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        min_image_tokens: Optional[int] = None,
        max_image_tokens: Optional[int] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
                The max pixels of the image to resize the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_expand_factor = patch_expand_factor if patch_expand_factor is not None else self.patch_expand_factor
        resize_mode = resize_mode if resize_mode is not None else self.resize_mode
        if resize_mode not in ("resize", "pad"):
            raise ValueError("resize_mode must be either 'resize' or 'pad'.")
        min_image_tokens = min_image_tokens if min_image_tokens is not None else self.min_image_tokens
        max_image_tokens = max_image_tokens if max_image_tokens is not None else self.max_image_tokens

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            # resize bounds are derived from min_image_tokens/max_image_tokens; pass a placeholder
            # so the base check (which requires `size` when do_resize) passes.
            size={"longest_edge": 1},
            resample=resample,
        )

        data = {}
        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    resize_mode=resize_mode,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        min_image_tokens = images_kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        factor = patch_size ** 2 * merge_size ** 2 * self.temporal_patch_size
        resized_height, resized_width = smart_resize(
            t=self.temporal_patch_size,
            h=height,
            w=width,
            t_factor=self.temporal_patch_size,
            h_factor=patch_size * merge_size * self.patch_expand_factor,
            w_factor=patch_size * merge_size * self.patch_expand_factor,
            min_pixels=min_image_tokens * factor,
            max_pixels=max_image_tokens * factor,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class Glm5nextFastImageProcessorKwargs(DefaultFastImageProcessorKwargs, total=False):
    """
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    patch_expand_factor (`int`, *optional*, defaults to 1):
        The patch expand factor of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    min_image_tokens (`int`, *optional*):
        Minimum number of tokens per image; the min pixel budget is derived from it.
    max_image_tokens (`int`, *optional*):
        Maximum number of tokens per image; the max pixel budget is derived from it.
    resize_mode (`str`, *optional*, defaults to `"pad"`):
        Image geometry operation. Must be either `"resize"` or `"pad"`. In `"pad"` mode, small images are
        enlarged to satisfy the min budget, oversized images are shrunk to satisfy the max budget, and
        padding is added only to the right and bottom sides.
    """

    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    patch_expand_factor: Optional[int]
    merge_size: Optional[int]
    min_image_tokens: Optional[int]
    max_image_tokens: Optional[int]
    resize_mode: Optional[ResizeMode]


@auto_docstring
class Glm5nextImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    # `size` is unused (kept only because the base class expects the attribute); resize is driven
    # by min_image_tokens/max_image_tokens.
    size = None
    do_rescale = True
    do_normalize = True
    image_mean = None
    image_std = None
    do_convert_rgb = True
    patch_size = None
    temporal_patch_size = None
    merge_size = None
    min_image_tokens = None
    max_image_tokens = None
    resize_mode = "pad"
    valid_kwargs = Glm5nextFastImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Glm5nextFastImageProcessorKwargs]):
        super().__init__(**kwargs)
        # min/max pixels are derived from the token bounds where needed:
        # tokens * patch_size**2 * merge_size**2.
        assert (
            self.min_image_tokens is not None and self.max_image_tokens is not None
        ), "min_image_tokens and max_image_tokens must be provided."
        if self.resize_mode not in ("resize", "pad"):
            raise ValueError("resize_mode must be either 'resize' or 'pad'.")

    def _validate_preprocess_kwargs(self, size: Optional[SizeDict] = None, **kwargs):
        # Resize bounds are derived from min_image_tokens/max_image_tokens, so `size` is unused.
        # Pass a non-null placeholder so the base check (which requires `size` when do_resize) passes.
        return super()._validate_preprocess_kwargs(size=size or SizeDict(longest_edge=1), **kwargs)

    def _resize_or_pad(
        self,
        stacked_images: "torch.Tensor",
        target_height: int,
        target_width: int,
        resize_mode: ResizeMode,
        interpolation: Optional["F.InterpolationMode"],
        allow_upscale: bool = False,
    ) -> "torch.Tensor":
        height, width = stacked_images.shape[-2:]

        if resize_mode == "resize":
            return self.resize(
                stacked_images,
                size=SizeDict(height=target_height, width=target_width),
                interpolation=interpolation,
            )

        if resize_mode != "pad":
            raise ValueError("resize_mode must be either 'resize' or 'pad'.")

        content_height, content_width = _get_pad_content_size(
            image_height=height,
            image_width=width,
            canvas_height=target_height,
            canvas_width=target_width,
            allow_upscale=allow_upscale,
        )

        if (content_height, content_width) != (height, width):
            stacked_images = self.resize(
                stacked_images,
                size=SizeDict(height=content_height, width=content_width),
                interpolation=interpolation,
            )

        # torchvision padding order: [left, top, right, bottom] -> pad only right and bottom.
        return F.pad(
            stacked_images,
            padding=[0, 0, target_width - content_width, target_height - content_height],
            fill=0,
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        patch_expand_factor: int,
        merge_size: int,
        resize_mode: ResizeMode,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        min_image_tokens: Optional[int] = None,
        max_image_tokens: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.
        """
        min_image_tokens = (
            min_image_tokens if min_image_tokens is not None else self.min_image_tokens
        )
        max_image_tokens = (
            max_image_tokens if max_image_tokens is not None else self.max_image_tokens
        )
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        # assert patch_expand_factor == 4, f"current version patch_expand_factor must be 4, the patch_expand_factor value is {patch_expand_factor}"
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                factor = patch_size ** 2 * merge_size ** 2 * temporal_patch_size
                min_pixels = self.min_image_tokens * factor
                max_pixels = self.max_image_tokens * factor
                resized_height, resized_width = smart_resize(
                    t=temporal_patch_size,
                    h=height,
                    w=width,
                    t_factor=temporal_patch_size,
                    h_factor=patch_size * merge_size * patch_expand_factor,
                    w_factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=min_image_tokens * factor,
                    max_pixels=max_image_tokens * factor,
                )
                stacked_images = self._resize_or_pad(
                    stacked_images=stacked_images,
                    target_height=resized_height,
                    target_width=resized_width,
                    resize_mode=resize_mode,
                    interpolation=interpolation,
                    allow_upscale=(temporal_patch_size * height * width < min_pixels),
                )
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]

            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:  # (B, C, H, W)
                patches = patches.unsqueeze(1)  # (B, T=1, C, H, W)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1, temporal_patch_size - (patches.shape[1] % temporal_patch_size), 1, 1, 1
                )
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, t_len, channel = patches.shape[:3]
            grid_t = t_len // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # (B, grid_t, gh, gw, mh, mw, C, tp, ph, pw)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Glm5nextFastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

__all__ = ["Glm5nextImageProcessor", "Glm5nextImageProcessorFast", "smart_resize"]
