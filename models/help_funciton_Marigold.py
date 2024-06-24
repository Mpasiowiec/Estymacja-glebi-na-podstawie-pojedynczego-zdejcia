# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
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
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import torch
import numpy as np
from typing import Optional, Tuple
from functools import partial
import math
import matplotlib
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize



def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST_EXACT,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method

# Search table for suggested max. inference batch size
bs_search_table = [
    # tested on A100-PCIE-80GB
    {"res": 768, "total_vram": 79, "bs": 35, "dtype": torch.float32},
    {"res": 1024, "total_vram": 79, "bs": 20, "dtype": torch.float32},
    # tested on A100-PCIE-40GB
    {"res": 768, "total_vram": 39, "bs": 15, "dtype": torch.float32},
    {"res": 1024, "total_vram": 39, "bs": 8, "dtype": torch.float32},
    {"res": 768, "total_vram": 39, "bs": 30, "dtype": torch.float16},
    {"res": 1024, "total_vram": 39, "bs": 15, "dtype": torch.float16},
    # tested on RTX3090, RTX4090
    {"res": 512, "total_vram": 23, "bs": 20, "dtype": torch.float32},
    {"res": 768, "total_vram": 23, "bs": 7, "dtype": torch.float32},
    {"res": 1024, "total_vram": 23, "bs": 3, "dtype": torch.float32},
    {"res": 512, "total_vram": 23, "bs": 40, "dtype": torch.float16},
    {"res": 768, "total_vram": 23, "bs": 18, "dtype": torch.float16},
    {"res": 1024, "total_vram": 23, "bs": 10, "dtype": torch.float16},
    # tested on GTX1080Ti
    {"res": 512, "total_vram": 10, "bs": 5, "dtype": torch.float32},
    {"res": 768, "total_vram": 10, "bs": 2, "dtype": torch.float32},
    {"res": 512, "total_vram": 10, "bs": 10, "dtype": torch.float16},
    {"res": 768, "total_vram": 10, "bs": 5, "dtype": torch.float16},
    {"res": 1024, "total_vram": 10, "bs": 3, "dtype": torch.float16},
]

def ensemble_depth(
    depth: torch.Tensor,
    scale_invariant: bool = True,
    shift_invariant: bool = True,
    output_uncertainty: bool = False,
    reduction: str = "median",
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    max_res: int = 1024,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Ensembles depth maps represented by the `depth` tensor with expected shape `(B, 1, H, W)`, where B is the
    number of ensemble members for a given prediction of size `(H x W)`. Even though the function is designed for
    depth maps, it can also be used with disparity maps as long as the input tensor values are non-negative. The
    alignment happens when the predictions have one or more degrees of freedom, that is when they are either
    affine-invariant (`scale_invariant=True` and `shift_invariant=True`), or just scale-invariant (only
    `scale_invariant=True`). For absolute predictions (`scale_invariant=False` and `shift_invariant=False`)
    alignment is skipped and only ensembling is performed.

    Args:
        depth (`torch.Tensor`):
            Input ensemble depth maps.
        scale_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as scale-invariant.
        shift_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as shift-invariant.
        output_uncertainty (`bool`, *optional*, defaults to `False`):
            Whether to output uncertainty map.
        reduction (`str`, *optional*, defaults to `"median"`):
            Reduction method used to ensemble aligned predictions. The accepted values are: `"mean"` and
            `"median"`.
        regularizer_strength (`float`, *optional*, defaults to `0.02`):
            Strength of the regularizer that pulls the aligned predictions to the unit range from 0 to 1.
        max_iter (`int`, *optional*, defaults to `2`):
            Maximum number of the alignment solver steps. Refer to `scipy.optimize.minimize` function, `options`
            argument.
        tol (`float`, *optional*, defaults to `1e-3`):
            Alignment solver tolerance. The solver stops when the tolerance is reached.
        max_res (`int`, *optional*, defaults to `1024`):
            Resolution at which the alignment is performed; `None` matches the `processing_resolution`.
    Returns:
        A tensor of aligned and ensembled depth maps and optionally a tensor of uncertainties of the same shape:
        `(1, 1, H, W)`.
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expecting 4D tensor of shape [B,1,H,W]; got {depth.shape}.")
    if reduction not in ("mean", "median"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")
    if not scale_invariant and shift_invariant:
        raise ValueError("Pure shift-invariant ensembling is not supported.")

    def init_param(depth: torch.Tensor):
        init_min = depth.reshape(ensemble_size, -1).min(dim=1).values
        init_max = depth.reshape(ensemble_size, -1).max(dim=1).values

        if scale_invariant and shift_invariant:
            init_s = 1.0 / (init_max - init_min).clamp(min=1e-6)
            init_t = -init_s * init_min
            param = torch.cat((init_s, init_t)).cpu().numpy()
        elif scale_invariant:
            init_s = 1.0 / init_max.clamp(min=1e-6)
            param = init_s.cpu().numpy()
        else:
            raise ValueError("Unrecognized alignment.")

        return param

    def align(depth: torch.Tensor, param: np.ndarray) -> torch.Tensor:
        if scale_invariant and shift_invariant:
            s, t = np.split(param, 2)
            s = torch.from_numpy(s).to(depth).view(ensemble_size, 1, 1, 1)
            t = torch.from_numpy(t).to(depth).view(ensemble_size, 1, 1, 1)
            out = depth * s + t
        elif scale_invariant:
            s = torch.from_numpy(param).to(depth).view(ensemble_size, 1, 1, 1)
            out = depth * s
        else:
            raise ValueError("Unrecognized alignment.")
        return out

    def ensemble(
        depth_aligned: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uncertainty = None
        if reduction == "mean":
            prediction = torch.mean(depth_aligned, dim=0, keepdim=True)
            if return_uncertainty:
                uncertainty = torch.std(depth_aligned, dim=0, keepdim=True)
        elif reduction == "median":
            prediction = torch.median(depth_aligned, dim=0, keepdim=True).values
            if return_uncertainty:
                uncertainty = torch.median(
                    torch.abs(depth_aligned - prediction), dim=0, keepdim=True
                ).values
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}.")
        return prediction, uncertainty

    def cost_fn(param: np.ndarray, depth: torch.Tensor) -> float:
        cost = 0.0
        depth_aligned = align(depth, param)

        for i, j in torch.combinations(torch.arange(ensemble_size)):
            diff = depth_aligned[i] - depth_aligned[j]
            cost += (diff**2).mean().sqrt().item()

        if regularizer_strength > 0:
            prediction, _ = ensemble(depth_aligned, return_uncertainty=False)
            err_near = (0.0 - prediction.min()).abs().item()
            err_far = (1.0 - prediction.max()).abs().item()
            cost += (err_near + err_far) * regularizer_strength

        return cost

    def compute_param(depth: torch.Tensor):
        import scipy

        depth_to_align = depth.to(torch.float32)
        if max_res is not None and max(depth_to_align.shape[2:]) > max_res:
            depth_to_align = resize_max_res(
                depth_to_align, max_res, get_tv_resample_method("nearest-exact")
            )

        param = init_param(depth_to_align)

        res = scipy.optimize.minimize(
            partial(cost_fn, depth=depth_to_align),
            param,
            method="BFGS",
            tol=tol,
            options={"maxiter": max_iter, "disp": False},
        )

        return res.x

    requires_aligning = scale_invariant or shift_invariant
    ensemble_size = depth.shape[0]

    if requires_aligning:
        param = compute_param(depth)
        depth = align(depth, param)

    depth, uncertainty = ensemble(depth, return_uncertainty=output_uncertainty)

    depth_max = depth.max()
    if scale_invariant and shift_invariant:
        depth_min = depth.min()
    elif scale_invariant:
        depth_min = 0
    else:
        raise ValueError("Unrecognized alignment.")
    depth_range = (depth_max - depth_min).clamp(min=1e-6)
    depth = (depth - depth_min) / depth_range
    if output_uncertainty:
        uncertainty /= depth_range

    return depth, uncertainty  # [1,1,H,W], [1,1,H,W]

def find_batch_size(ensemble_size: int, input_res: int, dtype: torch.dtype) -> int:
    """
    Automatically search for suitable operating batch size.

    Args:
        ensemble_size (`int`):
            Number of predictions to be ensembled.
        input_res (`int`):
            Operating resolution of the input image.

    Returns:
        `int`: Operating batch size.
    """
    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.mem_get_info()[1] / 1024.0**3
    filtered_bs_search_table = [s for s in bs_search_table if s["dtype"] == dtype]
    for settings in sorted(
        filtered_bs_search_table,
        key=lambda k: (k["res"], -k["total_vram"]),
    ):
        if input_res <= settings["res"] and total_vram >= settings["total_vram"]:
            bs = settings["bs"]
            if bs > ensemble_size:
                bs = ensemble_size
            elif bs > math.ceil(ensemble_size / 2) and bs < ensemble_size:
                bs = math.ceil(ensemble_size / 2)
            return bs

    return 1