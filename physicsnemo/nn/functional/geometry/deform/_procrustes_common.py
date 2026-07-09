# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Shared tensor algebra for Procrustes backends."""

from collections.abc import Callable

import torch


def solve_procrustes(
    source: torch.Tensor,
    target: torch.Tensor,
    scale: bool,
    project_rotation: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve normalized rank-three Procrustes with a backend rotation projector."""

    # Scale each cloud independently before any reductions. The detached
    # positive normalizers leave the fitted transform unchanged after the
    # analytic scale restoration and prevent finite float32 coordinates from
    # overflowing or underflowing covariance and variance calculations.
    source_magnitude = source.detach().abs().amax(dim=(-2, -1), keepdim=True)
    target_magnitude = target.detach().abs().amax(dim=(-2, -1), keepdim=True)
    source_magnitude = torch.where(
        source_magnitude > 0,
        source_magnitude,
        torch.ones_like(source_magnitude),
    )
    target_magnitude = torch.where(
        target_magnitude > 0,
        target_magnitude,
        torch.ones_like(target_magnitude),
    )

    source_normalized = source / source_magnitude
    target_normalized = target / target_magnitude
    source_mean_normalized = source_normalized.mean(dim=-2, keepdim=True)
    target_mean_normalized = target_normalized.mean(dim=-2, keepdim=True)
    source_centered = source_normalized - source_mean_normalized
    target_centered = target_normalized - target_mean_normalized

    covariance = (source_centered.transpose(-2, -1) @ target_centered) / source.shape[
        -2
    ]
    rotation = project_rotation(covariance)

    if scale:
        source_variance = source_centered.square().sum(dim=-1).mean(dim=-1)
        numerator = torch.einsum("bij,bji->b", rotation, covariance).clamp_min(0)
        safe_source_variance = torch.where(
            source_variance > 0,
            source_variance,
            torch.ones_like(source_variance),
        )
        magnitude_ratio = (target_magnitude / source_magnitude).squeeze(-1).squeeze(-1)
        scale_factor = numerator / safe_source_variance * magnitude_ratio
    else:
        scale_factor = torch.ones_like(source_mean_normalized[:, 0, 0])

    source_mean = (source_magnitude * source_mean_normalized).squeeze(-2)
    target_mean = (target_magnitude * target_mean_normalized).squeeze(-2)
    rotated_source_mean = source_mean.unsqueeze(-2) @ rotation.transpose(-2, -1)
    translation = target_mean - scale_factor.unsqueeze(
        -1
    ) * rotated_source_mean.squeeze(-2)
    return rotation, translation, scale_factor


__all__ = ["solve_procrustes"]
