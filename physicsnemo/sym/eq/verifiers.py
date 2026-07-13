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

"""Physics priors ("verifiers") for gray-box / inverse-problem discovery.

Small, reusable loss terms that constrain a discovered quantity -- an unknown
coefficient, or an unknown term's network -- to stay physically admissible in
regions the data does not cover, where the physics residual alone leaves it
under-determined. See ``examples/graybox_discovery`` for a worked example
where an equilibrium prior identifies a reaction term outside the observed
range.
"""

from __future__ import annotations

import torch


def equilibrium_verifier(
    module: torch.nn.Module,
    states: torch.Tensor,
    values: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Pin ``module`` to known values at known states.

    Many gray-box / inverse problems have a handful of points where the
    discovered quantity is known analytically even though it is not
    directly observed -- e.g. known equilibria, boundary values, or
    symmetry points. This evaluates ``module`` at ``states`` and returns
    the mean-squared deviation from ``values``, for use as an additional
    physics-loss term.

    Parameters
    ----------
    module : torch.nn.Module
        The network standing in for the discovered quantity, e.g. the
        ``u -> R`` network in a gray-box closure-discovery problem.
    states : torch.Tensor
        Input states at which ``module``'s output is known, shape
        :math:`(N, D_{in})`.
    values : torch.Tensor or float
        The known value(s) of ``module(states)``. A scalar broadcasts to
        every state (the common case: known equilibria are usually zero).

    Returns
    -------
    torch.Tensor
        Scalar mean-squared-error loss.

    Examples
    --------
    >>> import torch
    >>> net = torch.nn.Linear(1, 1)
    >>> anchors = torch.tensor([[0.0], [1.0]])
    >>> loss = equilibrium_verifier(net, anchors, values=0.0)
    >>> loss.shape
    torch.Size([])
    """
    pred = module(states)
    if not torch.is_tensor(values):
        values = torch.full_like(pred, float(values))
    return torch.nn.functional.mse_loss(pred, values)


class SignConstraint(torch.nn.Module):
    """Penalize ``module`` for the wrong sign over a range of states.

    Complements :func:`equilibrium_verifier` for problems where the
    discovered quantity is known to keep one sign on part of its domain
    (e.g. a diffusivity or reaction rate that must stay non-negative)
    rather than an exact value.

    Parameters
    ----------
    sign : {"+", "-"}
        Enforce ``module(states) >= 0`` (``"+"``) or ``<= 0`` (``"-"``).

    Forward
    -------
    module : torch.nn.Module
        The network standing in for the discovered quantity.
    states : torch.Tensor
        Input states to enforce the sign constraint over, shape
        :math:`(N, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Scalar penalty; zero when the constraint already holds everywhere
        in ``states``.

    Examples
    --------
    >>> import torch
    >>> net = torch.nn.Linear(1, 1)
    >>> grid = torch.linspace(0, 1, 10).unsqueeze(1)
    >>> constraint = SignConstraint(sign="+")
    >>> penalty = constraint(net, grid)
    >>> penalty.shape
    torch.Size([])
    """

    def __init__(self, sign: str = "+"):
        super().__init__()
        if sign not in ("+", "-"):
            raise ValueError(f'sign must be "+" or "-", got {sign!r}')
        self._sign = 1.0 if sign == "+" else -1.0

    def forward(self, module: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
        """Compute the sign-violation penalty for ``module`` on ``states``."""
        violation = torch.relu(-self._sign * module(states))
        return (violation**2).mean()
