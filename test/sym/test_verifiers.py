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

import pytest
import torch

from physicsnemo.sym.eq.verifiers import SignConstraint, equilibrium_verifier


class _Const(torch.nn.Module):
    """Outputs a constant, independent of input, via a trainable parameter."""

    def __init__(self, value):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([[float(value)]]))

    def forward(self, x):
        return self.bias.expand(x.shape[0], 1)


def test_equilibrium_verifier_zero_loss_when_module_matches():
    net = _Const(0.0)
    states = torch.tensor([[0.0], [1.0]])
    loss = equilibrium_verifier(net, states, values=0.0)
    assert loss.item() == pytest.approx(0.0)


def test_equilibrium_verifier_mse_against_known_values():
    net = _Const(3.0)
    states = torch.zeros(4, 1)
    loss = equilibrium_verifier(net, states, values=1.0)
    assert loss.item() == pytest.approx(4.0)


def test_equilibrium_verifier_accepts_tensor_values():
    net = _Const(2.0)
    states = torch.zeros(2, 1)
    values = torch.tensor([[0.0], [4.0]])
    loss = equilibrium_verifier(net, states, values=values)
    # mean((2-0)**2, (2-4)**2) = mean(4, 4) = 4.0
    assert loss.item() == pytest.approx(4.0)


def test_equilibrium_verifier_gradient_flows_into_module():
    net = _Const(5.0)
    loss = equilibrium_verifier(net, torch.zeros(1, 1), values=0.0)
    loss.backward()
    assert net.bias.grad is not None
    assert net.bias.grad.item() != 0.0


def test_sign_constraint_invalid_sign_raises():
    with pytest.raises(ValueError, match="sign"):
        SignConstraint(sign="bogus")


def test_sign_constraint_zero_when_already_nonnegative():
    net = _Const(1.0)
    grid = torch.linspace(0, 1, 5).unsqueeze(1)
    penalty = SignConstraint(sign="+")(net, grid)
    assert penalty.item() == pytest.approx(0.0)


def test_sign_constraint_penalizes_negative_values():
    net = _Const(-2.0)
    grid = torch.linspace(0, 1, 5).unsqueeze(1)
    penalty = SignConstraint(sign="+")(net, grid)
    assert penalty.item() == pytest.approx(4.0)


def test_sign_constraint_negative_sign_flips_direction():
    net = _Const(2.0)
    grid = torch.linspace(0, 1, 5).unsqueeze(1)
    # sign="-" enforces module(states) <= 0; a positive constant violates it.
    penalty = SignConstraint(sign="-")(net, grid)
    assert penalty.item() == pytest.approx(4.0)


def test_sign_constraint_gradient_flows_into_module():
    net = _Const(-1.0)
    grid = torch.linspace(0, 1, 5).unsqueeze(1)
    penalty = SignConstraint(sign="+")(net, grid)
    penalty.backward()
    assert net.bias.grad is not None
    assert net.bias.grad.item() != 0.0
