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
from sympy import Function, Number, Symbol

from physicsnemo.sym.eq.pde import PDE


class Closure(PDE):
    """u_t - R = 0, with R the discovered term."""

    def __init__(self):
        x, t = Symbol("x"), Symbol("t")
        u = Function("u")(x, t)
        r = Function("R")(x, t)
        self.equations = {"residual": u.diff(t) - r}


class NavierStokes(PDE):
    """2D steady Navier-Stokes, matching examples/cfd/inverse_pinns."""

    def __init__(self, nu=0.01):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        u = Function("u")(x, y)
        v = Function("v")(x, y)
        p = Function("p")(x, y)
        nu = Function(nu)(x, y) if isinstance(nu, str) else Number(nu)
        self.equations = {
            "continuity": u.diff(x) + v.diff(y),
            "momentum_x": (
                u * u.diff(x)
                + v * u.diff(y)
                + p.diff(x)
                - nu * (u.diff(x, 2) + u.diff(y, 2))
            ),
            "momentum_y": (
                u * v.diff(x)
                + v * v.diff(y)
                + p.diff(y)
                - nu * (v.diff(x, 2) + v.diff(y, 2))
            ),
        }


def test_discover_terms_auto_derives_detach_names():
    comps = Closure().make_computations(discover_terms=["R"])
    assert comps[0].evaluate.detach_names == ["u__t"]


def test_discover_terms_matches_hand_written_inverse_pinns_list():
    # Regression: mirrors the hand-written detach_names list in
    # examples/cfd/inverse_pinns/train_inverse_pinn.py for the momentum_x
    # residual (minus the harmless extra bare "p", which isn't a required
    # key for this residual either way).
    hand_written = {
        "u",
        "u__x",
        "u__x__x",
        "u__y",
        "u__y__y",
        "v",
        "v__x",
        "v__x__x",
        "v__y",
        "v__y__y",
        "p__x",
        "p__y",
    }
    comps = NavierStokes(nu="nu").make_computations(discover_terms=["nu"])
    momentum_x = next(c for c in comps if c.outputs == ["momentum_x"])
    assert set(momentum_x.evaluate.detach_names) == hand_written


def test_discover_terms_unions_with_explicit_detach_names():
    comps = Closure().make_computations(detach_names=["extra"], discover_terms=["R"])
    assert set(comps[0].evaluate.detach_names) == {"u__t", "extra"}


def test_discover_terms_unknown_name_raises():
    with pytest.raises(ValueError, match="nope"):
        Closure().make_computations(discover_terms=["nope"])


def test_discover_terms_omitted_preserves_default_behavior():
    comps = Closure().make_computations()
    assert comps[0].evaluate.detach_names == []


def test_discover_terms_gradient_flows_only_into_discovered_term():
    comp = Closure().make_computations(discover_terms=["R"])[0]
    u_t = torch.tensor([[1.0]], requires_grad=True)
    r = torch.tensor([[2.0]], requires_grad=True)
    out = comp.evaluate({"u__t": u_t, "R": r})["residual"]
    out.sum().backward()
    assert u_t.grad is None
    assert r.grad is not None
    assert r.grad.item() == pytest.approx(-1.0)
