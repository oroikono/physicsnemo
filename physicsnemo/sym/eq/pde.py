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

"""Base class for symbolic PDE definitions."""

from __future__ import annotations

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.sym.computation import Computation

sympy = OptionalImport("sympy")


class PDE:
    """Base class for all partial differential equations.

    Subclasses must populate ``self.equations`` (a ``dict[str, sympy.Expr]``)
    and set ``self.dim``.

    Examples
    --------
    Define a 2-D advection-diffusion equation:

    >>> from sympy import Symbol, Function, Number
    >>> from physicsnemo.sym.eq.pde import PDE
    >>>
    >>> class AdvectionDiffusion(PDE):
    ...     def __init__(self, D=0.1):
    ...         self.dim = 2
    ...         x, y = Symbol("x"), Symbol("y")
    ...         T = Function("T")(x, y)
    ...         u = Function("u")(x, y)
    ...         v = Function("v")(x, y)
    ...         self.equations = {
    ...             "advection_diffusion": (
    ...                 u * T.diff(x) + v * T.diff(y)
    ...                 - D * (T.diff(x, 2) + T.diff(y, 2))
    ...             ),
    ...         }
    ...
    >>> pde = AdvectionDiffusion(D=0.01)
    >>> pde.pprint()
    advection_diffusion: ...
    """

    name = "PDE"

    def pprint(self, print_latex: bool = False) -> None:
        """Pretty-print the equations."""
        sympy.init_printing(use_latex=True)
        for key, value in self.equations.items():
            print(str(key) + ": " + str(value))
        if print_latex:
            sympy.preview(
                sympy.Matrix(
                    [
                        sympy.Eq(sympy.Function(name, real=True), eq)
                        for name, eq in self.equations.items()
                    ]
                ),
                mat_str="cases",
                mat_delim="",
            )

    def subs(self, x, y):
        """Substitute *x* with *y* in all equations (calls SymPy ``subs``)."""
        for name, eq in self.equations.items():
            self.equations[name] = eq.subs(x, y).doit()

    def make_computations(
        self,
        detach_names: list[str] | None = None,
        discover_terms: list[str] | None = None,
    ) -> list[Computation]:
        r"""Convert each equation into a :class:`Computation`.

        Parameters
        ----------
        detach_names : list[str] or None
            Names to detach from the autograd graph before the residual is
            evaluated (see :meth:`Computation.from_sympy`).
        discover_terms : list[str] or None
            Names of the quantities being discovered -- typically the
            output of a network standing in for an unknown coefficient or
            an unknown term. Every *other* name referenced by the
            equations is automatically added to ``detach_names``, so the
            physics loss shapes only the discovered quantities without
            hand-enumerating every known field and its derivatives (this
            generalizes the ``freeze_terms`` positional-index mechanism in
            :meth:`Computation.from_sympy` to stable, named terms).
            Combines with an explicit ``detach_names`` via union.

        Returns
        -------
        list[Computation]
            One computation per equation in ``self.equations``.

        Raises
        ------
        ValueError
            If a name in ``discover_terms`` is not referenced by any
            equation in ``self.equations``.

        Examples
        --------
        >>> from sympy import Symbol, Function
        >>> class Closure(PDE):
        ...     def __init__(self):
        ...         x, t = Symbol("x"), Symbol("t")
        ...         u = Function("u")(x, t)
        ...         r = Function("R")(x, t)
        ...         self.equations = {"residual": u.diff(t) - r}
        ...
        >>> pde = Closure()
        >>> pde.make_computations(discover_terms=["R"])[0].evaluate.detach_names
        ['u__t']
        """
        if detach_names is None:
            detach_names = []
        if discover_terms:
            discover_set = set(discover_terms)
            referenced: set[str] = set()
            for eq in self.equations.values():
                referenced |= Computation.free_symbol_names(eq)
            unknown = discover_set - referenced
            if unknown:
                raise ValueError(
                    f"discover_terms {sorted(unknown)} not referenced by any "
                    f"equation in {sorted(self.equations)}"
                )
            detach_names = sorted((set(detach_names) | referenced) - discover_set)

        return [
            Computation.from_sympy(eq, str(name), detach_names=detach_names)
            for name, eq in self.equations.items()
        ]
