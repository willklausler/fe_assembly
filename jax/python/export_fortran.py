"""Export the JAX-differentiated materials as plain Fortran: ../../src/materials_ad.f90.

JAX traces and differentiates each energy Psi(I1, I2, J) per configuration (fe_jax.FULL);
this script evaluates the resulting jaxpr on arrays of Fortran names instead of numbers.
Every arithmetic primitive emits one scalar statement; shape primitives only rearrange
names; constants fold (the iota/eq masks of trace and eye, multiplications by 0 and 1);
identical statements are shared and unused ones dropped. The generated routines have the
per-point interfaces of src/materials.f90, so no Python or XLA is needed at run time.

Usage (from jax/): ../.venv/bin/python python/export_fortran.py
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import jax  # noqa: E402
from jax.extend import core as jcore  # noqa: E402

import fe_jax  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "materials_ad.f90")
NPAR = {"svk": 2, "nh": 2, "mnh": 2, "mr": 3}
VI, VJ = fe_jax.VI, fe_jax.VJ


class Emitter:
    """Scalar SSA statements with constant folding, sharing and dead-code removal."""

    def __init__(self):
        self.stmts = []  # (name, expression, operand names)
        self.cse = {}

    @staticmethod
    def const(x):
        return isinstance(x, (bool, int, float, np.bool_, np.integer, np.floating))

    @staticmethod
    def lit(x):
        x = float(x)
        s = repr(x).replace("e", "e") + "_dp"
        return f"({s})" if x < 0 else s

    def ref(self, x):
        return self.lit(x) if self.const(x) else x

    def emit(self, expr, deps):
        if expr in self.cse:
            return self.cse[expr]
        name = f"t{len(self.stmts) + 1}"
        self.stmts.append((name, expr, [d for d in deps if not self.const(d)]))
        self.cse[expr] = name
        return name

    def unary(self, op, a):
        if self.const(a):
            return {"neg": lambda v: -v, "log": np.log, "sqrt": np.sqrt}[op](float(a))
        expr = {"neg": f"-{a}", "log": f"log({a})", "sqrt": f"sqrt({a})"}[op]
        return self.emit(expr, [a])

    def binary(self, op, a, b):
        ca, cb = self.const(a), self.const(b)
        if ca and cb:
            a, b = float(a), float(b)
            fold = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
                    "div": lambda: a / b, "pow": lambda: a**b}
            return fold[op]()
        if op == "add":
            if ca and a == 0:
                return b
            if cb and b == 0:
                return a
        elif op == "sub":
            if cb and b == 0:
                return a
            if ca and a == 0:
                return self.unary("neg", b)
        elif op == "mul":
            if (ca and a == 0) or (cb and b == 0):
                return 0.0
            if ca and a == 1:
                return b
            if cb and b == 1:
                return a
            if ca and a == -1:
                return self.unary("neg", b)
            if cb and b == -1:
                return self.unary("neg", a)
        elif op == "div":
            if ca and a == 0:
                return 0.0
            if cb and b == 1:
                return a
        elif op == "pow":
            if cb and b == 1:
                return a
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}[op]
        x, y = (a, b) if op in ("sub", "div", "pow") or str(a) <= str(b) else (b, a)
        return self.emit(f"{self.ref(x)}{sym}{self.ref(y)}", [x, y])

    def total(self, terms):
        acc = 0.0
        for t in terms:
            acc = self.binary("add", acc, t)
        return acc

    def live(self, outputs):
        keep, todo = set(), [o for o in outputs if not self.const(o)]
        deps = {n: d for n, _, d in self.stmts}
        while todo:
            n = todo.pop()
            if n not in keep and n in deps:  # inputs f(i,j), par(k) have no statement
                keep.add(n)
                todo.extend(deps[n])
        return [(n, e) for n, e, _ in self.stmts if n in keep]


def evaluate(em, jaxpr, consts, args):
    """Run a jaxpr on object arrays of names/constants."""
    env = {}

    def read(v):
        if isinstance(v, jcore.Literal):
            return np.asarray(v.val, dtype=object)
        return env[v]

    for v, x in zip(jaxpr.constvars, consts):
        env[v] = np.asarray(x, dtype=object)
    for v, x in zip(jaxpr.invars, args):
        env[v] = x
    for eqn in jaxpr.eqns:
        ins = [read(v) for v in eqn.invars]
        outs = apply(em, eqn.primitive.name, eqn.params, ins, eqn.outvars)
        for v, x in zip(eqn.outvars, outs):
            env[v] = np.asarray(x, dtype=object).reshape(v.aval.shape)
    return [read(v) for v in jaxpr.outvars]


def elementwise(fn, *xs):
    xs = np.broadcast_arrays(*[np.asarray(x, dtype=object) for x in xs])
    out = np.empty(xs[0].shape, dtype=object)
    for idx in np.ndindex(out.shape):
        out[idx] = fn(*[x[idx] for x in xs])
    return out


def apply(em, prim, params, ins, outvars):
    if prim in ("add", "add_any", "sub", "mul", "div", "pow"):
        op = "add" if prim == "add_any" else prim
        return [elementwise(lambda a, b: em.binary(op, a, b), *ins)]
    if prim in ("neg", "log", "sqrt"):
        return [elementwise(lambda a: em.unary(prim, a), ins[0])]
    if prim == "integer_pow":
        y = params["y"]

        def ipow(a):
            if em.const(a):
                return float(a) ** y
            if y == 2:
                return em.binary("mul", a, a)
            return em.binary("pow", a, float(y)) if y < 0 else em.emit(f"{a}**{y}", [a])

        return [elementwise(ipow, ins[0])]
    if prim == "eq":
        return [elementwise(lambda a, b: bool(a == b) if em.const(a) and em.const(b) else _fail(prim), *ins)]
    if prim == "select_n":
        pred, cases = ins[0], ins[1:]
        return [elementwise(lambda p, *c: c[int(p)] if em.const(p) else _fail(prim), pred, *cases)]
    if prim == "convert_element_type":
        return [elementwise(lambda a: float(a) if em.const(a) else a, ins[0])]
    if prim == "iota":
        shape, dim = params["shape"], params["dimension"]
        idx = np.indices(shape)[dim]
        return [idx.astype(object)]
    if prim == "broadcast_in_dim":
        x, shape, dims = ins[0], params["shape"], params["broadcast_dimensions"]
        expand = [1] * len(shape)
        for k, d in enumerate(dims):
            expand[d] = x.shape[k]
        return [np.broadcast_to(x.reshape(expand), shape)]
    if prim == "reshape":
        return [ins[0].reshape(params["new_sizes"])]
    if prim == "squeeze":
        return [np.squeeze(ins[0], axis=tuple(params["dimensions"]))]
    if prim == "transpose":
        return [np.transpose(ins[0], params["permutation"])]
    if prim == "slice":
        st, lim, strides = params["start_indices"], params["limit_indices"], params["strides"]
        strides = strides or (1,) * len(st)
        return [ins[0][tuple(slice(a, b, c) for a, b, c in zip(st, lim, strides))]]
    if prim == "split":
        cuts = np.cumsum(params["sizes"])[:-1]
        return np.split(ins[0], cuts, axis=params["axis"])
    if prim == "pad":
        x, pv = ins[0], ins[1].item()
        cfg = params["padding_config"]
        shape = [lo + hi + n + max(n - 1, 0) * it for (lo, hi, it), n in zip(cfg, x.shape)]
        out = np.full(shape, pv, dtype=object)
        for idx in np.ndindex(x.shape):
            out[tuple(lo + i * (it + 1) for (lo, _, it), i in zip(cfg, idx))] = x[idx]
        return [out]
    if prim == "reduce_sum":
        x, axes = ins[0], tuple(params["axes"])
        keep = [d for d in range(x.ndim) if d not in axes]
        xt = np.transpose(x, keep + list(axes)).reshape([x.shape[d] for d in keep] + [-1])
        out = np.empty(xt.shape[:-1], dtype=object)
        for idx in np.ndindex(out.shape):
            out[idx] = em.total(xt[idx])
        return [out]
    if prim == "dot_general":
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        a, b = ins
        af = [d for d in range(a.ndim) if d not in lc and d not in lb]
        bf = [d for d in range(b.ndim) if d not in rc and d not in rb]
        at = np.transpose(a, list(lb) + af + list(lc))
        bt = np.transpose(b, list(rb) + list(rc) + bf)
        nb, nc = len(lb), len(lc)
        shape = at.shape[: nb + len(af)] + bt.shape[nb + nc :]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            bi, ai, bfi = idx[:nb], idx[nb : nb + len(af)], idx[nb + len(af) :]
            terms = []
            for ci in np.ndindex(at.shape[nb + len(af) :]):
                terms.append(em.binary("mul", at[bi + ai + ci], bt[bi + ci + bfi]))
            out[idx] = em.total(terms)
        return [out]
    if prim in ("jit", "pjit", "closed_call"):
        cj = params["jaxpr"]
        return evaluate(em, cj.jaxpr, cj.consts, ins)
    return _fail(prim)


def _fail(prim):
    raise NotImplementedError(f"export_fortran: primitive {prim} with non-constant operands")


def routine(model, cfg):
    """Fortran source of one per-point routine ad_<model>_<cfg>."""
    em = Emitter()
    full = fe_jax.FULL[cfg]
    f0 = np.eye(3) + 0.1
    closed = jax.make_jaxpr(lambda f, p: full(fe_jax.PSI[model], f, p))(f0, np.ones(NPAR[model]))
    f = np.array([[f"f({i + 1},{j + 1})" for j in range(3)] for i in range(3)], dtype=object)
    par = np.array([f"par({k + 1})" for k in range(NPAR[model])], dtype=object)
    stress, tangent = evaluate(em, closed.jaxpr, closed.consts, [f, par])

    out = []
    if cfg == "mix":
        for jj in range(3):
            for i in range(3):
                out.append((f"p({i + 1},{jj + 1})", stress[i, jj]))
        for s in range(9):
            for r in range(9):
                out.append((f"a({r + 1},{s + 1})", tangent[r, s]))
        args, decl = "p, a", "real(dp), intent(out) :: p(3,3), a(9,9)"
    else:
        for a in range(6):
            out.append((f"s({a + 1})", stress[VI[a], VJ[a]]))
        for b in range(6):
            for a in range(6):
                out.append((f"c({a + 1},{b + 1})", tangent[VI[a], VJ[a], VI[b], VJ[b]]))
        args, decl = "s, c", "real(dp), intent(out) :: s(6), c(6,6)"

    body = em.live([v for _, v in out])
    names = [n for n, _ in body]
    lines = [f"pure subroutine ad_{model}_{cfg}(f, par, {args})",
             f"  !! {model} {cfg}: stress and tangent by JAX AD, {len(body)} statements",
             "  real(dp), intent(in) :: f(3,3), par(:)",
             f"  {decl}"]
    for k in range(0, len(names), 12):
        lines.append("  real(dp) :: " + ", ".join(names[k : k + 12]))
    lines += [f"  {n} = {e}" for n, e in body]
    lines += [f"  {t} = {em.ref(v)}" for t, v in out]
    lines.append(f"end subroutine ad_{model}_{cfg}")
    return "\n".join(lines), len(body)


def main():
    routines, counts = [], {}
    for model in NPAR:
        for cfg in ("ref", "cur", "mix"):
            src, n = routine(model, cfg)
            routines.append(src)
            counts[(model, cfg)] = n
    # one public statement per model: lines stay under the 132-character limit (gfortran 13 enforces it)
    names = "\n".join("  public :: " + ", ".join(f"ad_{m}_{c}" for c in ("ref", "cur", "mix")) for m in NPAR)
    setters = []
    for k, m in enumerate(NPAR, start=1):
        setters += [f"  m({k})%name = '{m}-ad'",
                    f"  m({k})%ref_pt => ad_{m}_ref",
                    f"  m({k})%cur_pt => ad_{m}_cur",
                    f"  m({k})%mix_pt => ad_{m}_mix"]
    text = f"""! SPDX-License-Identifier: MIT
! ======================================================================
! GENERATED FILE - DO NOT EDIT. Edit jax/python/fe_jax.py (the energies) or
! jax/python/export_fortran.py and run it (needs JAX; see AGENTS.md).
! ======================================================================
module materials_ad
  !! The four material models with stresses and tangents differentiated by JAX
  !! and exported as Fortran: the per-point routines of materials.f90, without
  !! Python or XLA at run time. Each configuration is differentiated in its own
  !! quantities, as in fe_jax.py.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use materials, only: material, material_set
  implicit none
  private

  public :: material_set_ad
{names}

contains

function material_set_ad(lam, mu) result(m)
  !! material_set(lam, mu) with the JAX-exported routines; energies stay the native ones
  real(dp), intent(in) :: lam, mu
  type(material) :: m(4)
  m = material_set(lam, mu)
{chr(10).join(setters)}
end function material_set_ad

{(chr(10) + chr(10)).join(routines)}

end module materials_ad
"""
    with open(OUT, "w") as fh:
        fh.write(text)
    for (m, c), n in counts.items():
        print(f"{m:4s} {c}: {n:5d} statements")
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
