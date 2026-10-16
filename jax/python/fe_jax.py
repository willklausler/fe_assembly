"""Hyperelastic stresses and tangents by automatic differentiation of the energy.

Each model is one energy Psi(I1, I2, J, par) in the invariants; par has the same
layout as the Fortran materials.material_set. Each configuration builds the
invariants from its own tensor and differentiates with respect to it, so no
configuration is pushed forward or pulled back from another.

make(model, cfg) returns run(f, par, stress, tangent): four buffers in Fortran
layout, F(3,3,n) in, stress and tangent out, as the Fortran bridge passes them.
"""
import os

# CPU only: the comparison is with single-threaded native Fortran on the same core
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Voigt order 11, 22, 33, 12, 23, 13 (0-based); see src/tensors.f90
VI = np.array([0, 1, 2, 0, 1, 0])
VJ = np.array([0, 1, 2, 1, 2, 2])


def _svk(i1, i2, j, p):
    # tr E = (I1 - 3)/2, tr(E^2) = (I1^2 - 2 I2 - 2 I1 + 3)/4
    return p[0] / 2 * ((i1 - 3) / 2) ** 2 + p[1] * (i1**2 - 2 * i2 - 2 * i1 + 3) / 4


def _nh(i1, i2, j, p):
    return p[1] / 2 * (i1 - 3) - p[1] * jnp.log(j) + p[0] / 2 * jnp.log(j) ** 2


def _u(j, kappa):
    return kappa / 4 * (j**2 - 2 * jnp.log(j) - 1)


def _mnh(i1, i2, j, p):
    return p[1] / 2 * (j ** (-2 / 3) * i1 - 3) + _u(j, p[0])


def _mr(i1, i2, j, p):
    return p[0] * (j ** (-2 / 3) * i1 - 3) + p[1] * (j ** (-4 / 3) * i2 - 3) + _u(j, p[2])


PSI = {"svk": _svk, "nh": _nh, "mnh": _mnh, "mr": _mr}


def _det3(a):
    # cofactor expansion, as tensors.det3: jnp.linalg.det goes through a pivoted LU,
    # whose second derivative made the mixed Mooney-Rivlin tangent 35% slower
    return (a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
            - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
            + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]))


def _invariants(x):
    # I1, I2, J of a symmetric tensor (C or b); written so that f(X) = f(X^T)
    i1 = jnp.trace(x)
    return i1, (i1**2 - jnp.trace(x @ x)) / 2, jnp.sqrt(_det3(x))


def _ref_point(psi, f, p):
    # S = 2 sym(dPsi/dC), CC = 4 sym(d2Psi/dCdC): AD treats the 9 entries of C as independent
    energy = lambda c: psi(*_invariants(c), p)
    c = f.T @ f
    g = jax.grad(energy)(c)
    h = jax.hessian(energy)(c)
    h = (h + h.transpose(1, 0, 2, 3) + h.transpose(0, 1, 3, 2) + h.transpose(1, 0, 3, 2)) / 4
    s = (g + g.T)[VI, VJ]
    cc = 4 * h[VI[:, None], VJ[:, None], VI[None, :], VJ[None, :]]
    return s, cc


def _cur_point(psi, f, p):
    # tau = 2 b dPsi/db (isotropic), then J cc from the Lie derivative of tau:
    # J cc : d = tau' - d tau - tau d with b' = d b + b d, which gives
    # J cc = 2 sym_kl[(dtau/db) . b] - (I(x)bar tau + I(x)under tau + tau(x)bar I + tau(x)under I)/2.
    # Only b and tau enter: nothing is pushed forward from the reference configuration.
    energy = lambda b: psi(*_invariants(b), p)

    def tau(b):
        g = jax.grad(energy)(b)
        return 2 * b @ ((g + g.T) / 2)

    b = f @ f.T
    t = tau(b)
    d = jax.jacfwd(tau)(b)
    d = (d + d.transpose(0, 1, 3, 2)) / 2
    jc = 2 * jnp.einsum("ijkn,nl->ijkl", d, b)
    jc = (jc + jc.transpose(0, 1, 3, 2)) / 2
    eye = jnp.eye(3)
    jc = jc - (jnp.einsum("ik,jl->ijkl", eye, t) + jnp.einsum("il,jk->ijkl", eye, t)
               + jnp.einsum("ik,jl->ijkl", t, eye) + jnp.einsum("il,jk->ijkl", t, eye)) / 2
    j = jnp.sqrt(_det3(b))
    return (t / j)[VI, VJ], jc[VI[:, None], VJ[:, None], VI[None, :], VJ[None, :]] / j


def _mix_point(psi, f, p):
    # P = dPsi/dF, AA = d2Psi/dFdF; I1, I2 from C = F^T F, J = det F
    def energy(f):
        c = f.T @ f
        i1 = jnp.trace(c)
        return psi(i1, (i1**2 - jnp.trace(c @ c)) / 2, _det3(f), p)

    pk = jax.grad(energy)(f)
    a = jax.hessian(energy)(f)
    # two-point index (i, J) -> i + 3 J, i fastest
    return pk, a.transpose(1, 0, 3, 2).reshape(9, 9)


POINT = {"ref": _ref_point, "cur": _cur_point, "mix": _mix_point}


def make(model, cfg):
    """Return run(f, par, stress, tangent) for one model and configuration.

    cfg "noop" returns a run that does nothing: the bridge round trip without JAX.
    """
    if cfg == "noop":
        return lambda f, par, stress, tangent: None
    psi = PSI[model]
    point = POINT[cfg]

    @jax.jit
    def batch(fc, p):
        # fc[g] is F^T: Fortran F(i, J, g) read in C order. Results go back in
        # Fortran order the same way, transposing every per-point array explicitly
        # (anisotropic models would not forgive a silent transpose).
        s, t = jax.vmap(lambda f: point(psi, f, p))(jnp.swapaxes(fc, 1, 2))
        if s.ndim == 3:
            s = jnp.swapaxes(s, 1, 2)
        return s, jnp.swapaxes(t, 1, 2)

    def run(f, par, stress, tangent):
        fc = np.frombuffer(f, dtype=np.float64).reshape(-1, 3, 3)
        n = fc.shape[0]
        s, t = batch(fc, np.frombuffer(par, dtype=np.float64))
        np.frombuffer(stress, dtype=np.float64).reshape((n,) + s.shape[1:])[...] = s
        np.frombuffer(tangent, dtype=np.float64).reshape((n,) + t.shape[1:])[...] = t

    return run
