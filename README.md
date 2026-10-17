# fe_assembly

[![CI](https://github.com/willklausler/fe_assembly/actions/workflows/ci.yml/badge.svg)](https://github.com/willklausler/fe_assembly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Verification and profiling of element-level assembly for finite-deformation
hyperelasticity in modern Fortran: four material models, three kinematic
configurations and several ways of evaluating the material, each checked
against the others and timed per element.

```fortran
use cubatures, only: cubature, CUB_HEX
use shapefuncs, only: shapefunc
use materials, only: material, material_set
use elements, only: elem_mix
q = cubature(CUB_HEX, 2)                ! 2x2x2 Gauss points
s = shapefunc(q, 1)                     ! trilinear hexahedron
mats = material_set(1.2_dp, 0.8_dp)     ! svk, nh, mnh, mr with Lame constants 1.2, 0.8
call elem_mix(q, s, xref, u, mats(4), r, k)   ! Mooney-Rivlin: R(24) = int P : grad N, K(24,24)
```

> Coding agents: contributor instructions are in [AGENTS.md](AGENTS.md).

## Features

- Material models: Saint Venant-Kirchhoff, Neo-Hooke, modified Neo-Hooke (volumetric-isochoric split), Mooney-Rivlin.
- Configurations, each computed in its own quantities: total Lagrangian (C, S, ℂ), updated Lagrangian (b, σ, 𝕔), two-point (F, P, 𝔸).
- Material evaluation: hand-derived Fortran; JAX automatic differentiation exported to Fortran (`materials_ad`, no Python at run time); JAX through embedded Python (`jax/`); a FEAP-style socket server (`socket/`).
- Element tangent assembled in 3x3 node blocks with symmetric fill; one batched material call per element.
- Every residual is checked against finite differences of the element energy, every tangent against finite differences of the residual, and all configurations and evaluation paradigms against each other.
- Typst notes with the stresses and tangents of every model in every configuration.

All reals are `real64`; Voigt order 11, 22, 33, 12, 23, 13; tensor notation follows Holzapfel.

## Installation and use

Requires fpm 0.13 or later; the shape functions and cubatures are fetched as dependencies.

```sh
fpm test --profile release          # all checks
fpm run --profile release           # benchmark: time per element, per model and configuration
./ci.sh                             # every installed compiler, both profiles, and the paradigms below
```

The Python paradigms need a virtual environment `.venv` with `jax` (Linux):

```sh
jax/run.sh test --profile release   # embedded JAX against the closed forms
jax/run.sh run --profile release    # native versus embedded JAX: first-call, batch 1 / element / mesh
(cd socket && fpm run --profile release)   # native versus the socket server
```

## Documentation

- [notes/notes.typ](notes/notes.typ): derivations (`typst compile notes/notes.typ`).
- [AGENTS.md](AGENTS.md): layout, commands and decided conventions.
- [CHANGELOG.md](CHANGELOG.md).

## Compilers

| Compiler | Linux | macOS | Windows |
|---|---|---|---|
| gfortran 14 | ✓ | ✓ | ✓ |
| gfortran 13 | ✓ | | |
| ifx 2026.1 | ✓ | | ✓ |
| nvfortran 26.1 | ✓ | | |
| LLVM flang 21 | ✓ | | |

`jax/` and `socket/` are Linux only and not in CI.

## References

- G. A. Holzapfel, *Nonlinear Solid Mechanics: A Continuum Approach for Engineering*, Wiley, 2000.
- J. C. Simo and T. J. R. Hughes, *Computational Inelasticity*, Springer, 1998.
- S. Hartmann and P. Neff, Polyconvexity of generalized polynomial-type hyperelastic strain energy functions for near-incompressibility, *Int. J. Solids Struct.* 40, 2003.
- J. Bradbury et al., JAX: composable transformations of Python+NumPy programs, 2018.
- R. L. Taylor, FEAP: A Finite Element Analysis Program, University of California, Berkeley.

## Status and support

Version 0.1.0, a study project in progress; nothing is stable. See the [CHANGELOG](CHANGELOG.md).

## License

[MIT](LICENSE)
