# fe_assembly

[![CI](https://github.com/willklausler/fe_assembly/actions/workflows/ci.yml/badge.svg)](https://github.com/willklausler/fe_assembly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Verification and profiling of element-level assembly for finite-deformation
hyperelasticity in modern Fortran. Each run assembles an element residual
vector and consistent tangent matrix, and measures kernel time against
call overhead.

> Coding agents: contributor instructions are in [AGENTS.md](AGENTS.md).

## Sweep

- Material models: Saint Venant-Kirchhoff, Neo-Hooke, modified Neo-Hooke (volumetric-isochoric split), Mooney-Rivlin.
- Configurations: total Lagrangian (C, S, ℂ), updated Lagrangian (b, σ, 𝕔), mixed (F, P, 𝔸).
- Material evaluation: pure Fortran; Fortran-C-Python bridge with JAX automatic differentiation; socket server in the style of FEAP's Python material.
- Outputs: time per element, FLOPs, memory allocations, scaling plots.

## Build

```sh
fpm test --profile release
fpm run --profile release
```

Tested with gfortran, ifx, LLVM flang and nvfortran on Linux, macOS and Windows.
Shape functions and cubatures come from
[fortran_shapefuncs](https://github.com/willklausler/fortran_shapefuncs) and
[fortran_cubatures](https://github.com/willklausler/fortran_cubatures).

## Status

Version 0.1.0, work in progress; nothing is stable. See the [CHANGELOG](CHANGELOG.md).

## License

[MIT](LICENSE)
