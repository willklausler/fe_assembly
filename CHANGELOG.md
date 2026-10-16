# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org).

## [Unreleased]

- Project infrastructure: fpm package with per-compiler debug and release profiles, `ci.sh`, GitHub CI on gfortran, ifx, LLVM flang and nvfortran.
- Depends on fortran_shapefuncs v0.2.0 (and through it fortran_cubatures v0.2.0).
- Element K assembled in 3x3 node blocks with the tangent-gradient product reused per node and the lower blocks mirrored; no runtime-sized temporaries in the element loop.
- App `bench`: time per element for R and K, and for the material call alone, per material and configuration.
- Typst notes: stresses and tangents of Saint Venant-Kirchhoff, Neo-Hooke, modified Neo-Hooke and Mooney-Rivlin in three configurations.
- Modules `tensors`, `materials`, `elements`: Saint Venant-Kirchhoff, Neo-Hooke, modified Neo-Hooke and Mooney-Rivlin in the reference, current and two-point configurations, and element R and K for each. `material_set(lam, mu)` gives all four with matching small-strain constants.
