# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org).

## [Unreleased]

- Project infrastructure: fpm package with per-compiler debug and release profiles, `ci.sh`, GitHub CI on gfortran, ifx, LLVM flang and nvfortran.
- Depends on fortran_shapefuncs v0.2.0 (and through it fortran_cubatures v0.2.0).
- App `volume_check`: integrated hex8 volume under homogeneous deformation equals det(F).
- Typst notes: stresses and tangents of Saint Venant-Kirchhoff, Neo-Hooke, modified Neo-Hooke and Mooney-Rivlin in three configurations.
- Modules `tensors`, `materials`, `elements`: Saint Venant-Kirchhoff in the reference, current and two-point configurations, and element R and K for each.
