# AGENTS.md

Instructions for coding agents working on this repository. It is a study and
benchmark project, not a library: it assembles element residuals and tangents
for finite-deformation hyperelasticity and measures how fast each approach is.
It follows the maintainer's Fortran package standard (`~/manage_packages.md`)
except where "Deviations from the package standard" below says otherwise.

## Layout

| Path | Role |
|---|---|
| `src/tensors.f90` | 3x3 helpers (`det3`, `inv3`) and the Voigt index map |
| `src/materials.f90` | Stress and tangent per model and configuration (`<model>_ref/_cur/_mix`) |
| `src/elements.f90` | Element R and K per configuration (`elem_ref/_cur/_mix`) |
| `test/test_fe_assembly.f90` | The test program |
| `app/main.f90` | `volume_check`, the fpm app |
| `notes/notes.typ` | Typst notes: notation, general relations, element R and K; includes the models |
| `notes/<model>.typ` | Stress and tangent of one material model in the three configurations |
| `notes/defs.typ` | Shared math symbols, imported by every notes file |
| `ci.sh` | Local check with every installed compiler |
| `.github/workflows/ci.yml` | GitHub CI: compiler × OS matrix |
| `CLAUDE.local.md` | Maintainer's goal and specification; gitignored |

Directories that come with later work, created when first needed: `python/`
(JAX bridge and socket server), `example/`.

## Commands

```sh
fpm test --profile debug             # fast check, gfortran
fpm test --profile release
fpm run --profile release            # the app
./ci.sh                              # gfortran, ifx, nvfortran if installed
typst compile notes/notes.typ build/notes/notes.pdf   # notes; output is not committed
```

Done means: the tests pass for every compiler you have in both profiles, and
the app runs.

## Decided conventions (do not change without the maintainer)

- Kinds come from `iso_fortran_env`; all reals are `real64`.
- Shape functions and cubatures come from fortran_shapefuncs and fortran_cubatures, pinned by tag.
- Error policy: `error stop` with a message that starts with the procedure name.
- Tensor notation follows Holzapfel, with fourth-order products defined by index in `notes/notes.typ`:
  `(A ⊗̄ B)_ijkl = A_ik B_jl`, `(A ⊗̲ B)_ijkl = A_il B_jk`.
- Each configuration uses its natural storage: Voigt 6 and 6x6 for the reference (S, CC) and
  current (sigma, cc) configurations, 3x3 and 9x9 for the two-point one (P, AA).
- Voigt order 11, 22, 33, 12, 23, 13. Stress-like vectors hold components; strain-like vectors hold
  2 E_12 etc., so 6x6 tangents hold tensor components unscaled and the factor 2 lives in the shear
  rows of the B-matrix. Two-point index (i, J) -> i + 3 (J - 1), column-major.
- Element dof order: node, then direction, `3 (a - 1) + i`.
- Material routines share one interface per storage scheme (`voigt_material`, `twopoint_material`):
  `(F, par, stress, tangent)`, so the element loop is identical across evaluation paradigms and only
  the material call changes.
- Every tangent is tested by central differences of the residual, and every residual by central
  differences of the element energy; R and K must agree across the three configurations.

## Deviations from the package standard

- Several modules in `src/`, one per concern (kinematics, materials, element, timing), instead of one module per package: the sweep is 4 materials × 3 configurations × 3 evaluation paradigms.
- Not yet published (no release, Zenodo, Pages): no `llms.txt`, `tools/check_docs.py`, man page, `CITATION.cff`, `CONTRIBUTING.md`, `ford.md`, `docs.yml` or `Makefile`. Add them when the maintainer decides to publish.
- stdlib, FLAP and fplotlib are planned dependencies; add each when code first uses it.

## Test rules

- Debug profiles trap floating-point exceptions (ifx `-fpe0`, nvfortran `-Ktrap=fp`): tests must not create NaN or Inf through arithmetic, nor compare NaN with `<`. Use `ieee_value` and `ieee_is_nan`.
- Use tolerances relative to `epsilon(1._dp)` (`tol = 64*eps`); `-O3` changes the last bits.
  Finite-difference checks use `fdtol = 64*sqrt(eps)` with step `h = 1e-5`.
- Tests must run clean under AddressSanitizer and UBSan (gfortran and ifx debug on Linux).

## Commits

Commit as the maintainer, Will Klausler, with the GitHub no-reply address
`226649647+willklausler@users.noreply.github.com` (set in the repository's git config).
No `Co-authored-by` trailers.
