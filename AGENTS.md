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
| `src/materials.f90` | Stress and tangent per model (`svk`, `nh`, `mnh`, `mr`) and configuration (`_ref/_cur/_mix`); `material_set` |
| `src/materials_ad.f90` | **Generated, never edit.** The same four models differentiated by JAX and exported as Fortran (`material_set_ad`); regenerate with `jax/python/export_fortran.py` |
| `src/elements.f90` | Element R and K per configuration (`elem_ref/_cur/_mix`) |
| `test/test_fe_assembly.f90` | The test program |
| `app/main.f90` | `bench`, the fpm app: times element R and K and the material call alone |
| `notes/notes.typ` | Typst notes: notation, general relations, element R and K; includes the models |
| `notes/<model>.typ` | Stress and tangent of one material model in the three configurations |
| `notes/defs.typ` | Shared math symbols, imported by every notes file |
| `jax/` | Subpackage: JAX automatic-differentiation paradigm through embedded Python; Linux, `../.venv`, not in CI |
| `jax/python/fe_jax.py` | Energies Psi(I1, I2, J) and AD stress and tangent per configuration |
| `jax/src/pybridge.c` | Embedded interpreter; Fortran buffers cross as memoryviews |
| `jax/src/jax_materials.f90` | `jax_material` extends `material`, overriding only the batched methods |
| `jax/test/test_jax.f90`, `jax/app/main.f90` | AD against the closed forms; native-versus-JAX bench |
| `socket/` | Subpackage: socket-server paradigm, `python/server.py` (fe_jax behind TCP) and a pure-Fortran POSIX client `socket_materials.f90`; Linux, not in CI |
| `ci.sh` | Local check with every installed compiler |
| `.github/workflows/ci.yml` | GitHub CI: compiler × OS matrix |
| `CLAUDE.local.md` | Maintainer's goal and specification; gitignored |

Directories that come with later work, created when first needed: `example/`.

## Commands

```sh
fpm test --profile debug             # fast check, gfortran
fpm test --profile release
fpm run --profile release            # benchmark: 2000 elements, best of 7 trials
fpm run --profile release -- 500 3   # nelem, ntrial
perf stat -e cycles,instructions $(fpm run --profile release --runner echo)   # IPC
valgrind $(fpm run --profile release --runner echo) 100 1                     # heap allocations
./ci.sh                              # gfortran, ifx, nvfortran if installed
typst compile notes/notes.typ build/notes/notes.pdf   # notes; output is not committed
(cd jax && ../.venv/bin/python python/export_fortran.py)   # regenerate src/materials_ad.f90 after editing an energy
jax/run.sh test --profile release    # JAX paradigm: AD against closed forms (needs ../.venv with jax)
jax/run.sh run --profile release     # native versus JAX: JIT, batch 1 / 8 / mesh, element
(cd socket && fpm test)              # socket paradigm: starts python/server.py on port 65400
(cd socket && fpm run --profile release)   # native versus socket server
```

Done means: the tests pass for every compiler you have in both profiles, and
the app runs.

## Decided conventions (do not change without the maintainer)

- Kinds come from `iso_fortran_env`; all reals are `real64`.
- Shape functions and cubatures come from fortran_shapefuncs and fortran_cubatures, pinned by tag.
- Error policy: `error stop` with a message that starts with the procedure name.
- Tensor notation follows Holzapfel, with fourth-order products defined by index in `notes/notes.typ`:
  `(A ⊗̄ B)_ijkl = A_ik B_jl`, `(A ⊗̲ B)_ijkl = A_il B_jk`.
- Each configuration is computed discretely in its own quantities: no code pushes forward or
  pulls back from another configuration. The updated Lagrangian element takes dN/dx and dv from
  the current coordinates and F from F^-1 = I - du/dx. Equality of R and K across configurations
  is a test, never an implementation shortcut.
- Kernels for different configurations are written in the same style (explicit fixed-length
  loops), so timing differences come from the algorithm, not the coding.
- Each configuration uses its natural storage: Voigt 6 and 6x6 for the reference (S, CC) and
  current (sigma, cc) configurations, 3x3 and 9x9 for the two-point one (P, AA).
- Voigt order 11, 22, 33, 12, 23, 13. Stress-like vectors hold components; strain-like vectors hold
  2 E_12 etc., so 6x6 tangents hold tensor components unscaled and the factor 2 lives in the shear
  rows of the B-matrix. Two-point index (i, J) -> i + 3 (J - 1), column-major.
- Element dof order: node, then direction, `3 (a - 1) + i`.
- Material evaluation is batched: elements gather F at all cubature points, make one call
  `mat%ref|cur|mix(n, F(3,3,n), stress(...,n), tangent(...,n))`, then assemble. Native models
  implement it by looping over per-point routines (`voigt_material`, `twopoint_material`
  interfaces); other paradigms extend type `material` and override only these three methods, so
  the element loop is identical across paradigms and only the material call changes.
- `src/materials_ad.f90` is generated from `jax/python/fe_jax.py` and committed, like fypp output
  in the package standard: CI has no JAX, so it compiles the committed file and does not check it
  is current. Regenerate and commit it with every energy change.
- Every tangent is tested by central differences of the residual, and every residual by central
  differences of the element energy; R and K must agree across the three configurations.

## Deviations from the package standard

- Several modules in `src/`, one per concern (kinematics, materials, element, timing), instead of one module per package: the sweep is 4 materials × 3 configurations × 3 evaluation paradigms.
- Not yet published (no release, Zenodo, Pages): no `llms.txt`, `tools/check_docs.py`, man page, `CITATION.cff`, `CONTRIBUTING.md`, `ford.md`, `docs.yml` or `Makefile`. Add them when the maintainer decides to publish.
- stdlib, FLAP and fplotlib are planned dependencies; add each when code first uses it.

## Benchmark rules

- Report the best of several trials (OS noise only adds time); every result feeds a checksum so
  no call is optimized away.
- The native bench times with `cpu_time`: ifx's `system_clock` follows the wall clock, which WSL2
  re-syncs with jumps (a sweep once measured -1 s).
- The material-only timing calls the same procedure pointer on the same deformation gradients as
  the element loop, so the difference is the element's own work.
- Under WSL2 perf exposes cycles and instructions but no floating-point counters; FLOPs are
  counted analytically.
- Paradigm benches report the transport alone (a no-op request of the same size), so kernel and
  overhead separate. Under WSL2 a TCP loopback round trip costs about 75 us whatever the size
  (process wake-up latency); compare socket results against that floor.

## Test rules

- Debug profiles trap floating-point exceptions (ifx `-fpe0`, nvfortran `-Ktrap=fp`): tests must not create NaN or Inf through arithmetic, nor compare NaN with `<`. Use `ieee_value` and `ieee_is_nan`.
- Use tolerances relative to `epsilon(1._dp)` (`tol = 64*eps`); `-O3` changes the last bits.
  Finite-difference checks use `fdtol = 64*sqrt(eps)` with step `h = 1e-5`.
- Tests must run clean under AddressSanitizer and UBSan (gfortran and ifx debug on Linux).

## Commits

Commit as the maintainer, Will Klausler, with the GitHub no-reply address
`226649647+willklausler@users.noreply.github.com` (set in the repository's git config).
No `Co-authored-by` trailers.
