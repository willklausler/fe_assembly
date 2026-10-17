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
| `src/harness.f90` | Test and bench support (test elements, checks, stopwatch, paradigm checks and table); never used by the kernels |
| `test/test_fe_assembly.f90` | The test program |
| `app/main.f90` | `bench`, the fpm app: element R and K and the material call alone, native and JAX-exported; table and CSV |
| `tools/sweep.py` | Runs every bench for every compiler and paradigm into `results/<timestamp>/`: CSV, `env.json`, `summary.md`, plots |
| `notes/notes.typ` | Typst notes: notation, general relations, element R and K; includes the models |
| `notes/<model>.typ` | Stress and tangent of one material model in the three configurations |
| `notes/defs.typ` | Shared math symbols, imported by every notes file |
| `jax/` | Subpackage: JAX automatic-differentiation paradigm through embedded Python; Linux, `../.venv`, not in CI |
| `jax/python/fe_jax.py` | Energies Psi(I1, I2, J) and AD stress and tangent per configuration |
| `jax/python/export_fortran.py` | Writes `src/materials_ad.f90`; `--check` compares its source hash |
| `jax/src/pybridge.c` | Embedded interpreter; Fortran buffers cross as memoryviews |
| `jax/src/jax_materials.f90` | `jax_material` extends `remote_material`: `make` and `evaluate` only |
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
fpm run --profile release -- --help  # options: --nelem --ntrial --csv --tag --models --cfgs --paradigms
.venv/bin/python tools/sweep.py      # whole sweep: all compilers and paradigms, CSV, summary.md, plots
.venv/bin/python tools/sweep.py --allocations   # also heap allocations per element (valgrind)
perf stat -e cycles,instructions $(fpm run --profile release --runner echo)   # IPC
./ci.sh                              # every installed compiler, both profiles; jax/ and socket/ if .venv exists
python3 jax/python/export_fortran.py --check   # generated file is current (no JAX needed)
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
- Error policy: element routines return an optional `status` (`ELEM_OK`, `ELEM_INVERTED` for
  det F <= 0 at a point, `ELEM_DEGENERATE` for a degenerate reference element) with R = K = 0, so
  a nonlinear solver can cut its step; without `status` they stop. This kinematics check is the
  only one: material routines assume det F > 0. Setup and API misuse use `error stop` with a
  message that starts with the procedure name.
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
  interfaces). Paradigms that evaluate elsewhere extend `remote_material` and supply only `make`
  and `evaluate`; `remote_setup` compiles and verifies every handle once, so production calls
  carry no checks beyond the transport's return code.
- Checks live in the test programs and in one-time setup; the kernels and production calls do not
  re-verify. Test and bench code shared between packages lives in `src/harness.f90`.
- `src/materials_ad.f90` is generated from `jax/python/fe_jax.py` and committed, like fypp output
  in the package standard. Its header records a hash of the generator sources; CI and `ci.sh` run
  `export_fortran.py --check` (no JAX needed). Regenerate and commit it with every change to either
  source, including comments.
- Every tangent is tested by central differences of the residual, and every residual by central
  differences of the element energy; R and K must agree across the three configurations.

## Adding or changing a material model

1. Write the energy in `jax/python/fe_jax.py` (`PSI`, `NPAR` in `export_fortran.py`) and regenerate
   `src/materials_ad.f90`.
2. Derive the stresses and tangents per configuration in `notes/<model>.typ` and check them with
   automatic differentiation before writing Fortran.
3. Add the per-point routines `<model>_energy/_ref/_cur/_mix` to `src/materials.f90` and an entry in
   `material_set`, with parameters matched to the small-strain Lame constants.
4. The test then covers the model automatically (finite differences, configurations, AD export,
   rigid rotation, small strain); run `./ci.sh` and update `CHANGELOG.md`.

## Compiler quirks that shaped the code

- gfortran mallocs automatic arrays unless `-fstack-arrays` (added to the gfortran release flags),
  and mallocs temporaries for array-valued helpers in long expressions: tangents accumulate in place.
- gfortran 14 leaks the allocatable components of function results inside an array constructor
  `[f(), g()]` (LeakSanitizer): fill arrays by sections.
- gfortran passes internal procedures as arguments through stack trampolines, which crash under
  AddressSanitizer: `harness%stopwatch` is a loop, not a procedure argument.
- gfortran 13 enforces the 132-character free-form line limit; gfortran 14 accepted longer lines.
- nvfortran 26.3 loses the procedure pointers of an array section passed to a `class(material)`
  dummy: pass whole arrays.
- nvfortran 26.3 returns array function results through the heap, and copies an array to a heap
  temporary when its own elements appear on the right of an assignment to it: `inv3` and `voigt` are
  subroutines. `-Mstack_arrays` is set but did not remove its remaining 2 allocations per point in
  svk cur/mix and mr ref (measured with `tools/sweep.py --allocations`).
- nvfortran 26.3 with `-Mstandard` emits FLAP's type descriptors twice (link failure): dropped from
  the nvfortran debug flags.
- ifx's `system_clock` follows the wall clock, which WSL2 re-syncs with jumps: the native bench uses
  `cpu_time`, the paradigm benches `clock_gettime(CLOCK_MONOTONIC)`.
- oneAPI `setvars.sh` silently does nothing when run twice in one environment: `ci.sh` uses `--force`
  when `ifx` is not on PATH.

## Deviations from the package standard

- Several modules in `src/`, one per concern (tensors, materials, generated materials, elements, harness), instead of one module per package: the sweep is 4 materials × 3 configurations × several evaluation paradigms.
- Two subpackages, `jax/` and `socket/`, keep Python out of the main package and its CI.
- Not yet published (no release, Zenodo, Pages): no `llms.txt`, `tools/check_docs.py`, man page, `CITATION.cff`, `CONTRIBUTING.md`, `ford.md`, `docs.yml` or `Makefile`. Add them when the maintainer decides to publish.
- FLAP (pinned v2.5.3; it pins neither FACE nor PENF) parses the bench command lines. stdlib and
  fplotlib remain planned; plots come from matplotlib in `tools/sweep.py`.

## Benchmark rules

- Report the best of several trials (OS noise only adds time) with `harness%stopwatch`, which
  ignores non-positive intervals; every result feeds a checksum so no call is optimized away.
- Benches share one command line (`harness%parse_options`, FLAP) and write CSV in long format:
  `paradigm,compiler,model,cfg,element,order,nnodes,npoints,batch,quantity,value`, with the compiler
  version and options as `#` comments. Quantities: `element_us`, `material_ns_per_point`,
  `first_call_s`, `transport_us_per_call`, and from `tools/sweep.py --allocations`
  `heap_allocs_per_element`. `tools/sweep.py` records the machine and versions in `env.json`.
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
