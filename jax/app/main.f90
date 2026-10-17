! SPDX-License-Identifier: MIT
program bench_jax
  !! Native versus JAX material evaluation (harness%bench_paradigm).
  !! Usage: ./run.sh run --profile release [-- --help]
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature
  use shapefuncs, only: shapefunc
  use materials, only: material, material_set
  use harness, only: bench_paradigm, bench_options, parse_options, make_element
  use jax_materials, only: jax_material, jax_init, jax_set, jax_finalize, jax_ping, wtime
  implicit none
  type(bench_options) :: opts
  integer :: stat
  character(256) :: python
  type(cubature) :: q
  type(shapefunc) :: s
  type(material) :: nat(4)
  type(jax_material) :: jx(4)

  opts%nelem = 200
  opts%ntrial = 5
  call parse_options(opts, 'native versus JAX through embedded Python')
  call get_environment_variable('FE_PYTHON', python, status=stat)
  if (stat /= 0) python = '../.venv/bin/python'
  call jax_init(python, 'python')
  call make_element(opts, q, s)
  nat = material_set(1.2_dp, 0.8_dp)
  jx = jax_set(1.2_dp, 0.8_dp, ['ref', 'cur', 'mix'], q%npoints)
  call bench_paradigm(nat, jx, q, s, opts, wtime, jax_ping, 'jax')
  call jax_finalize()
end program bench_jax
