! SPDX-License-Identifier: MIT
program bench_jax
  !! Native versus JAX material evaluation (harness%bench_paradigm).
  !! Usage: ./run.sh run --profile release [-- nelem [ntrial]]
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature, CUB_HEX
  use shapefuncs, only: shapefunc
  use materials, only: material, material_set
  use harness, only: bench_paradigm, int_arg
  use jax_materials, only: jax_material, jax_init, jax_set, jax_finalize, jax_ping, wtime
  implicit none
  integer :: nelem = 200, ntrial = 5, stat
  character(256) :: python
  type(cubature) :: q
  type(shapefunc) :: s
  type(material) :: nat(4)
  type(jax_material) :: jx(4)

  call int_arg(1, nelem)
  call int_arg(2, ntrial)
  call get_environment_variable('FE_PYTHON', python, status=stat)
  if (stat /= 0) python = '../.venv/bin/python'
  call jax_init(python, 'python')
  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  nat = material_set(1.2_dp, 0.8_dp)
  jx = jax_set(1.2_dp, 0.8_dp, ['ref', 'cur', 'mix'], q%npoints)
  call bench_paradigm(nat, jx, q, s, nelem, ntrial, wtime, jax_ping, 'JAX')
  call jax_finalize()
end program bench_jax
