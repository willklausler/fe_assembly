! SPDX-License-Identifier: MIT
program test_jax
  !! JAX automatic differentiation against the hand-derived native routines:
  !! two independent routes to the same stresses, tangents, R and K.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature, CUB_HEX
  use shapefuncs, only: shapefunc
  use materials, only: material, material_set
  use harness, only: compare_paradigm, check_summary
  use jax_materials, only: jax_material, jax_init, jax_set, jax_finalize
  implicit none
  ! AD and the closed forms round differently; both are exact derivatives
  real(dp), parameter :: tol = 1024*epsilon(1._dp)
  character(256) :: python
  type(cubature) :: q
  type(shapefunc) :: s
  type(material) :: nat(4)
  type(jax_material) :: jx(4)
  integer :: stat

  call get_environment_variable('FE_PYTHON', python, status=stat)
  if (stat /= 0) python = '../.venv/bin/python'
  call jax_init(python, 'python')
  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  nat = material_set(1.2_dp, 0.8_dp)
  jx = jax_set(1.2_dp, 0.8_dp, ['ref', 'cur', 'mix'], q%npoints)
  call compare_paradigm(nat, jx, q, s, tol)
  call jax_finalize()
  call check_summary()
end program test_jax
