! SPDX-License-Identifier: MIT
program bench_socket
  !! Native versus the socket server (fe_jax over TCP) (harness%bench_paradigm).
  !! Usage: fpm run --profile release [-- nelem [ntrial]]
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature, CUB_HEX
  use shapefuncs, only: shapefunc
  use materials, only: material, material_set
  use harness, only: bench_paradigm, int_arg
  use socket_materials, only: socket_material, socket_start, socket_set, socket_stop, socket_ping, wtime
  implicit none
  integer :: nelem = 200, ntrial = 5, stat
  character(256) :: python
  type(cubature) :: q
  type(shapefunc) :: s
  type(material) :: nat(4)
  type(socket_material) :: jx(4)

  call int_arg(1, nelem)
  call int_arg(2, ntrial)
  call get_environment_variable('FE_PYTHON', python, status=stat)
  if (stat /= 0) python = '../.venv/bin/python'
  call socket_start(python, 65400)
  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  nat = material_set(1.2_dp, 0.8_dp)
  jx = socket_set(1.2_dp, 0.8_dp, ['ref', 'cur', 'mix'], q%npoints)
  call bench_paradigm(nat, jx, q, s, nelem, ntrial, wtime, socket_ping, 'sock')
  call socket_stop()
end program bench_socket
