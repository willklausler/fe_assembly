! SPDX-License-Identifier: MIT
program bench_socket
  !! Native versus the socket server (fe_jax over TCP) (harness%bench_paradigm).
  !! Usage: fpm run --profile release [-- --help]
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature
  use shapefuncs, only: shapefunc
  use materials, only: material, material_set
  use harness, only: bench_paradigm, bench_options, parse_options, make_element
  use socket_materials, only: socket_material, socket_start, socket_set, socket_stop, socket_ping, wtime
  implicit none
  type(bench_options) :: opts
  integer :: stat
  character(256) :: python
  type(cubature) :: q
  type(shapefunc) :: s
  type(material) :: nat(4)
  type(socket_material) :: jx(4)

  opts%nelem = 200
  opts%ntrial = 5
  call parse_options(opts, 'native versus the fe_jax socket server')
  call get_environment_variable('FE_PYTHON', python, status=stat)
  if (stat /= 0) python = '../.venv/bin/python'
  call socket_start(python, 65400)
  call make_element(opts, q, s)
  nat = material_set(1.2_dp, 0.8_dp)
  jx = socket_set(1.2_dp, 0.8_dp, ['ref', 'cur', 'mix'], q%npoints)
  call bench_paradigm(nat, jx, q, s, opts, wtime, socket_ping, 'socket')
  call socket_stop()
end program bench_socket
