! SPDX-License-Identifier: MIT
program test_jax
  !! JAX automatic differentiation against the hand-derived native routines:
  !! two independent routes to the same stresses, tangents, R and K.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures
  use shapefuncs
  use tensors, only: eye3
  use materials, only: material, material_set
  use elements, only: elem_ref, elem_mix, shape_gradients
  use jax_materials, only: jax_material, jax_init, jax_set, jax_finalize
  implicit none
  integer :: ncheck = 0, nfail = 0
  ! AD and the closed forms round differently; both are exact derivatives
  real(dp), parameter :: tol = 1024*epsilon(1._dp)
  character(256) :: python
  type(cubature) :: q
  type(shapefunc) :: s
  type(material), allocatable :: nat(:)
  type(jax_material), allocatable :: jx(:)
  real(dp) :: xref(3,8), u(3,8), f(3,3,8), dndx(3,8), dv
  real(dp) :: s1(6,8), c1(6,6,8), s2(6,8), c2(6,6,8), p1(3,3,8), a1(9,9,8), p2(3,3,8), a2(9,9,8)
  real(dp) :: r1(24), k1(24,24), r2(24), k2(24,24)
  integer :: m, a, i, g, stat
  character(:), allocatable :: id

  call get_environment_variable('FE_PYTHON', python, status=stat)
  if (stat /= 0) python = '../.venv/bin/python'
  call jax_init(python, 'python')

  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  do a = 1, 8
    do i = 1, 3
      xref(i,a) = (s%coords(i,a) + 1)/2 + 0.05_dp*cos(real(3*a + i, dp))
      u(i,a) = 0.1_dp*sin(real(5*a + 2*i, dp))
    end do
  end do
  do g = 1, 8
    call shape_gradients(q, s, g, xref, dndx, dv)
    f(:,:,g) = eye3 + matmul(u, transpose(dndx))
  end do

  nat = material_set(1.2_dp, 0.8_dp)
  jx = jax_set(1.2_dp, 0.8_dp, ['ref', 'mix'])
  do m = 1, size(nat)
    id = trim(nat(m)%name) // ' '
    call nat(m)%ref(8, f, s1, c1)
    call jx(m)%ref(8, f, s2, c2)
    call check(close([s2], [s1]), id // 'ref: S, AD = closed form')
    call check(close([c2], [c1]), id // 'ref: CC, AD = closed form')
    call nat(m)%mix(8, f, p1, a1)
    call jx(m)%mix(8, f, p2, a2)
    call check(close([p2], [p1]), id // 'mix: P, AD = closed form')
    call check(close([a2], [a1]), id // 'mix: AA, AD = closed form')
    ! one point at a time gives the same as the batch
    do g = 1, 8
      call jx(m)%mix(1, f(:,:,g), p2(:,:,g), a2(:,:,g))
    end do
    call check(close([a2], [a1]), id // 'mix: batch of 1 = batch of 8')

    call elem_ref(q, s, xref, u, nat(m), r1, k1)
    call elem_ref(q, s, xref, u, jx(m), r2, k2)
    call check(close(r2, r1) .and. close([k2], [k1]), id // 'ref: element R, K with JAX = native')
    call elem_mix(q, s, xref, u, nat(m), r1, k1)
    call elem_mix(q, s, xref, u, jx(m), r2, k2)
    call check(close(r2, r1) .and. close([k2], [k1]), id // 'mix: element R, K with JAX = native')
  end do

  call jax_finalize()
  deallocate(nat, jx, id)
  print '(i0, a, i0, a)', ncheck - nfail, ' of ', ncheck, ' checks passed'
  if (nfail > 0) error stop 1

contains

subroutine check(cond, name)
  logical, intent(in) :: cond
  character(*), intent(in) :: name
  ncheck = ncheck + 1
  if (.not. cond) then
    nfail = nfail + 1
    print '(a)', 'FAIL: ' // name
  end if
end subroutine check

pure logical function close(x, y)
  ! max-norm comparison relative to the size of y
  real(dp), intent(in) :: x(:), y(:)
  close = maxval(abs(x - y)) <= tol*maxval(abs(y))
end function close

end program test_jax
