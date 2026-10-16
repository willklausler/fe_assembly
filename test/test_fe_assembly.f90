! SPDX-License-Identifier: MIT
program test_fe_assembly
  !! Checks for the element-level building blocks.
  !! Avoids NaN/Inf arithmetic so it runs under floating-point trapping.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures
  use shapefuncs
  implicit none
  integer :: ncheck = 0, nfail = 0
  real(dp), parameter :: eps = epsilon(1._dp), tol = 64*eps
  type(cubature) :: q
  type(shapefunc) :: s

  ! Dependencies: trilinear hexahedron ------------------------------------
  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  call check(s%nnodes == 8 .and. s%npoints == 8, 'hex8: 8 nodes, 2x2x2 points')
  call check(abs(sum(q%weights) - 8) < tol, 'hex8: weights sum to reference volume 8')
  call check(all(abs(sum(s%func, dim=1) - 1) < tol), 'hex8: partition of unity')
  call check(all(abs(sum(s%derv, dim=2)) < tol), 'hex8: derivatives sum to zero')

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

end program test_fe_assembly
