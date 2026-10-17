! SPDX-License-Identifier: MIT
module tensors
  !! Small 3x3 tensor helpers and the Voigt index map.
  !!
  !! Voigt order 11, 22, 33, 12, 23, 13. Stress-like vectors hold the components;
  !! strain-like vectors hold 2 E_12 etc., so a 6x6 tangent holds the tensor
  !! components unscaled. Two-point tensors flatten column-major: (i, J) -> i + 3 (J - 1).
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  private

  public :: eye3, vi, vj
  public :: det3, inv3, voigt

  real(dp), parameter :: eye3(3,3) = reshape([1._dp, 0._dp, 0._dp, 0._dp, 1._dp, 0._dp, &
                                            & 0._dp, 0._dp, 1._dp], [3, 3])
    !! identity
  integer, parameter :: vi(6) = [1, 2, 3, 1, 2, 1], vj(6) = [1, 2, 3, 2, 3, 3]
    !! Voigt index a holds tensor component (vi(a), vj(a))

contains

pure function det3(a) result(d)
  !! determinant
  real(dp), intent(in) :: a(3,3)
  real(dp) :: d
  d = a(1,1)*(a(2,2)*a(3,3) - a(2,3)*a(3,2)) - a(1,2)*(a(2,1)*a(3,3) - a(2,3)*a(3,1)) &
    & + a(1,3)*(a(2,1)*a(3,2) - a(2,2)*a(3,1))
end function det3

pure subroutine inv3(a, b)
  !! b = a^-1 by the adjugate; the caller ensures det3(a) /= 0
  !! (a subroutine: nvfortran 26.3 returns even a 3x3 function result through the heap)
  real(dp), intent(in) :: a(3,3)
  real(dp), intent(out) :: b(3,3)
  real(dp) :: d
  b(1,1) = a(2,2)*a(3,3) - a(2,3)*a(3,2)
  b(1,2) = a(1,3)*a(3,2) - a(1,2)*a(3,3)
  b(1,3) = a(1,2)*a(2,3) - a(1,3)*a(2,2)
  b(2,1) = a(2,3)*a(3,1) - a(2,1)*a(3,3)
  b(2,2) = a(1,1)*a(3,3) - a(1,3)*a(3,1)
  b(2,3) = a(1,3)*a(2,1) - a(1,1)*a(2,3)
  b(3,1) = a(2,1)*a(3,2) - a(2,2)*a(3,1)
  b(3,2) = a(1,2)*a(3,1) - a(1,1)*a(3,2)
  b(3,3) = a(1,1)*a(2,2) - a(1,2)*a(2,1)
  ! the determinant first: b = b/(... b(1,1) ...) made nvfortran copy b to a heap temporary
  d = a(1,1)*b(1,1) + a(1,2)*b(2,1) + a(1,3)*b(3,1)
  b = b/d
end subroutine inv3

pure subroutine voigt(t, v)
  !! stress-like Voigt vector v of a symmetric tensor t
  !! (a subroutine: nvfortran 26.3 returns array function results through the heap)
  real(dp), intent(in) :: t(3,3)
  real(dp), intent(out) :: v(6)
  integer :: a
  do a = 1, 6
    v(a) = t(vi(a), vj(a))
  end do
end subroutine voigt

end module tensors
