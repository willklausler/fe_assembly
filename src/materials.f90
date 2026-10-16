! SPDX-License-Identifier: MIT
module materials
  !! Hyperelastic stresses and tangents, one routine per model and configuration.
  !!
  !! Every routine takes the deformation gradient F and a parameter array and
  !! returns the stress and tangent in the configuration's natural storage:
  !! *_ref: S and CC in Voigt form (6, 6x6); *_cur: sigma and cc in Voigt form;
  !! *_mix: P (3x3) and AA (9x9, two-point index i + 3 (J - 1)).
  !! Formulas and notation: notes/notes.typ.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use tensors, only: eye3, vi, vj, det3, voigt
  implicit none
  private

  public :: material, material_set
  public :: energy_function, voigt_material, twopoint_material
  public :: svk_energy, svk_ref, svk_cur, svk_mix

  type :: material
    !! one model: its name, parameters and routines for every configuration
    character(16) :: name = ''
    real(dp), allocatable :: par(:)
    procedure(energy_function), pointer, nopass :: energy => null()
    procedure(voigt_material), pointer, nopass :: ref => null(), cur => null()
    procedure(twopoint_material), pointer, nopass :: mix => null()
  end type material

  abstract interface
    pure function energy_function(f, par) result(w)
      !! strain energy per reference volume
      import :: dp
      real(dp), intent(in) :: f(3,3), par(:)
      real(dp) :: w
    end function energy_function
    pure subroutine voigt_material(f, par, s, c)
      !! stress s(6) and tangent c(6,6) in Voigt form
      import :: dp
      real(dp), intent(in) :: f(3,3), par(:)
      real(dp), intent(out) :: s(6), c(6,6)
    end subroutine voigt_material
    pure subroutine twopoint_material(f, par, p, a)
      !! first Piola-Kirchhoff stress p(3,3) and first elasticity tensor a(9,9)
      import :: dp
      real(dp), intent(in) :: f(3,3), par(:)
      real(dp), intent(out) :: p(3,3), a(9,9)
    end subroutine twopoint_material
  end interface

contains

function material_set(lam, mu) result(m)
  !! every model, with parameters that give the same small-strain Lame constants lam, mu
  real(dp), intent(in) :: lam, mu
  type(material) :: m(1)
  m(1)%name = 'svk'
  m(1)%par = [lam, mu]
  m(1)%energy => svk_energy
  m(1)%ref => svk_ref
  m(1)%cur => svk_cur
  m(1)%mix => svk_mix
end function material_set

pure function iso4(x, lam, mu) result(c)
  ! Voigt form of lam X (x) X + mu (X (x)bar X + X (x)under X)
  real(dp), intent(in) :: x(3,3), lam, mu
  real(dp) :: c(6,6)
  integer :: a, b
  do b = 1, 6
    do a = 1, 6
      c(a,b) = lam*x(vi(a), vj(a))*x(vi(b), vj(b)) &
             & + mu*(x(vi(a), vi(b))*x(vj(a), vj(b)) + x(vi(a), vj(b))*x(vj(a), vi(b)))
    end do
  end do
end function iso4

!=======================================================================
! Saint Venant-Kirchhoff: par = [lambda, mu]
!=======================================================================

pure function svk_energy(f, par) result(w)
  !! strain energy per reference volume
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp) :: w, e(3,3)
  e = (matmul(transpose(f), f) - eye3)/2
  w = par(1)/2*(e(1,1) + e(2,2) + e(3,3))**2 + par(2)*sum(e*e)
end function svk_energy

pure subroutine svk_ref(f, par, s, c)
  !! S and CC
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: e(3,3)
  e = (matmul(transpose(f), f) - eye3)/2
  s = voigt(par(1)*(e(1,1) + e(2,2) + e(3,3))*eye3 + 2*par(2)*e)
  c = iso4(eye3, par(1), par(2))
end subroutine svk_ref

pure subroutine svk_cur(f, par, s, c)
  !! sigma and cc
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: b(3,3), j
  b = matmul(f, transpose(f))
  j = det3(f)
  s = voigt(par(1)*(b(1,1) + b(2,2) + b(3,3) - 3)/2*b + par(2)*(matmul(b, b) - b))/j
  c = iso4(b, par(1), par(2))/j
end subroutine svk_cur

pure subroutine svk_mix(f, par, p, a)
  !! P and AA
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: p(3,3), a(9,9)
  real(dp) :: e(3,3), s(3,3), b(3,3)
  integer :: i, jj, k, l
  e = (matmul(transpose(f), f) - eye3)/2
  s = par(1)*(e(1,1) + e(2,2) + e(3,3))*eye3 + 2*par(2)*e
  b = matmul(f, transpose(f))
  p = matmul(f, s)
  do l = 1, 3
    do k = 1, 3
      do jj = 1, 3
        do i = 1, 3
          a(i + 3*(jj - 1), k + 3*(l - 1)) = eye3(i,k)*s(jj,l) + par(1)*f(i,jj)*f(k,l) &
            & + par(2)*(f(i,l)*f(k,jj) + b(i,k)*eye3(jj,l))
        end do
      end do
    end do
  end do
end subroutine svk_mix

end module materials
