! SPDX-License-Identifier: MIT
module materials
  !! Hyperelastic stresses and tangents, one routine per model and configuration.
  !!
  !! Every routine takes the deformation gradient F and a parameter array and
  !! returns the stress and tangent in the configuration's natural storage:
  !! *_ref: S and CC in Voigt form (6, 6x6); *_cur: sigma and cc in Voigt form;
  !! *_mix: P (3x3) and AA (9x9, two-point index i + 3 (J - 1)).
  !! Formulas and notation: notes/notes.typ.
  !! Models with ln J stop with error stop when det F <= 0 (an inverted element).
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use tensors, only: eye3, vi, vj, det3, inv3, voigt
  implicit none
  private

  public :: material, material_set
  public :: energy_function, voigt_material, twopoint_material
  public :: svk_energy, svk_ref, svk_cur, svk_mix
  public :: nh_energy, nh_ref, nh_cur, nh_mix
  public :: mnh_energy, mnh_ref, mnh_cur, mnh_mix
  public :: mr_energy, mr_ref, mr_cur, mr_mix

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
  type(material) :: m(4)
  real(dp) :: kappa
  kappa = lam + 2*mu/3
  m(1)%name = 'svk'
  m(1)%par = [lam, mu]
  m(1)%energy => svk_energy
  m(1)%ref => svk_ref
  m(1)%cur => svk_cur
  m(1)%mix => svk_mix
  m(2)%name = 'nh'
  m(2)%par = [lam, mu]
  m(2)%energy => nh_energy
  m(2)%ref => nh_ref
  m(2)%cur => nh_cur
  m(2)%mix => nh_mix
  m(3)%name = 'mnh'
  m(3)%par = [kappa, mu]
  m(3)%energy => mnh_energy
  m(3)%ref => mnh_ref
  m(3)%cur => mnh_cur
  m(3)%mix => mnh_mix
  ! mu = 2 (c1 + c2); the split c2 = mu/8 is arbitrary but keeps both terms active
  m(4)%name = 'mr'
  m(4)%par = [3*mu/8, mu/8, kappa]
  m(4)%energy => mr_energy
  m(4)%ref => mr_ref
  m(4)%cur => mr_cur
  m(4)%mix => mr_mix
end function material_set

!-----------------------------------------------------------------------
! Fourth-order products in Voigt (6x6) and two-point (9x9) storage
!-----------------------------------------------------------------------

pure function outer6(a, b) result(c)
  ! A (x) B of symmetric A, B
  real(dp), intent(in) :: a(3,3), b(3,3)
  real(dp) :: c(6,6), va(6), vb(6)
  integer :: k
  va = voigt(a)
  vb = voigt(b)
  do k = 1, 6
    c(:,k) = va*vb(k)
  end do
end function outer6

pure function isym(a) result(c)
  ! II_A = (A (x)bar A + A (x)under A)/2
  real(dp), intent(in) :: a(3,3)
  real(dp) :: c(6,6)
  c = iso4(a, 0._dp, 0.5_dp)
end function isym

pure function outer9(a, b) result(c)
  ! (A (x) B)_iJkL = A_iJ B_kL
  real(dp), intent(in) :: a(3,3), b(3,3)
  real(dp) :: c(9,9)
  integer :: s
  do s = 1, 9
    c(:,s) = reshape(a, [9])*b(mod(s - 1, 3) + 1, (s - 1)/3 + 1)
  end do
end function outer9

pure function over9(a, b) result(c)
  ! (A (x)bar B)_iJkL = A_ik B_JL
  real(dp), intent(in) :: a(3,3), b(3,3)
  real(dp) :: c(9,9)
  integer :: i, jj, k, l
  do l = 1, 3
    do k = 1, 3
      do jj = 1, 3
        do i = 1, 3
          c(i + 3*(jj - 1), k + 3*(l - 1)) = a(i,k)*b(jj,l)
        end do
      end do
    end do
  end do
end function over9

pure function under9(a, b) result(c)
  ! (A (x)under B)_iJkL = A_iL B_Jk
  real(dp), intent(in) :: a(3,3), b(3,3)
  real(dp) :: c(9,9)
  integer :: i, jj, k, l
  do l = 1, 3
    do k = 1, 3
      do jj = 1, 3
        do i = 1, 3
          c(i + 3*(jj - 1), k + 3*(l - 1)) = a(i,l)*b(jj,k)
        end do
      end do
    end do
  end do
end function under9

pure function detf(f, caller) result(j)
  ! det F, stopping on an inverted element
  real(dp), intent(in) :: f(3,3)
  character(*), intent(in) :: caller
  real(dp) :: j
  j = det3(f)
  if (j <= 0) error stop caller // ': det F <= 0'
end function detf

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

!=======================================================================
! Neo-Hooke (coupled): par = [lambda, mu]
!=======================================================================

pure function nh_energy(f, par) result(w)
  !! strain energy per reference volume
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp) :: w, lnj
  lnj = log(detf(f, 'nh_energy'))
  w = par(2)/2*(sum(f*f) - 3) - par(2)*lnj + par(1)/2*lnj**2
end function nh_energy

pure subroutine nh_ref(f, par, s, c)
  !! S and CC
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: ci(3,3), lnj
  lnj = log(detf(f, 'nh_ref'))
  ci = inv3(matmul(transpose(f), f))
  s = voigt(par(2)*(eye3 - ci) + par(1)*lnj*ci)
  c = par(1)*outer6(ci, ci) + 2*(par(2) - par(1)*lnj)*isym(ci)
end subroutine nh_ref

pure subroutine nh_cur(f, par, s, c)
  !! sigma and cc
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: j, lnj
  j = detf(f, 'nh_cur')
  lnj = log(j)
  s = voigt(par(2)*(matmul(f, transpose(f)) - eye3) + par(1)*lnj*eye3)/j
  c = iso4(eye3, par(1), par(2) - par(1)*lnj)/j
end subroutine nh_cur

pure subroutine nh_mix(f, par, p, a)
  !! P and AA
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: p(3,3), a(9,9)
  real(dp) :: fi(3,3), lnj
  lnj = log(detf(f, 'nh_mix'))
  fi = inv3(f)
  p = par(2)*(f - transpose(fi)) + par(1)*lnj*transpose(fi)
  a = par(2)*over9(eye3, eye3) + (par(2) - par(1)*lnj)*under9(transpose(fi), fi) &
    & + par(1)*outer9(transpose(fi), transpose(fi))
end subroutine nh_mix

!=======================================================================
! Modified Neo-Hooke: par = [kappa, mu], U = kappa/4 (J^2 - 2 ln J - 1)
!=======================================================================

pure function mnh_energy(f, par) result(w)
  !! strain energy per reference volume
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp) :: w, j
  j = detf(f, 'mnh_energy')
  w = par(2)/2*(j**(-2._dp/3)*sum(f*f) - 3) + par(1)/4*(j**2 - 2*log(j) - 1)
end function mnh_energy

pure subroutine mnh_ref(f, par, s, c)
  !! S and CC
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: ci(3,3), siso(3,3), j, i1, p, pt, g
  j = detf(f, 'mnh_ref')
  ci = inv3(matmul(transpose(f), f))
  i1 = sum(f*f)
  p = par(1)/2*(j - 1/j)
  pt = par(1)*j
  g = par(2)*j**(-2._dp/3)
  siso = g*(eye3 - i1/3*ci)
  s = voigt(j*p*ci + siso)
  c = j*pt*outer6(ci, ci) - 2*j*p*isym(ci) + 2*g*i1/3*(isym(ci) - outer6(ci, ci)/3) &
    & - 2*(outer6(ci, siso) + outer6(siso, ci))/3
end subroutine mnh_ref

pure subroutine mnh_cur(f, par, s, c)
  !! sigma and cc
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: bb(3,3), tiso(3,3), j, p, pt, trbb
  j = detf(f, 'mnh_cur')
  p = par(1)/2*(j - 1/j)
  pt = par(1)*j
  bb = j**(-2._dp/3)*matmul(f, transpose(f))
  trbb = bb(1,1) + bb(2,2) + bb(3,3)
  tiso = par(2)*(bb - trbb/3*eye3)
  s = voigt(p*eye3 + tiso/j)
  c = pt*outer6(eye3, eye3) - 2*p*isym(eye3) &
    & + (2*par(2)*trbb/3*(isym(eye3) - outer6(eye3, eye3)/3) &
    &    - 2*(outer6(eye3, tiso) + outer6(tiso, eye3))/3)/j
end subroutine mnh_cur

pure subroutine mnh_mix(f, par, p, a)
  !! P and AA
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: p(3,3), a(9,9)
  real(dp) :: fi(3,3), fit(3,3), j, i1, pr, pt, g
  j = detf(f, 'mnh_mix')
  fi = inv3(f)
  fit = transpose(fi)
  i1 = sum(f*f)
  pr = par(1)/2*(j - 1/j)
  pt = par(1)*j
  g = par(2)*j**(-2._dp/3)
  p = j*pr*fit + g*(f - i1/3*fit)
  a = j*pt*outer9(fit, fit) - j*pr*under9(fit, fi) &
    & + g*(over9(eye3, eye3) + i1/3*under9(fit, fi) - 2*(outer9(fit, f) + outer9(f, fit))/3 &
    &      + 2*i1/9*outer9(fit, fit))
end subroutine mnh_mix

!=======================================================================
! Mooney-Rivlin (decoupled): par = [c1, c2, kappa], U as modified Neo-Hooke
!=======================================================================

pure function mr_energy(f, par) result(w)
  !! strain energy per reference volume
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp) :: w, j, c(3,3), i1
  j = detf(f, 'mr_energy')
  c = matmul(transpose(f), f)
  i1 = c(1,1) + c(2,2) + c(3,3)
  w = par(1)*(j**(-2._dp/3)*i1 - 3) + par(2)*(j**(-4._dp/3)*(i1**2 - sum(c*c))/2 - 3) &
    & + par(3)/4*(j**2 - 2*log(j) - 1)
end function mr_energy

pure subroutine mr_ref(f, par, s, c)
  !! S and CC; the projection P : CCbar : P^T is expanded in closed form
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: cg(3,3), ci(3,3), cb(3,3), sbar(3,3), siso(3,3), d1(3,3)
  real(dp) :: j, j23, i1, p, pt, scc
  j = detf(f, 'mr_ref')
  j23 = j**(-2._dp/3)
  cg = matmul(transpose(f), f)
  ci = inv3(cg)
  i1 = cg(1,1) + cg(2,2) + cg(3,3)
  p = par(3)/2*(j - 1/j)
  pt = par(3)*j
  cb = j23*cg
  sbar = 2*par(1)*eye3 + 2*par(2)*(j23*i1*eye3 - cb)
  scc = sum(sbar*cg)
  siso = j23*(sbar - scc/3*ci)
  s = voigt(j*p*ci + siso)
  d1 = eye3 - i1/3*ci
  c = j*pt*outer6(ci, ci) - 2*j*p*isym(ci) &
    & + 4*par(2)*j23**2*(outer6(d1, d1) - isym(eye3) + (outer6(ci, cg) + outer6(cg, ci))/3 &
    &                    - sum(cg*cg)/9*outer6(ci, ci)) &
    & + 2*j23*scc/3*(isym(ci) - outer6(ci, ci)/3) - 2*(outer6(ci, siso) + outer6(siso, ci))/3
end subroutine mr_ref

pure subroutine mr_cur(f, par, s, c)
  !! sigma and cc; the projection P : ccbar : P is expanded in closed form
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: bb(3,3), bb2(3,3), tbar(3,3), tiso(3,3), dbb(3,3), j, p, pt, i1b, trt
  j = detf(f, 'mr_cur')
  p = par(3)/2*(j - 1/j)
  pt = par(3)*j
  bb = j**(-2._dp/3)*matmul(f, transpose(f))
  bb2 = matmul(bb, bb)
  i1b = bb(1,1) + bb(2,2) + bb(3,3)
  tbar = 2*par(1)*bb + 2*par(2)*(i1b*bb - bb2)
  trt = tbar(1,1) + tbar(2,2) + tbar(3,3)
  tiso = tbar - trt/3*eye3
  dbb = bb - i1b/3*eye3
  s = voigt(p*eye3 + tiso/j)
  c = pt*outer6(eye3, eye3) - 2*p*isym(eye3) &
    & + (4*par(2)*(outer6(dbb, dbb) - isym(bb) + (outer6(eye3, bb2) + outer6(bb2, eye3))/3 &
    &              - (bb2(1,1) + bb2(2,2) + bb2(3,3))/9*outer6(eye3, eye3)) &
    &    + 2*trt/3*(isym(eye3) - outer6(eye3, eye3)/3) &
    &    - 2*(outer6(eye3, tiso) + outer6(tiso, eye3))/3)/j
end subroutine mr_cur

pure subroutine mr_mix(f, par, p, a)
  !! P and AA
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: p(3,3), a(9,9)
  real(dp) :: fi(3,3), fit(3,3), b(3,3), cg(3,3), gi2(3,3), j, j23, j43, i1, i2, pr, pt
  j = detf(f, 'mr_mix')
  j23 = j**(-2._dp/3)
  j43 = j23**2
  fi = inv3(f)
  fit = transpose(fi)
  b = matmul(f, transpose(f))
  cg = matmul(transpose(f), f)
  i1 = b(1,1) + b(2,2) + b(3,3)
  i2 = (i1**2 - sum(b*b))/2
  gi2 = 2*(i1*f - matmul(b, f))
  pr = par(3)/2*(j - 1/j)
  pt = par(3)*j
  p = j*pr*fit + par(1)*j23*(2*f - 2*i1/3*fit) + par(2)*j43*(gi2 - 4*i2/3*fit)
  a = j*pt*outer9(fit, fit) - j*pr*under9(fit, fi) &
    & + par(1)*j23*(2*over9(eye3, eye3) + 2*i1/3*under9(fit, fi) &
    &               - 4*(outer9(fit, f) + outer9(f, fit))/3 + 4*i1/9*outer9(fit, fit)) &
    & + par(2)*j43*(2*i1*over9(eye3, eye3) + 4*outer9(f, f) &
    &               - 2*(over9(eye3, cg) + under9(f, transpose(f)) + over9(b, eye3)) &
    &               - 4*(outer9(fit, gi2) + outer9(gi2, fit))/3 &
    &               + 4*i2/3*under9(fit, fi) + 16*i2/9*outer9(fit, fit))
end subroutine mr_mix

end module materials
