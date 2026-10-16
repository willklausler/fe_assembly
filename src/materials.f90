! SPDX-License-Identifier: MIT
module materials
  !! Hyperelastic stresses and tangents, one routine per model and configuration.
  !!
  !! Every per-point routine takes the deformation gradient F and a parameter array
  !! and returns the stress and tangent in the configuration's natural storage:
  !! *_ref: S and CC in Voigt form (6, 6x6); *_cur: sigma and cc in Voigt form;
  !! *_mix: P (3x3) and AA (9x9, two-point index i + 3 (J - 1)).
  !! Elements call the batched methods mat%ref, mat%cur, mat%mix on all points of
  !! an element at once; other evaluation paradigms (jax/) extend type material
  !! and override only these three methods.
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
    !! one model: its name, parameters, per-point routines and batched methods
    character(16) :: name = ''
    real(dp), allocatable :: par(:)
    procedure(energy_function), pointer, nopass :: energy => null()
    procedure(voigt_material), pointer, nopass :: ref_pt => null(), cur_pt => null()
    procedure(twopoint_material), pointer, nopass :: mix_pt => null()
  contains
    procedure :: ref => material_ref
      !! S and CC at n points: f(3,3,n) -> s(6,n), c(6,6,n)
    procedure :: cur => material_cur
      !! sigma and cc at n points: f(3,3,n) -> s(6,n), c(6,6,n)
    procedure :: mix => material_mix
      !! P and AA at n points: f(3,3,n) -> p(3,3,n), a(9,9,n)
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

!-----------------------------------------------------------------------
! Batched evaluation: native loops over the per-point routines
!-----------------------------------------------------------------------

subroutine material_ref(self, n, f, s, c)
  class(material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: s(6,n), c(6,6,n)
  integer :: g
  do g = 1, n
    call self%ref_pt(f(:,:,g), self%par, s(:,g), c(:,:,g))
  end do
end subroutine material_ref

subroutine material_cur(self, n, f, s, c)
  class(material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: s(6,n), c(6,6,n)
  integer :: g
  do g = 1, n
    call self%cur_pt(f(:,:,g), self%par, s(:,g), c(:,:,g))
  end do
end subroutine material_cur

subroutine material_mix(self, n, f, p, a)
  class(material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: p(3,3,n), a(9,9,n)
  integer :: g
  do g = 1, n
    call self%mix_pt(f(:,:,g), self%par, p(:,:,g), a(:,:,g))
  end do
end subroutine material_mix

function material_set(lam, mu) result(m)
  !! every model, with parameters that give the same small-strain Lame constants lam, mu
  real(dp), intent(in) :: lam, mu
  type(material) :: m(4)
  real(dp) :: kappa
  kappa = lam + 2*mu/3
  m(1)%name = 'svk'
  m(1)%par = [lam, mu]
  m(1)%energy => svk_energy
  m(1)%ref_pt => svk_ref
  m(1)%cur_pt => svk_cur
  m(1)%mix_pt => svk_mix
  m(2)%name = 'nh'
  m(2)%par = [lam, mu]
  m(2)%energy => nh_energy
  m(2)%ref_pt => nh_ref
  m(2)%cur_pt => nh_cur
  m(2)%mix_pt => nh_mix
  m(3)%name = 'mnh'
  m(3)%par = [kappa, mu]
  m(3)%energy => mnh_energy
  m(3)%ref_pt => mnh_ref
  m(3)%cur_pt => mnh_cur
  m(3)%mix_pt => mnh_mix
  ! mu = 2 (c1 + c2); the split c2 = mu/8 is arbitrary but keeps both terms active
  m(4)%name = 'mr'
  m(4)%par = [3*mu/8, mu/8, kappa]
  m(4)%energy => mr_energy
  m(4)%ref_pt => mr_ref
  m(4)%cur_pt => mr_cur
  m(4)%mix_pt => mr_mix
end function material_set

!-----------------------------------------------------------------------
! Fourth-order products in Voigt (6x6) and two-point (9x9) storage
!-----------------------------------------------------------------------
! Each adds coef times a product into the tangent in place: array-valued helpers
! inside long expressions made gfortran malloc a 6x6 or 9x9 temporary per term.

pure subroutine add_outer6(c, coef, a, b)
  ! c += coef A (x) B, A and B symmetric
  real(dp), intent(inout) :: c(6,6)
  real(dp), intent(in) :: coef, a(3,3), b(3,3)
  integer :: k, l
  do l = 1, 6
    do k = 1, 6
      c(k,l) = c(k,l) + coef*a(vi(k), vj(k))*b(vi(l), vj(l))
    end do
  end do
end subroutine add_outer6

pure subroutine add_isym6(c, coef, a)
  ! c += coef II_A, II_A = (A (x)bar A + A (x)under A)/2, A symmetric
  real(dp), intent(inout) :: c(6,6)
  real(dp), intent(in) :: coef, a(3,3)
  integer :: k, l
  do l = 1, 6
    do k = 1, 6
      c(k,l) = c(k,l) + coef*(a(vi(k), vi(l))*a(vj(k), vj(l)) + a(vi(k), vj(l))*a(vj(k), vi(l)))/2
    end do
  end do
end subroutine add_isym6

pure subroutine add_outer9(m, coef, a, b)
  ! m += coef A (x) B: (A (x) B)_iJkL = A_iJ B_kL
  real(dp), intent(inout) :: m(9,9)
  real(dp), intent(in) :: coef, a(3,3), b(3,3)
  integer :: i, jj, k, l
  do l = 1, 3
    do k = 1, 3
      do jj = 1, 3
        do i = 1, 3
          m(i + 3*(jj - 1), k + 3*(l - 1)) = m(i + 3*(jj - 1), k + 3*(l - 1)) + coef*a(i,jj)*b(k,l)
        end do
      end do
    end do
  end do
end subroutine add_outer9

pure subroutine add_over9(m, coef, a, b)
  ! m += coef A (x)bar B: (A (x)bar B)_iJkL = A_ik B_JL
  real(dp), intent(inout) :: m(9,9)
  real(dp), intent(in) :: coef, a(3,3), b(3,3)
  integer :: i, jj, k, l
  do l = 1, 3
    do k = 1, 3
      do jj = 1, 3
        do i = 1, 3
          m(i + 3*(jj - 1), k + 3*(l - 1)) = m(i + 3*(jj - 1), k + 3*(l - 1)) + coef*a(i,k)*b(jj,l)
        end do
      end do
    end do
  end do
end subroutine add_over9

pure subroutine add_under9(m, coef, a, b)
  ! m += coef A (x)under B: (A (x)under B)_iJkL = A_iL B_Jk
  real(dp), intent(inout) :: m(9,9)
  real(dp), intent(in) :: coef, a(3,3), b(3,3)
  integer :: i, jj, k, l
  do l = 1, 3
    do k = 1, 3
      do jj = 1, 3
        do i = 1, 3
          m(i + 3*(jj - 1), k + 3*(l - 1)) = m(i + 3*(jj - 1), k + 3*(l - 1)) + coef*a(i,l)*b(jj,k)
        end do
      end do
    end do
  end do
end subroutine add_under9

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
  c = 0
  call add_outer6(c, par(1), ci, ci)
  call add_isym6(c, 2*(par(2) - par(1)*lnj), ci)
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
  real(dp) :: fi(3,3), fit(3,3), lnj
  lnj = log(detf(f, 'nh_mix'))
  fi = inv3(f)
  fit = transpose(fi)
  p = par(2)*(f - fit) + par(1)*lnj*fit
  a = 0
  call add_over9(a, par(2), eye3, eye3)
  call add_under9(a, par(2) - par(1)*lnj, fit, fi)
  call add_outer9(a, par(1), fit, fit)
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
  c = 0
  call add_outer6(c, j*pt - 2*g*i1/9, ci, ci)
  call add_isym6(c, -2*j*p + 2*g*i1/3, ci)
  call add_outer6(c, -2._dp/3, ci, siso)
  call add_outer6(c, -2._dp/3, siso, ci)
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
  c = 0
  call add_outer6(c, pt - 2*par(2)*trbb/(9*j), eye3, eye3)
  call add_isym6(c, -2*p + 2*par(2)*trbb/(3*j), eye3)
  call add_outer6(c, -2/(3*j), eye3, tiso)
  call add_outer6(c, -2/(3*j), tiso, eye3)
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
  a = 0
  call add_outer9(a, j*pt + 2*g*i1/9, fit, fit)
  call add_under9(a, -j*pr + g*i1/3, fit, fi)
  call add_over9(a, g, eye3, eye3)
  call add_outer9(a, -2*g/3, fit, f)
  call add_outer9(a, -2*g/3, f, fit)
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
  real(dp) :: j, j23, i1, p, pt, scc, k4, k2
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
  ! k4 multiplies P : CCbar : P^T, k2 the trace term
  k4 = 4*par(2)*j23**2
  k2 = 2*j23*scc/3
  c = 0
  call add_outer6(c, j*pt - k4*sum(cg*cg)/9 - k2/3, ci, ci)
  call add_isym6(c, -2*j*p + k2, ci)
  call add_outer6(c, k4, d1, d1)
  call add_isym6(c, -k4, eye3)
  call add_outer6(c, k4/3, ci, cg)
  call add_outer6(c, k4/3, cg, ci)
  call add_outer6(c, -2._dp/3, ci, siso)
  call add_outer6(c, -2._dp/3, siso, ci)
end subroutine mr_ref

pure subroutine mr_cur(f, par, s, c)
  !! sigma and cc; the projection P : ccbar : P is expanded in closed form
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: s(6), c(6,6)
  real(dp) :: bb(3,3), bb2(3,3), tbar(3,3), tiso(3,3), dbb(3,3), j, p, pt, i1b, trt, k4, k2
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
  ! k4 multiplies P : ccbar : P, k2 the trace term
  k4 = 4*par(2)/j
  k2 = 2*trt/(3*j)
  c = 0
  call add_outer6(c, pt - k4*(bb2(1,1) + bb2(2,2) + bb2(3,3))/9 - k2/3, eye3, eye3)
  call add_isym6(c, -2*p + k2, eye3)
  call add_outer6(c, k4, dbb, dbb)
  call add_isym6(c, -k4, bb)
  call add_outer6(c, k4/3, eye3, bb2)
  call add_outer6(c, k4/3, bb2, eye3)
  call add_outer6(c, -2/(3*j), eye3, tiso)
  call add_outer6(c, -2/(3*j), tiso, eye3)
end subroutine mr_cur

pure subroutine mr_mix(f, par, p, a)
  !! P and AA
  real(dp), intent(in) :: f(3,3), par(:)
  real(dp), intent(out) :: p(3,3), a(9,9)
  real(dp) :: fi(3,3), fit(3,3), ft(3,3), b(3,3), cg(3,3), gi2(3,3), j, j23, j43, i1, i2, pr, pt, a1, a2
  j = detf(f, 'mr_mix')
  j23 = j**(-2._dp/3)
  j43 = j23**2
  fi = inv3(f)
  fit = transpose(fi)
  ft = transpose(f)
  b = matmul(f, ft)
  cg = matmul(ft, f)
  i1 = b(1,1) + b(2,2) + b(3,3)
  i2 = (i1**2 - sum(b*b))/2
  gi2 = 2*(i1*f - matmul(b, f))
  pr = par(3)/2*(j - 1/j)
  pt = par(3)*j
  p = j*pr*fit + par(1)*j23*(2*f - 2*i1/3*fit) + par(2)*j43*(gi2 - 4*i2/3*fit)
  a1 = par(1)*j23
  a2 = par(2)*j43
  a = 0
  call add_outer9(a, j*pt + 4*a1*i1/9 + 16*a2*i2/9, fit, fit)
  call add_under9(a, -j*pr + 2*a1*i1/3 + 4*a2*i2/3, fit, fi)
  call add_over9(a, 2*a1 + 2*a2*i1, eye3, eye3)
  call add_outer9(a, -4*a1/3, fit, f)
  call add_outer9(a, -4*a1/3, f, fit)
  call add_outer9(a, 4*a2, f, f)
  call add_over9(a, -2*a2, eye3, cg)
  call add_under9(a, -2*a2, f, ft)
  call add_over9(a, -2*a2, b, eye3)
  call add_outer9(a, -4*a2/3, fit, gi2)
  call add_outer9(a, -4*a2/3, gi2, fit)
end subroutine mr_mix

end module materials
