! SPDX-License-Identifier: MIT
program test_fe_assembly
  !! Checks for the element-level building blocks.
  !! Avoids NaN/Inf arithmetic so it runs under floating-point trapping.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures
  use shapefuncs
  use tensors, only: eye3, det3, inv3
  use materials, only: material, material_set
  use elements, only: elem_ref, elem_cur, elem_mix, shape_gradients
  implicit none
  integer :: ncheck = 0, nfail = 0
  real(dp), parameter :: eps = epsilon(1._dp), tol = 64*eps, fdtol = 64*sqrt(eps)
  ! Central differences: h ~ eps**(1/3) balances truncation and rounding
  real(dp), parameter :: h = 1e-5_dp
  character(*), parameter :: cfgname(3) = ['ref', 'cur', 'mix']
  type(cubature) :: q
  type(shapefunc) :: s
  type(material), allocatable :: mats(:)
  type(material) :: mat
  real(dp) :: xref(3,8), u(3,8), urot(3,8), rot(3,3), w(3,3), jac(3,3), vol(2)
  real(dp) :: r(24,3), k(24,24,3), rfd(24), kfd(24,24), up(3,8), rp(24), rm(24), kk(24,24), k0(24,24)
  integer :: c, i, j, a, m, g
  character(:), allocatable :: id

  ! Dependencies: trilinear hexahedron ------------------------------------
  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  call check(s%nnodes == 8 .and. s%npoints == 8, 'hex8: 8 nodes, 2x2x2 points')
  call check(abs(sum(q%weights) - 8) < tol, 'hex8: weights sum to reference volume 8')
  call check(all(abs(sum(s%func, dim=1) - 1) < tol), 'hex8: partition of unity')
  call check(all(abs(sum(s%derv, dim=2)) < tol), 'hex8: derivatives sum to zero')

  ! Tensors ---------------------------------------------------------------
  w = reshape([2._dp, 1._dp, 0._dp, 0.5_dp, 3._dp, 1._dp, 0._dp, 0.25_dp, 1.5_dp], [3, 3])
  call check(maxval(abs(matmul(w, inv3(w)) - eye3)) < tol, 'inv3: A A^-1 = I')
  call check(abs(det3(w) - 7.75_dp) < tol*8, 'det3 of a fixed matrix')

  ! Distorted unit cube and a finite displacement
  do a = 1, 8
    do i = 1, 3
      xref(i,a) = (s%coords(i,a) + 1)/2 + 0.05_dp*cos(real(3*a + i, dp))
      u(i,a) = 0.1_dp*sin(real(5*a + 2*i, dp))
    end do
  end do

  ! Kinematics: x = W X maps every volume element by det W ------------------
  vol = 0
  do g = 1, q%npoints
    jac = matmul(xref, transpose(s%derv(:,:,g)))
    vol = vol + [det3(jac), det3(matmul(w, jac))]*q%weights(g)
  end do
  call check(abs(vol(2) - det3(w)*vol(1)) < tol*vol(2), 'hex8: current volume = det F * reference volume')

  ! Rigid rotation about (1, 1, 1)/sqrt(3) by 0.7 rad
  w = reshape([0._dp, 1._dp, -1._dp, -1._dp, 0._dp, 1._dp, 1._dp, -1._dp, 0._dp], [3, 3])/sqrt(3._dp)
  rot = eye3 + sin(0.7_dp)*w + (1 - cos(0.7_dp))*matmul(w, w)
  urot = matmul(rot - eye3, xref)

  ! Every material in every configuration ---------------------------------
  mats = material_set(1.2_dp, 0.8_dp)
  do m = 1, size(mats)
    mat = mats(m)
    do c = 1, 3
      call assemble(c, u, r(:,c), k(:,:,c))
    end do
    do j = 1, 24
      up = u
      call bump(up, j, h)
      rfd(j) = energy(up)
      call bump(up, j, -2*h)
      rfd(j) = (rfd(j) - energy(up))/(2*h)
    end do
    do c = 1, 3
      id = trim(mat%name) // ' ' // cfgname(c) // ': '
      call check(close(r(:,c), rfd, fdtol), id // 'R = dPi/du (finite differences)')
      do j = 1, 24
        up = u
        call bump(up, j, h)
        call assemble(c, up, rp, kk)
        call bump(up, j, -2*h)
        call assemble(c, up, rm, kk)
        kfd(:,j) = (rp - rm)/(2*h)
      end do
      call check(close([k(:,:,c)], [kfd], fdtol), id // 'K = dR/du (finite differences)')
      call check(close([k(:,:,c)], [transpose(k(:,:,c))], tol), id // 'K symmetric')
      call check(close(r(:,c), r(:,1), tol), id // 'R equals reference configuration')
      call check(close([k(:,:,c)], [k(:,:,1)], tol), id // 'K equals reference configuration')

      call assemble(c, 0*u, rp, kk)
      call check(maxval(abs(rp)) < tol*maxval(abs(kk)), id // 'R = 0 at u = 0')
      if (m == 1 .and. c == 1) k0 = kk
      call check(close([kk], [k0], tol), id // 'K at u = 0 equals the small-strain K of ' // trim(mats(1)%name))
      call assemble(c, urot, rp, kk)
      call check(maxval(abs(rp)) < tol*maxval(abs(kk)), id // 'R = 0 under rigid rotation')
    end do
  end do

  ! gfortran does not free main-program allocatables at exit; LeakSanitizer would report them
  deallocate(mats, mat%par, id)

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

pure logical function close(x, y, rtol)
  ! max-norm comparison relative to the size of y; flatten matrices with [m]
  real(dp), intent(in) :: x(:), y(:), rtol
  close = maxval(abs(x - y)) <= rtol*maxval(abs(y))
end function close

subroutine bump(uu, j, dh)
  ! add dh to dof j = 3 (a - 1) + i
  real(dp), intent(inout) :: uu(3,8)
  integer, intent(in) :: j
  real(dp), intent(in) :: dh
  uu(mod(j - 1, 3) + 1, (j - 1)/3 + 1) = uu(mod(j - 1, 3) + 1, (j - 1)/3 + 1) + dh
end subroutine bump

subroutine assemble(cfg, uu, rr, kk)
  integer, intent(in) :: cfg
  real(dp), intent(in) :: uu(3,8)
  real(dp), intent(out) :: rr(24), kk(24,24)
  select case (cfg)
  case (1)
    call elem_ref(q, s, xref, uu, mat%ref, mat%par, rr, kk)
  case (2)
    call elem_cur(q, s, xref, uu, mat%cur, mat%par, rr, kk)
  case default
    call elem_mix(q, s, xref, uu, mat%mix, mat%par, rr, kk)
  end select
end subroutine assemble

function energy(uu) result(pi)
  ! element strain energy, integrated like the element routines
  real(dp), intent(in) :: uu(3,8)
  real(dp) :: pi, dndx(3,8), dv
  integer :: g
  pi = 0
  do g = 1, q%npoints
    call shape_gradients(q, s, g, xref, dndx, dv)
    pi = pi + mat%energy(eye3 + matmul(uu, transpose(dndx)), mat%par)*dv
  end do
end function energy

end program test_fe_assembly
