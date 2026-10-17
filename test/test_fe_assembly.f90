! SPDX-License-Identifier: MIT
program test_fe_assembly
  !! Checks for the element-level building blocks.
  !! Avoids NaN/Inf arithmetic so it runs under floating-point trapping.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures
  use shapefuncs
  use tensors, only: eye3, det3, inv3
  use materials, only: material, material_set, cfgname
  use materials_ad, only: material_set_ad
  use elements, only: shape_gradients
  use harness, only: test_element, assemble, check, close, check_summary, compare_paradigm
  implicit none
  real(dp), parameter :: eps = epsilon(1._dp), tol = 64*eps, fdtol = 64*sqrt(eps)
  ! JAX-exported and hand-derived tangents round differently; both are exact derivatives
  real(dp), parameter :: adtol = 1024*eps
  ! Central differences: h ~ eps**(1/3) balances truncation and rounding
  real(dp), parameter :: h = 1e-5_dp
  type(cubature) :: q
  type(shapefunc) :: s
  type(material), allocatable :: mats(:)
  type(material), allocatable :: nat(:), ad(:)
  type(material) :: mat
  real(dp) :: xref(3,8), u(3,8), urot(3,8), rot(3,3), w(3,3), jac(3,3), vol(2)
  real(dp) :: r(24,3), k(24,24,3), rfd(24), kfd(24,24), up(3,8), rp(24), rm(24), kk(24,24), k0(24,24)
  integer :: c, j, m, g
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
  call test_element(s, 0, 0.1_dp, xref, u)

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
  ! the hand-derived models, then the same four exported from JAX (materials_ad);
  ! sections, not an array constructor: gfortran 14 leaks the allocatable components
  ! of function results inside [ ... ]
  allocate(mats(8))
  mats(1:4) = material_set(1.2_dp, 0.8_dp)
  mats(5:8) = material_set_ad(1.2_dp, 0.8_dp)
  ! separate arrays, not sections: nvfortran 26.3 loses the procedure pointers of an
  ! array section passed to a class(material) dummy (segfault in material_ref)
  nat = mats(1:4)
  ad = mats(5:8)
  call compare_paradigm(nat, ad, q, s, adtol)
  do m = 1, size(mats)
    mat = mats(m)
    do c = 1, 3
      call assemble(c, q, s, xref, u, mat, r(:,c), k(:,:,c))
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
        call assemble(c, q, s, xref, up, mat, rp, kk)
        call bump(up, j, -2*h)
        call assemble(c, q, s, xref, up, mat, rm, kk)
        kfd(:,j) = (rp - rm)/(2*h)
      end do
      call check(close([k(:,:,c)], [kfd], fdtol), id // 'K = dR/du (finite differences)')
      call check(close([k(:,:,c)], [transpose(k(:,:,c))], tol), id // 'K symmetric')
      call check(close(r(:,c), r(:,1), tol), id // 'R equals reference configuration')
      call check(close([k(:,:,c)], [k(:,:,1)], tol), id // 'K equals reference configuration')

      call assemble(c, q, s, xref, 0*u, mat, rp, kk)
      call check(maxval(abs(rp)) < tol*maxval(abs(kk)), id // 'R = 0 at u = 0')
      if (m == 1 .and. c == 1) k0 = kk
      call check(close([kk], [k0], tol), id // 'K at u = 0 equals the small-strain K of ' // trim(mats(1)%name))
      call assemble(c, q, s, xref, urot, mat, rp, kk)
      call check(maxval(abs(rp)) < tol*maxval(abs(kk)), id // 'R = 0 under rigid rotation')
    end do
  end do

  ! gfortran does not free main-program allocatables at exit; LeakSanitizer would report them
  deallocate(mats, nat, ad, mat%par, id)
  call check_summary()

contains

subroutine bump(uu, j, dh)
  ! add dh to dof j = 3 (a - 1) + i
  real(dp), intent(inout) :: uu(3,8)
  integer, intent(in) :: j
  real(dp), intent(in) :: dh
  uu(mod(j - 1, 3) + 1, (j - 1)/3 + 1) = uu(mod(j - 1, 3) + 1, (j - 1)/3 + 1) + dh
end subroutine bump

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
