! SPDX-License-Identifier: MIT
module elements
  !! Element residual R and tangent K in the three configurations.
  !!
  !! Nodal arrays are (3, nnodes); R(3 nnodes) and K(3 nnodes, 3 nnodes) are
  !! ordered by node, then direction: dof 3 (a - 1) + i. The cubature and shape
  !! functions are built once by the caller and reused for every element.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature
  use shapefuncs, only: shapefunc
  use tensors, only: eye3, det3, inv3, unvoigt
  use materials, only: voigt_material, twopoint_material
  implicit none
  private

  public :: elem_ref, elem_cur, elem_mix, ref_gradients

contains

pure subroutine ref_gradients(q, s, g, xref, dndx, dv)
  !! dN/dX and the reference volume weight at cubature point g
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  integer, intent(in) :: g
  real(dp), intent(in) :: xref(:,:)
  real(dp), intent(out) :: dndx(:,:), dv
  real(dp) :: jac(3,3)
  jac = matmul(xref, transpose(s%derv(:,:,g)))
  dv = det3(jac)*q%weights(g)
  dndx = matmul(transpose(inv3(jac)), s%derv(:,:,g))
end subroutine ref_gradients

pure subroutine bmatrix(f, dndx, bm)
  ! Voigt strain-displacement matrix: delta E (strain-like) = bm delta u
  real(dp), intent(in) :: f(3,3), dndx(:,:)
  real(dp), intent(out) :: bm(:,:)
  integer :: a, i, col
  do a = 1, size(dndx, 2)
    do i = 1, 3
      col = 3*(a - 1) + i
      bm(1:3,col) = f(i,1:3)*dndx(1:3,a)
      bm(4,col) = f(i,1)*dndx(2,a) + f(i,2)*dndx(1,a)
      bm(5,col) = f(i,2)*dndx(3,a) + f(i,3)*dndx(2,a)
      bm(6,col) = f(i,1)*dndx(3,a) + f(i,3)*dndx(1,a)
    end do
  end do
end subroutine bmatrix

pure subroutine add_geometric(dndx, stress, dv, k)
  ! k(ai, bi) += dN_a . stress . dN_b dv, the initial-stress stiffness
  real(dp), intent(in) :: dndx(:,:), stress(3,3), dv
  real(dp), intent(inout) :: k(:,:)
  real(dp) :: gg(size(dndx, 2), size(dndx, 2))
  integer :: a, b, i
  gg = matmul(transpose(dndx), matmul(stress, dndx))*dv
  do b = 1, size(dndx, 2)
    do a = 1, size(dndx, 2)
      do i = 1, 3
        k(3*(a - 1) + i, 3*(b - 1) + i) = k(3*(a - 1) + i, 3*(b - 1) + i) + gg(a,b)
      end do
    end do
  end do
end subroutine add_geometric

pure subroutine elem_ref(q, s, xref, u, mat, par, r, k)
  !! total Lagrangian: R = int B^T S dV, K = int (B^T CC B + geometric(S)) dV
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:), par(:)
  procedure(voigt_material) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  real(dp) :: dndx(3, s%nnodes), bm(6, 3*s%nnodes), f(3,3), sv(6), cv(6,6), dv
  integer :: g
  r = 0
  k = 0
  do g = 1, q%npoints
    call ref_gradients(q, s, g, xref, dndx, dv)
    f = eye3 + matmul(u, transpose(dndx))
    call mat(f, par, sv, cv)
    call bmatrix(f, dndx, bm)
    r = r + matmul(transpose(bm), sv)*dv
    k = k + matmul(transpose(bm), matmul(cv, bm))*dv
    call add_geometric(dndx, unvoigt(sv), dv, k)
  end do
end subroutine elem_ref

pure subroutine elem_cur(q, s, xref, u, mat, par, r, k)
  !! updated Lagrangian: R = int B^T sigma dv, K = int (B^T cc B + geometric(sigma)) dv
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:), par(:)
  procedure(voigt_material) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  real(dp) :: dndx(3, s%nnodes), dndxc(3, s%nnodes), bm(6, 3*s%nnodes), f(3,3), sv(6), cv(6,6), dv
  integer :: g
  r = 0
  k = 0
  do g = 1, q%npoints
    call ref_gradients(q, s, g, xref, dndx, dv)
    f = eye3 + matmul(u, transpose(dndx))
    call mat(f, par, sv, cv)
    ! dN/dx = F^(-T) dN/dX and dv = J dV
    dndxc = matmul(transpose(inv3(f)), dndx)
    dv = det3(f)*dv
    call bmatrix(eye3, dndxc, bm)
    r = r + matmul(transpose(bm), sv)*dv
    k = k + matmul(transpose(bm), matmul(cv, bm))*dv
    call add_geometric(dndxc, unvoigt(sv), dv, k)
  end do
end subroutine elem_cur

pure subroutine elem_mix(q, s, xref, u, mat, par, r, k)
  !! two-point: R = int G^T P dV, K = int G^T AA G dV (geometric term is inside AA)
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:), par(:)
  procedure(twopoint_material) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  real(dp) :: dndx(3, s%nnodes), gm(9, 3*s%nnodes), f(3,3), p(3,3), am(9,9), dv
  integer :: g, a, i, jj
  r = 0
  k = 0
  gm = 0
  do g = 1, q%npoints
    call ref_gradients(q, s, g, xref, dndx, dv)
    f = eye3 + matmul(u, transpose(dndx))
    call mat(f, par, p, am)
    ! delta F_iJ = delta u_ia dN_a/dX_J
    do a = 1, s%nnodes
      do jj = 1, 3
        do i = 1, 3
          gm(i + 3*(jj - 1), 3*(a - 1) + i) = dndx(jj,a)
        end do
      end do
    end do
    r = r + matmul(transpose(gm), reshape(p, [9]))*dv
    k = k + matmul(transpose(gm), matmul(am, gm))*dv
  end do
end subroutine elem_mix

end module elements
