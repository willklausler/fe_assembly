! SPDX-License-Identifier: MIT
module elements
  !! Element residual R and tangent K in the three configurations.
  !!
  !! Nodal arrays are (3, nnodes); R(3 nnodes) and K(3 nnodes, 3 nnodes) are
  !! ordered by node, then direction: dof 3 (a - 1) + i. The cubature and shape
  !! functions are built once by the caller and reused for every element.
  !! K is assembled in 3x3 node blocks: the product tangent . grad N_b is formed
  !! once per node b and reused for every a <= b; the lower blocks are mirrored,
  !! which assumes a symmetric K (hyperelastic materials).
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature
  use shapefuncs, only: shapefunc
  use tensors, only: eye3, det3, inv3
  use materials, only: voigt_material, twopoint_material
  implicit none
  private

  public :: elem_ref, elem_cur, elem_mix, shape_gradients

contains

pure subroutine shape_gradients(q, s, g, x, dndx, dv)
  !! dN/dx and the volume weight at cubature point g, for nodal coordinates x
  !! (reference coordinates give dN/dX and dV)
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  integer, intent(in) :: g
  real(dp), intent(in) :: x(:,:)
  real(dp), intent(out) :: dndx(:,:), dv
  real(dp) :: jac(3,3), ji(3,3)
  integer :: a, jj
  jac = 0
  do a = 1, s%nnodes
    do jj = 1, 3
      jac(:,jj) = jac(:,jj) + x(:,a)*s%derv(jj,a,g)
    end do
  end do
  ji = inv3(jac)
  dv = det3(jac)*q%weights(g)
  ! dN/dx_j = sum_k (J^-1)_kj dN/dxi_k
  do a = 1, s%nnodes
    do jj = 1, 3
      dndx(jj,a) = sum(ji(:,jj)*s%derv(:,a,g))
    end do
  end do
end subroutine shape_gradients

pure function grad_u(u, dndx) result(f)
  ! F = I + u dN^T
  real(dp), intent(in) :: u(:,:), dndx(:,:)
  real(dp) :: f(3,3)
  integer :: a, jj
  f = eye3
  do a = 1, size(dndx, 2)
    do jj = 1, 3
      f(:,jj) = f(:,jj) + u(:,a)*dndx(jj,a)
    end do
  end do
end function grad_u

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

pure subroutine add_voigt(bm, dndx, sv, cv, dv, r, k)
  ! r += B^T s dv; upper blocks of k += (B_a^T C B_b + I dN_a . S . dN_b) dv
  ! Explicit fixed-length loops, written like elem_mix, so configurations differ by algorithm only.
  real(dp), intent(in), contiguous :: bm(:,:), dndx(:,:)
  real(dp), intent(in) :: sv(6), cv(6,6), dv
  real(dp), intent(inout), contiguous :: r(:), k(:,:)
  real(dp) :: cb(6,3), sd(3), sm(3,3), t
  integer :: a, b, i, kk, v, z, ia, ib
  sm = reshape([sv(1), sv(4), sv(6), sv(4), sv(2), sv(5), sv(6), sv(5), sv(3)], [3, 3])
  do i = 1, size(r)
    t = 0
    do v = 1, 6
      t = t + bm(v,i)*sv(v)
    end do
    r(i) = r(i) + t*dv
  end do
  do b = 1, size(dndx, 2)
    ib = 3*(b - 1)
    ! (C B_b) dv, 108 multiply-adds, reused for every a <= b
    do kk = 1, 3
      cb(:,kk) = 0
      do z = 1, 6
        cb(:,kk) = cb(:,kk) + cv(:,z)*bm(z, ib + kk)
      end do
      cb(:,kk) = cb(:,kk)*dv
    end do
    do i = 1, 3
      sd(i) = (sm(i,1)*dndx(1,b) + sm(i,2)*dndx(2,b) + sm(i,3)*dndx(3,b))*dv
    end do
    do a = 1, b
      ia = 3*(a - 1)
      ! K_ab = B_a^T (C B_b), 54 multiply-adds, plus the geometric diagonal
      t = dndx(1,a)*sd(1) + dndx(2,a)*sd(2) + dndx(3,a)*sd(3)
      do kk = 1, 3
        do i = 1, 3
          k(ia + i, ib + kk) = k(ia + i, ib + kk) &
            & + bm(1, ia + i)*cb(1,kk) + bm(2, ia + i)*cb(2,kk) + bm(3, ia + i)*cb(3,kk) &
            & + bm(4, ia + i)*cb(4,kk) + bm(5, ia + i)*cb(5,kk) + bm(6, ia + i)*cb(6,kk)
        end do
        k(ia + kk, ib + kk) = k(ia + kk, ib + kk) + t
      end do
    end do
  end do
end subroutine add_voigt

pure subroutine mirror(k)
  ! copy the upper node blocks into the lower ones
  real(dp), intent(inout) :: k(:,:)
  integer :: a, b, ia, ib
  do b = 1, size(k, 2)/3
    ib = 3*(b - 1)
    do a = 1, b - 1
      ia = 3*(a - 1)
      k(ib+1:ib+3, ia+1:ia+3) = transpose(k(ia+1:ia+3, ib+1:ib+3))
    end do
  end do
end subroutine mirror

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
    call shape_gradients(q, s, g, xref, dndx, dv)
    f = grad_u(u, dndx)
    call mat(f, par, sv, cv)
    call bmatrix(f, dndx, bm)
    call add_voigt(bm, dndx, sv, cv, dv, r, k)
  end do
  call mirror(k)
end subroutine elem_ref

pure subroutine elem_cur(q, s, xref, u, mat, par, r, k)
  !! updated Lagrangian: R = int B^T sigma dv, K = int (B^T cc B + geometric(sigma)) dv
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:), par(:)
  procedure(voigt_material) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  real(dp) :: x(3, s%nnodes), dndx(3, s%nnodes), bm(6, 3*s%nnodes), f(3,3), sv(6), cv(6,6), dv
  integer :: g, a, jj
  r = 0
  k = 0
  x = xref + u
  do g = 1, q%npoints
    ! Discrete updated Lagrangian: dN/dx and dv from the current coordinates,
    ! F from F^-1 = I - du/dx; nothing is pushed forward from the reference configuration
    call shape_gradients(q, s, g, x, dndx, dv)
    f = eye3
    do a = 1, s%nnodes
      do jj = 1, 3
        f(:,jj) = f(:,jj) - u(:,a)*dndx(jj,a)
      end do
    end do
    f = inv3(f)
    call mat(f, par, sv, cv)
    ! ponytail: dense B with F = I spends half its work on zeros; a sparse spatial B if cur matters
    call bmatrix(eye3, dndx, bm)
    call add_voigt(bm, dndx, sv, cv, dv, r, k)
  end do
  call mirror(k)
end subroutine elem_cur

pure subroutine elem_mix(q, s, xref, u, mat, par, r, k)
  !! two-point: R = int P : grad N dV, K_aibk = int N_a,J A_iJkL N_b,L dV (geometric term inside AA)
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:), par(:)
  procedure(twopoint_material) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  real(dp) :: dndx(3, s%nnodes), f(3,3), p(3,3), am(9,9), w(9,3), dv
  integer :: g, a, b, i, kk, ia, ib
  r = 0
  k = 0
  do g = 1, q%npoints
    call shape_gradients(q, s, g, xref, dndx, dv)
    f = grad_u(u, dndx)
    call mat(f, par, p, am)
    do a = 1, s%nnodes
      ia = 3*(a - 1)
      r(ia+1:ia+3) = r(ia+1:ia+3) + matmul(p, dndx(:,a))*dv
    end do
    do b = 1, s%nnodes
      ib = 3*(b - 1)
      ! W(iJ, k) = A_iJkL N_b,L dv, 81 multiply-adds, reused for every a <= b
      do kk = 1, 3
        w(:,kk) = (am(:,kk)*dndx(1,b) + am(:,kk+3)*dndx(2,b) + am(:,kk+6)*dndx(3,b))*dv
      end do
      do a = 1, b
        ia = 3*(a - 1)
        ! K_ab(i, k) = N_a,J W(iJ, k), 27 multiply-adds
        do kk = 1, 3
          do i = 1, 3
            k(ia + i, ib + kk) = k(ia + i, ib + kk) &
              & + dndx(1,a)*w(i,kk) + dndx(2,a)*w(i+3,kk) + dndx(3,a)*w(i+6,kk)
          end do
        end do
      end do
    end do
  end do
  call mirror(k)
end subroutine elem_mix

end module elements
