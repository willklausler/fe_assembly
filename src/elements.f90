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
  !! Each element routine runs in two passes: kinematics at every point, one
  !! batched material call, then assembly.
  !! The kinematics pass is the only check: status = 1 if det F <= 0 at a point (an
  !! inverted element, e.g. a Newton step too large), 2 if the reference element is
  !! degenerate; then R = K = 0 and the material is not called. Without the optional
  !! status argument such an element stops with error stop.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature
  use shapefuncs, only: shapefunc
  use tensors, only: eye3, det3, inv3
  use materials, only: material
  implicit none
  private

  public :: elem_ref, elem_cur, elem_mix, shape_gradients
  public :: geometry, reference_geometry
  public :: ELEM_OK, ELEM_INVERTED, ELEM_DEGENERATE

  integer, parameter :: ELEM_OK = 0
    !! status: R and K computed
  integer, parameter :: ELEM_INVERTED = 1
    !! status: det F <= 0 at a cubature point; R = K = 0
  integer, parameter :: ELEM_DEGENERATE = 2
    !! status: the reference element has det(dX/dxi) <= 0 at a cubature point; R = K = 0

  type :: geometry
    !! reference geometry of one element, fixed across Newton iterations: dN/dX and dV
    !! at every cubature point (the total Lagrangian and two-point forms can reuse it;
    !! the updated Lagrangian form cannot, its gradients follow the deformation)
    real(dp), allocatable :: dndx(:,:,:)
      !! dN/dX(3, nnodes, npoints)
    real(dp), allocatable :: dv(:)
      !! volume weight dV(npoints)
  end type geometry

contains

subroutine reference_geometry(q, s, xref, geo)
  !! dN/dX and dV at every cubature point of the element with reference coordinates xref
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:)
  type(geometry), intent(out) :: geo
  integer :: g
  allocate(geo%dndx(3, s%nnodes, q%npoints), geo%dv(q%npoints))
  do g = 1, q%npoints
    call shape_gradients(q, s, g, xref, geo%dndx(:,:,g), geo%dv(g))
  end do
end subroutine reference_geometry

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
  call inv3(jac, ji)
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
  integer :: a, b, ia, ib, i, j
  do b = 1, size(k, 2)/3
    ib = 3*(b - 1)
    do a = 1, b - 1
      ia = 3*(a - 1)
      ! explicit loop: nvfortran 26.3 makes a heap temporary for transpose of a section
      do i = 1, 3
        do j = 1, 3
          k(ib + i, ia + j) = k(ia + j, ib + i)
        end do
      end do
    end do
  end do
end subroutine mirror

subroutine check_point(reference_ok, current_ok, caller, r, k, status, bad)
  ! the kinematics check: bad (and R = K = 0, status set or error stop) for a bad element
  logical, intent(in) :: reference_ok, current_ok
  character(*), intent(in) :: caller
  real(dp), intent(inout) :: r(:), k(:,:)
  integer, intent(out), optional :: status
  logical, intent(out) :: bad
  character(:), allocatable :: msg
  bad = .not. (reference_ok .and. current_ok)
  if (.not. bad) return
  r = 0
  k = 0
  if (present(status)) then
    status = merge(ELEM_INVERTED, ELEM_DEGENERATE, reference_ok)
    return
  end if
  if (reference_ok) then
    msg = caller // ': det F <= 0 at a cubature point (inverted element)'
  else
    msg = caller // ': degenerate reference element'
  end if
  error stop msg
end subroutine check_point

subroutine elem_ref(q, s, xref, u, mat, r, k, status, geo)
  !! total Lagrangian: R = int B^T S dV, K = int (B^T CC B + geometric(S)) dV
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:)
  class(material), intent(in) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  integer, intent(out), optional :: status
    !! ELEM_OK, ELEM_INVERTED or ELEM_DEGENERATE; without it a bad element stops
  type(geometry), intent(in), optional :: geo
    !! reference_geometry of this element: dN/dX and dV are not recomputed
  real(dp) :: dndx(3, s%nnodes, q%npoints), dv(q%npoints)
  integer :: g
  if (present(geo)) then
    call ref_core(q, s, geo%dndx, geo%dv, u, mat, r, k, status)
    return
  end if
  do g = 1, q%npoints
    call shape_gradients(q, s, g, xref, dndx(:,:,g), dv(g))
  end do
  call ref_core(q, s, dndx, dv, u, mat, r, k, status)
end subroutine elem_ref

subroutine ref_core(q, s, dndx, dv, u, mat, r, k, status)
  ! elem_ref after the reference geometry
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: dndx(3, s%nnodes, q%npoints), dv(q%npoints), u(:,:)
  class(material), intent(in) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  integer, intent(out), optional :: status
  real(dp) :: f(3,3,q%npoints), sv(6, q%npoints), cv(6,6,q%npoints), bm(6, 3*s%nnodes)
  logical :: bad
  integer :: g
  r = 0
  k = 0
  do g = 1, q%npoints
    f(:,:,g) = grad_u(u, dndx(:,:,g))
    call check_point(dv(g) > 0, det3(f(:,:,g)) > 0, 'elem_ref', r, k, status, bad)
    if (bad) return
  end do
  if (present(status)) status = ELEM_OK
  call mat%ref(q%npoints, f, sv, cv)
  do g = 1, q%npoints
    call bmatrix(f(:,:,g), dndx(:,:,g), bm)
    call add_voigt(bm, dndx(:,:,g), sv(:,g), cv(:,:,g), dv(g), r, k)
  end do
  call mirror(k)
end subroutine ref_core

subroutine elem_cur(q, s, xref, u, mat, r, k, status)
  !! updated Lagrangian: R = int B^T sigma dv, K = int (B^T cc B + geometric(sigma)) dv
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:)
  class(material), intent(in) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  integer, intent(out), optional :: status
    !! ELEM_OK, ELEM_INVERTED or ELEM_DEGENERATE; without it a bad element stops
  real(dp) :: x(3, s%nnodes), dndx(3, s%nnodes, q%npoints), dv(q%npoints), f(3,3,q%npoints)
  real(dp) :: sv(6, q%npoints), cv(6,6,q%npoints), bm(6, 3*s%nnodes), fi(3,3)
  logical :: bad
  integer :: g, a, jj
  r = 0
  k = 0
  x = xref + u
  do g = 1, q%npoints
    ! Discrete updated Lagrangian: dN/dx and dv from the current coordinates,
    ! F from F^-1 = I - du/dx; nothing is pushed forward from the reference configuration
    call shape_gradients(q, s, g, x, dndx(:,:,g), dv(g))
    fi = eye3
    do a = 1, s%nnodes
      do jj = 1, 3
        fi(:,jj) = fi(:,jj) - u(:,a)*dndx(jj,a,g)
      end do
    end do
    ! the current volume weight and det F^-1 > 0 both say the element is not inverted
    call check_point(.true., dv(g) > 0 .and. det3(fi) > 0, 'elem_cur', r, k, status, bad)
    if (bad) return
    call inv3(fi, f(:,:,g))
  end do
  if (present(status)) status = ELEM_OK
  call mat%cur(q%npoints, f, sv, cv)
  do g = 1, q%npoints
    ! ponytail: dense B with F = I spends half its work on zeros; a sparse spatial B if cur matters
    call bmatrix(eye3, dndx(:,:,g), bm)
    call add_voigt(bm, dndx(:,:,g), sv(:,g), cv(:,:,g), dv(g), r, k)
  end do
  call mirror(k)
end subroutine elem_cur

subroutine elem_mix(q, s, xref, u, mat, r, k, status, geo)
  !! two-point: R = int P : grad N dV, K_aibk = int N_a,J A_iJkL N_b,L dV (geometric term inside AA)
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:)
  class(material), intent(in) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  integer, intent(out), optional :: status
    !! ELEM_OK, ELEM_INVERTED or ELEM_DEGENERATE; without it a bad element stops
  type(geometry), intent(in), optional :: geo
    !! reference_geometry of this element: dN/dX and dV are not recomputed
  real(dp) :: dndx(3, s%nnodes, q%npoints), dv(q%npoints)
  integer :: g
  if (present(geo)) then
    call mix_core(q, s, geo%dndx, geo%dv, u, mat, r, k, status)
    return
  end if
  do g = 1, q%npoints
    call shape_gradients(q, s, g, xref, dndx(:,:,g), dv(g))
  end do
  call mix_core(q, s, dndx, dv, u, mat, r, k, status)
end subroutine elem_mix

subroutine mix_core(q, s, dndx, dv, u, mat, r, k, status)
  ! elem_mix after the reference geometry
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: dndx(3, s%nnodes, q%npoints), dv(q%npoints), u(:,:)
  class(material), intent(in) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  integer, intent(out), optional :: status
  real(dp) :: f(3,3,q%npoints), p(3,3,q%npoints), am(9,9,q%npoints), w(9,3)
  logical :: bad
  integer :: g, a, b, i, kk, ia, ib
  r = 0
  k = 0
  do g = 1, q%npoints
    f(:,:,g) = grad_u(u, dndx(:,:,g))
    call check_point(dv(g) > 0, det3(f(:,:,g)) > 0, 'elem_mix', r, k, status, bad)
    if (bad) return
  end do
  if (present(status)) status = ELEM_OK
  call mat%mix(q%npoints, f, p, am)
  do g = 1, q%npoints
    do a = 1, s%nnodes
      ia = 3*(a - 1)
      do i = 1, 3
        r(ia + i) = r(ia + i) + (p(i,1,g)*dndx(1,a,g) + p(i,2,g)*dndx(2,a,g) + p(i,3,g)*dndx(3,a,g))*dv(g)
      end do
    end do
    do b = 1, s%nnodes
      ib = 3*(b - 1)
      ! W(iJ, k) = A_iJkL N_b,L dv, 81 multiply-adds, reused for every a <= b
      do kk = 1, 3
        w(:,kk) = (am(:,kk,g)*dndx(1,b,g) + am(:,kk+3,g)*dndx(2,b,g) + am(:,kk+6,g)*dndx(3,b,g))*dv(g)
      end do
      do a = 1, b
        ia = 3*(a - 1)
        ! K_ab(i, k) = N_a,J W(iJ, k), 27 multiply-adds
        do kk = 1, 3
          do i = 1, 3
            k(ia + i, ib + kk) = k(ia + i, ib + kk) &
              & + dndx(1,a,g)*w(i,kk) + dndx(2,a,g)*w(i+3,kk) + dndx(3,a,g)*w(i+6,kk)
          end do
        end do
      end do
    end do
  end do
  call mirror(k)
end subroutine mix_core

end module elements
