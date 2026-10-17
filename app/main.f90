! SPDX-License-Identifier: MIT
program bench
  !! Times element R and K assembly for every material and configuration, and
  !! the material call alone on the same deformation gradients.
  !! Usage: fpm run --profile release [-- nelem [ntrial]]
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature, CUB_HEX
  use shapefuncs, only: shapefunc
  use materials, only: material, material_set, cfgname
  use materials_ad, only: material_set_ad
  use harness, only: test_element, element_gradients, assemble, stopwatch, start_watch, cpu_seconds, int_arg
  implicit none
  integer :: nelem = 2000, ntrial = 7
  type(cubature) :: q
  type(shapefunc) :: s
  type(material), allocatable :: mats(:)
  real(dp), allocatable :: xref(:,:,:), u(:,:,:), fg(:,:,:,:)
  real(dp) :: telem, tmat, checksum = 0
  type(stopwatch) :: sw
  integer :: e, m, c

  call int_arg(1, nelem)
  call int_arg(2, ntrial)
  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  ! hand-derived models, then the same models exported from JAX; sections, not an
  ! array constructor: gfortran 14 leaks the allocatable components of function results in [ ... ]
  allocate(mats(8))
  mats(1:4) = material_set(1.2_dp, 0.8_dp)
  mats(5:8) = material_set_ad(1.2_dp, 0.8_dp)

  ! Distinct distorted elements under moderate deformation, so no two calls
  ! see the same data; F at every point feeds the material-only timing.
  allocate(xref(3, s%nnodes, nelem), u(3, s%nnodes, nelem), fg(3, 3, q%npoints, nelem))
  do e = 1, nelem
    call test_element(s, e, 0.05_dp, xref(:,:,e), u(:,:,e))
    call element_gradients(q, s, xref(:,:,e), u(:,:,e), fg(:,:,:,e))
  end do

  write(*, '(a, i0, a, i0, a, i0, a)') 'element: ', s%nnodes, ' nodes, ', nelem, ' elements, best of ', &
    & ntrial, ' trials, CPU time'
  write(*, '(a8, a5, 3a12)') 'model', 'cfg', 'element ns', 'material ns', 'material %'
  do m = 1, size(mats)
    do c = 1, 3
      sw = start_watch(ntrial, cpu_seconds)
      do while (sw%next())
        call element_sweep()
      end do
      telem = sw%best
      sw = start_watch(ntrial, cpu_seconds)
      do while (sw%next())
        call material_sweep()
      end do
      tmat = sw%best
      write(*, '(a8, a5, 2f12.0, f12.1)') trim(mats(m)%name), cfgname(c), &
        & 1e9_dp*telem/nelem, 1e9_dp*tmat/nelem, 100*tmat/telem
    end do
  end do
  write(*, '(a, es12.4)') 'checksum ', checksum
  ! gfortran does not free main-program allocatables at exit; LeakSanitizer would report them
  deallocate(mats, xref, u, fg)

contains

subroutine element_sweep()
  ! R and K of every element; K feeds the checksum so no call is optimized away
  real(dp) :: r(3*s%nnodes), k(3*s%nnodes, 3*s%nnodes)
  integer :: e
  do e = 1, nelem
    call assemble(c, q, s, xref(:,:,e), u(:,:,e), mats(m), r, k)
    checksum = checksum + k(1,1)
  end do
end subroutine element_sweep

subroutine material_sweep()
  ! one batched material call per element, as the element routines make
  real(dp) :: st(9, q%npoints), tg(81, q%npoints)
  integer :: e
  do e = 1, nelem
    select case (c)
    case (1)
      call mats(m)%ref(q%npoints, fg(:,:,:,e), st, tg)
    case (2)
      call mats(m)%cur(q%npoints, fg(:,:,:,e), st, tg)
    case default
      call mats(m)%mix(q%npoints, fg(:,:,:,e), st, tg)
    end select
    checksum = checksum + tg(1,1)
  end do
end subroutine material_sweep

end program bench
