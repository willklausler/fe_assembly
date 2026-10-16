program bench
  !! Times element R and K assembly for every material and configuration, and
  !! the material call alone on the same deformation gradients.
  !! Usage: fpm run --profile release [-- nelem [ntrial]]
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature, CUB_HEX
  use shapefuncs, only: shapefunc
  use tensors, only: eye3
  use materials, only: material, material_set
  use elements, only: elem_ref, elem_cur, elem_mix, shape_gradients
  implicit none
  character(*), parameter :: cfgname(3) = ['ref', 'cur', 'mix']
  integer :: nelem = 2000, ntrial = 7
  type(cubature) :: q
  type(shapefunc) :: s
  type(material), allocatable :: mats(:)
  real(dp), allocatable :: xref(:,:,:), u(:,:,:), fg(:,:,:,:)
  real(dp) :: dndx(3,8), dv, telem, tmat, checksum = 0
  integer :: a, i, e, g, m, c

  call int_arg(1, nelem)
  call int_arg(2, ntrial)
  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  mats = material_set(1.2_dp, 0.8_dp)

  ! Distinct distorted elements under moderate deformation, so no two calls
  ! see the same data; F at every point feeds the material-only timing.
  allocate(xref(3, 8, nelem), u(3, 8, nelem), fg(3, 3, q%npoints, nelem))
  do e = 1, nelem
    do a = 1, 8
      do i = 1, 3
        xref(i,a,e) = (s%coords(i,a) + 1)/2 + 0.05_dp*cos(real(3*a + i + 7*e, dp))
        u(i,a,e) = 0.05_dp*sin(real(5*a + 2*i + 11*e, dp))
      end do
    end do
    do g = 1, q%npoints
      call shape_gradients(q, s, g, xref(:,:,e), dndx, dv)
      fg(:,:,g,e) = eye3 + matmul(u(:,:,e), transpose(dndx))
    end do
  end do

  write(*, '(a, i0, a, i0, a)') 'hex8, 2x2x2 points, ', nelem, ' elements, best of ', ntrial, ' trials'
  write(*, '(a8, a5, 3a12)') 'model', 'cfg', 'element ns', 'material ns', 'material %'
  do m = 1, size(mats)
    do c = 1, 3
      telem = best_time(m, c, .false.)
      tmat = best_time(m, c, .true.)
      write(*, '(a8, a5, 2f12.0, f12.1)') trim(mats(m)%name), cfgname(c), &
        & 1e9_dp*telem/nelem, 1e9_dp*tmat/nelem, 100*tmat/telem
    end do
  end do
  write(*, '(a, es12.4)') 'checksum ', checksum
  ! gfortran does not free main-program allocatables at exit; LeakSanitizer would report them
  deallocate(mats, xref, u, fg)

contains

subroutine int_arg(n, val)
  ! replace val by command argument n, if present
  integer, intent(in) :: n
  integer, intent(inout) :: val
  character(32) :: arg
  integer :: stat
  call get_command_argument(n, arg, status=stat)
  if (stat == 0) read(arg, *) val
end subroutine int_arg

function best_time(m, c, matonly) result(t)
  ! minimum process CPU time over ntrial sweeps, in seconds
  ! cpu_time, not system_clock: ifx's system_clock follows the wall clock, which
  ! WSL2 re-syncs with jumps (one sweep measured -1 s); the kernels are single-threaded
  integer, intent(in) :: m, c
  logical, intent(in) :: matonly
  real(dp) :: t, t0, t1
  integer :: trial
  t = huge(t)
  do trial = 1, ntrial
    call cpu_time(t0)
    call sweep(m, c, matonly)
    call cpu_time(t1)
    t = min(t, t1 - t0)
  end do
end function best_time

subroutine sweep(m, c, matonly)
  ! one pass over all elements; results feed the checksum so no call is optimized away
  integer, intent(in) :: m, c
  logical, intent(in) :: matonly
  real(dp) :: r(24), k(24,24), sv(6, q%npoints), cv(6,6,q%npoints), p(3,3,q%npoints), am(9,9,q%npoints)
  integer :: e
  associate (mt => mats(m))
    do e = 1, nelem
      if (matonly) then
        ! one batched call per element, as the element routines make
        select case (c)
        case (1)
          call mt%ref(q%npoints, fg(:,:,:,e), sv, cv)
        case (2)
          call mt%cur(q%npoints, fg(:,:,:,e), sv, cv)
        case default
          call mt%mix(q%npoints, fg(:,:,:,e), p, am)
          cv(1,1,1) = am(1,1,1)
        end select
        checksum = checksum + cv(1,1,1)
      else
        select case (c)
        case (1)
          call elem_ref(q, s, xref(:,:,e), u(:,:,e), mt, r, k)
        case (2)
          call elem_cur(q, s, xref(:,:,e), u(:,:,e), mt, r, k)
        case default
          call elem_mix(q, s, xref(:,:,e), u(:,:,e), mt, r, k)
        end select
        checksum = checksum + k(1,1)
      end if
    end do
  end associate
end subroutine sweep

end program bench
