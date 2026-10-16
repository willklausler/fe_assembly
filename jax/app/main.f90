program bench_jax
  !! Native versus JAX material evaluation: JIT time of the first call, then wall
  !! time per point for batches of 1 point, 1 element (8 points) and the whole
  !! mesh, and the element time with each.
  !! Usage: ./run.sh run --profile release [-- nelem [ntrial]]
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature, CUB_HEX
  use shapefuncs, only: shapefunc
  use tensors, only: eye3
  use materials, only: material, material_set
  use elements, only: elem_ref, elem_cur, elem_mix, shape_gradients
  use jax_materials, only: jax_material, jax_init, jax_set, jax_finalize, jax_ping, wtime
  implicit none
  character(*), parameter :: cfgname(3) = ['ref', 'cur', 'mix']
  integer, parameter :: cfgs(3) = [1, 2, 3]
  integer :: nelem = 200, ntrial = 5, npt, ntot
  character(256) :: python
  type(cubature) :: q
  type(shapefunc) :: s
  type(material), allocatable :: nat(:)
  type(jax_material), allocatable :: jx(:)
  real(dp), allocatable :: xref(:,:,:), u(:,:,:), fg(:,:,:), st(:,:), tg(:,:)
  real(dp) :: dndx(3,8), dv, tjit, t0, tn, t1, t8, tall, en, ej, checksum = 0
  integer :: a, i, e, g, m, ic, c, stat

  call int_arg(1, nelem)
  call int_arg(2, ntrial)
  call get_environment_variable('FE_PYTHON', python, status=stat)
  if (stat /= 0) python = '../.venv/bin/python'
  call jax_init(python, 'python')

  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  npt = q%npoints
  ntot = npt*nelem
  nat = material_set(1.2_dp, 0.8_dp)
  jx = jax_set(1.2_dp, 0.8_dp, ['ref', 'cur', 'mix'], npt)

  ! Same elements as the native bench; F of element e at columns npt (e - 1) + 1 ... npt e
  allocate(xref(3, 8, nelem), u(3, 8, nelem), fg(3, 3, ntot), st(9, ntot), tg(81, ntot))
  do e = 1, nelem
    do a = 1, 8
      do i = 1, 3
        xref(i,a,e) = (s%coords(i,a) + 1)/2 + 0.05_dp*cos(real(3*a + i + 7*e, dp))
        u(i,a,e) = 0.05_dp*sin(real(5*a + 2*i + 11*e, dp))
      end do
    end do
    do g = 1, npt
      call shape_gradients(q, s, g, xref(:,:,e), dndx, dv)
      fg(:,:,npt*(e - 1) + g) = eye3 + matmul(u(:,:,e), transpose(dndx))
    end do
  end do

  write(*, '(a, i0, a, i0, a)') 'hex8, ', nelem, ' elements, best of ', ntrial, ' trials, wall time'
  ! the bridge alone: Fortran -> C -> Python and back, no JAX
  write(*, '(a, 3f9.2, a)') 'bridge round trip without JAX, us per call (n = 1, 8, mesh):', &
    & 1e6_dp*ping(1), 1e6_dp*ping(npt), 1e6_dp*ping(ntot)
  write(*, '(a)') '                     -------- material, ns per point --------   -- element, us --'
  write(*, '(a5, a4, a9, 5a11, 2a9)') 'model', 'cfg', 'JIT1 s', 'native', 'JAX 1/call', 'JAX 8/call', &
    & 'JAX mesh', 'JAX/nat 8', 'native', 'JAX'
  do m = 1, size(nat)
    do ic = 1, size(cfgs)
      c = cfgs(ic)
      ! jax_set compiled for npt; the first call of another batch size compiles again
      t0 = wtime()
      call batches(jx(m), c, 1)
      tjit = wtime() - t0
      call batches(jx(m), c, 1)
      call batches(jx(m), c, ntot)
      tn = best(nat(m), c, npt, .false.)
      t1 = best(jx(m), c, 1, .false.)
      t8 = best(jx(m), c, npt, .false.)
      tall = best(jx(m), c, ntot, .false.)
      en = best(nat(m), c, npt, .true.)
      ej = best(jx(m), c, npt, .true.)
      write(*, '(a5, a4, f9.2, 4f11.0, f11.1, 2f9.1)') trim(nat(m)%name), cfgname(c), tjit, &
        & 1e9_dp*[tn, t1, t8, tall]/ntot, t8/tn, 1e6_dp*[en, ej]/nelem
    end do
  end do
  write(*, '(a, es12.4)') 'checksum ', checksum

  call jax_finalize()
  deallocate(nat, jx, xref, u, fg, st, tg)

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

function best(mt, c, nb, elements) result(t)
  ! minimum wall time over ntrial sweeps, in seconds
  class(material), intent(in) :: mt
  integer, intent(in) :: c, nb
  logical, intent(in) :: elements
  real(dp) :: t, t0
  integer :: trial
  t = huge(t)
  do trial = 1, ntrial
    t0 = wtime()
    if (elements) then
      call element_sweep(mt, c)
    else
      call batches(mt, c, nb)
    end if
    t = min(t, wtime() - t0)
  end do
end function best

function ping(nb) result(t)
  ! best wall time of one bridge round trip without JAX on nb points, in seconds
  integer, intent(in) :: nb
  real(dp) :: t, t0
  integer :: trial, k
  t = huge(t)
  do trial = 1, ntrial
    t0 = wtime()
    do k = 1, ntot, nb
      call jax_ping(nb, fg(:,:,k), st(:,k), tg(:,k))
    end do
    t = min(t, (wtime() - t0)/(ntot/nb))
  end do
end function ping

subroutine batches(mt, c, nb)
  ! all ntot points in calls of nb points
  class(material), intent(in) :: mt
  integer, intent(in) :: c, nb
  integer :: k
  do k = 1, ntot, nb
    select case (c)
    case (1)
      call mt%ref(nb, fg(:,:,k), st(:,k), tg(:,k))
    case (2)
      call mt%cur(nb, fg(:,:,k), st(:,k), tg(:,k))
    case default
      call mt%mix(nb, fg(:,:,k), st(:,k), tg(:,k))
    end select
  end do
  checksum = checksum + tg(1,1)
end subroutine batches

subroutine element_sweep(mt, c)
  ! R and K of every element
  class(material), intent(in) :: mt
  integer, intent(in) :: c
  real(dp) :: r(24), k(24,24)
  integer :: e
  do e = 1, nelem
    select case (c)
    case (1)
      call elem_ref(q, s, xref(:,:,e), u(:,:,e), mt, r, k)
    case (2)
      call elem_cur(q, s, xref(:,:,e), u(:,:,e), mt, r, k)
    case default
      call elem_mix(q, s, xref(:,:,e), u(:,:,e), mt, r, k)
    end select
    checksum = checksum + k(1,1)
  end do
end subroutine element_sweep

end program bench_jax
