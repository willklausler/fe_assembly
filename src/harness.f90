! SPDX-License-Identifier: MIT
module harness
  !! Test and benchmark support shared by every package of the project: deterministic
  !! distorted elements, pass/fail counting, best-of timing, and the checks and the
  !! timing table common to all material evaluation paradigms. The kernels
  !! (tensors, materials, elements) never use it.
  use, intrinsic :: iso_fortran_env, only: dp => real64, compiler_version, compiler_options
  use cubatures, only: cubature, CUB_LIN, CUB_QUA, CUB_HEX, CUB_TRI, CUB_TET, CUB_WED
  use shapefuncs, only: shapefunc
  use tensors, only: eye3
  use materials, only: material, cfgname
  use elements, only: elem_ref, elem_cur, elem_mix, shape_gradients
  implicit none
  private

  public :: test_element, element_gradients, assemble
  public :: check, close, check_summary
  public :: stopwatch, start_watch, cpu_seconds
  public :: bench_options, parse_options, selected, csv_open, csv_row
  public :: compare_paradigm, bench_paradigm

  integer :: ncheck = 0, nfail = 0
    ! checks counted by check, reported by check_summary

  abstract interface
    function clock_interface() result(t)
      !! a clock in seconds
      import :: dp
      real(dp) :: t
    end function clock_interface
    subroutine ping_interface(n, f, stress, tangent)
      !! a transport round trip on n points with nothing computed
      import :: dp
      integer, intent(in) :: n
      real(dp), intent(in) :: f(3,3,n)
      real(dp), intent(out) :: stress(6,n), tangent(36,n)
    end subroutine ping_interface
  end interface

  type :: bench_options
    !! command-line options shared by every bench; set defaults before parse_options
    integer :: nelem = 2000
      !! distinct elements per sweep
    integer :: ntrial = 7
      !! timings per measurement; the best is reported
    character(256) :: csv = 'none'
      !! CSV file for the results (long format), or none
    character(32) :: tag = 'untagged'
      !! value of the compiler column, set by tools/sweep.py
    character(64) :: models = 'all'
      !! comma-separated subset of svk, nh, mnh, mr
    character(64) :: cfgs = 'all'
      !! comma-separated subset of ref, cur, mix
    character(64) :: paradigms = 'all'
      !! comma-separated subset of the paradigms a bench compares
  end type bench_options

  type :: stopwatch
    !! best of ntrial timings: do while (sw%next()); call work(); end do; t = sw%best
    !! (a loop, not a procedure argument: gfortran passes internal procedures through
    !! stack trampolines, which crash under AddressSanitizer)
    integer :: left = 0
    real(dp) :: best = huge(1._dp), t0 = 0
    logical :: running = .false.
    procedure(clock_interface), pointer, nopass :: clock => null()
  contains
    procedure :: next => stopwatch_next
  end type stopwatch

contains

!-----------------------------------------------------------------------
! Elements
!-----------------------------------------------------------------------

subroutine test_element(s, e, amp, xref, u)
  !! element e of a family of distorted reference elements, with displacement amplitude amp
  type(shapefunc), intent(in) :: s
  integer, intent(in) :: e
  real(dp), intent(in) :: amp
  real(dp), intent(out) :: xref(3, s%nnodes), u(3, s%nnodes)
  integer :: a, i
  do a = 1, s%nnodes
    do i = 1, 3
      xref(i,a) = (s%coords(i,a) + 1)/2 + 0.05_dp*cos(real(3*a + i + 7*e, dp))
      u(i,a) = amp*sin(real(5*a + 2*i + 11*e, dp))
    end do
  end do
end subroutine test_element

subroutine element_gradients(q, s, xref, u, f)
  !! deformation gradient at every cubature point
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:)
  real(dp), intent(out) :: f(3, 3, q%npoints)
  real(dp) :: dndx(3, s%nnodes), dv
  integer :: g
  do g = 1, q%npoints
    call shape_gradients(q, s, g, xref, dndx, dv)
    f(:,:,g) = eye3 + matmul(u, transpose(dndx))
  end do
end subroutine element_gradients

subroutine assemble(c, q, s, xref, u, mat, r, k)
  !! element R and K in configuration c (1 ref, 2 cur, 3 mix)
  integer, intent(in) :: c
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: xref(:,:), u(:,:)
  class(material), intent(in) :: mat
  real(dp), intent(out) :: r(:), k(:,:)
  select case (c)
  case (1)
    call elem_ref(q, s, xref, u, mat, r, k)
  case (2)
    call elem_cur(q, s, xref, u, mat, r, k)
  case default
    call elem_mix(q, s, xref, u, mat, r, k)
  end select
end subroutine assemble

!-----------------------------------------------------------------------
! Checks
!-----------------------------------------------------------------------

subroutine check(cond, name)
  !! count a check; print its name if it fails
  logical, intent(in) :: cond
  character(*), intent(in) :: name
  ncheck = ncheck + 1
  if (.not. cond) then
    nfail = nfail + 1
    print '(a)', 'FAIL: ' // name
  end if
end subroutine check

pure logical function close(x, y, rtol)
  !! max-norm comparison relative to the size of y; flatten matrices with [m]
  real(dp), intent(in) :: x(:), y(:), rtol
  close = maxval(abs(x - y)) <= rtol*maxval(abs(y))
end function close

subroutine check_summary()
  !! print the count and stop with an error if any check failed
  print '(i0, a, i0, a)', ncheck - nfail, ' of ', ncheck, ' checks passed'
  if (nfail > 0) error stop 1
end subroutine check_summary

!-----------------------------------------------------------------------
! Timing and command line
!-----------------------------------------------------------------------

function start_watch(ntrial, clock) result(sw)
  !! a stopwatch for ntrial timings with clock
  integer, intent(in) :: ntrial
  procedure(clock_interface) :: clock
  type(stopwatch) :: sw
  sw%left = ntrial
  sw%clock => clock
end function start_watch

logical function stopwatch_next(sw)
  !! record the trial that just ended (if any); true while trials remain
  class(stopwatch), intent(inout) :: sw
  real(dp) :: t
  t = sw%clock()
  ! a non-positive interval cannot be real work: a clock glitch must not win the minimum
  ! (one nvfortran sweep reported -2.8 s once, not reproducible)
  if (sw%running .and. t > sw%t0) sw%best = min(sw%best, t - sw%t0)
  sw%running = sw%left > 0
  stopwatch_next = sw%running
  sw%left = sw%left - 1
  sw%t0 = sw%clock()
end function stopwatch_next

function cpu_seconds() result(t)
  !! process CPU time: monotonic, unlike ifx's system_clock under WSL2
  real(dp) :: t
  call cpu_time(t)
end function cpu_seconds

subroutine parse_options(opts, description)
  !! read the command line into opts (whose values are the defaults); --help lists the options
  ! FLAP is used here only: at module level nvfortran 26.3 debug builds emitted FLAP's type
  ! descriptor into every program using harness (duplicate symbols at link time)
  use flap, only: command_line_interface
  type(bench_options), intent(inout) :: opts
  character(*), intent(in) :: description
  type(command_line_interface) :: cli
  character(16) :: nelem, ntrial
  integer :: err
  write(nelem, '(i0)') opts%nelem
  write(ntrial, '(i0)') opts%ntrial
  call cli%init(description=description)
  call cli%add(switch='--nelem', switch_ab='-n', help='distinct elements per sweep', required=.false., &
    & act='store', def=nelem)
  call cli%add(switch='--ntrial', switch_ab='-t', help='timings per measurement; the best is reported', &
    & required=.false., act='store', def=ntrial)
  call cli%add(switch='--csv', switch_ab='-o', help='CSV file for the results, or none', required=.false., &
    & act='store', def=trim(opts%csv))
  call cli%add(switch='--tag', help='value of the compiler column in the CSV', required=.false., act='store', &
    & def=trim(opts%tag))
  call cli%add(switch='--models', help='comma-separated subset of svk,nh,mnh,mr, or all', required=.false., &
    & act='store', def=trim(opts%models))
  call cli%add(switch='--cfgs', help='comma-separated subset of ref,cur,mix, or all', required=.false., &
    & act='store', def=trim(opts%cfgs))
  call cli%add(switch='--paradigms', help='comma-separated subset of the paradigms compared, or all', &
    & required=.false., act='store', def=trim(opts%paradigms))
  call cli%parse(error=err)
  if (err /= 0) error stop 'parse_options: invalid command line (see --help)'
  call cli%get(switch='--nelem', val=opts%nelem, error=err)
  call cli%get(switch='--ntrial', val=opts%ntrial, error=err)
  call cli%get(switch='--csv', val=opts%csv, error=err)
  call cli%get(switch='--tag', val=opts%tag, error=err)
  call cli%get(switch='--models', val=opts%models, error=err)
  call cli%get(switch='--cfgs', val=opts%cfgs, error=err)
  call cli%get(switch='--paradigms', val=opts%paradigms, error=err)
  if (opts%nelem < 1 .or. opts%ntrial < 1) error stop 'parse_options: nelem and ntrial must be positive'
end subroutine parse_options

pure logical function selected(name, list)
  !! name is in the comma-separated list, or the list is all
  character(*), intent(in) :: name, list
  selected = trim(list) == 'all' .or. index(',' // trim(list) // ',', ',' // trim(name) // ',') > 0
end function selected

subroutine csv_open(opts, unit)
  !! open opts%csv with its header (compiler and options as # comments), or unit = -1 for none
  !! (newunit= returns negative units but never -1)
  type(bench_options), intent(in) :: opts
  integer, intent(out) :: unit
  unit = -1
  if (trim(opts%csv) == 'none') return
  open(newunit=unit, file=trim(opts%csv), status='replace', action='write')
  write(unit, '(a)') '# compiler_version: ' // compiler_version()
  write(unit, '(a)') '# compiler_options: ' // compiler_options()
  write(unit, '(a, i0, a, i0)') '# nelem: ', opts%nelem, ', ntrial: ', opts%ntrial
  write(unit, '(a)') 'paradigm,compiler,model,cfg,element,order,nnodes,npoints,batch,quantity,value'
end subroutine csv_open

subroutine csv_row(unit, opts, s, paradigm, model, cfg, batch, quantity, value)
  !! one measurement in long format (nothing if unit = -1)
  integer, intent(in) :: unit
  type(bench_options), intent(in) :: opts
  type(shapefunc), intent(in) :: s
  character(*), intent(in) :: paradigm, model, cfg, quantity
  integer, intent(in) :: batch
  real(dp), intent(in) :: value
  character(*), parameter :: names(6) = [character(5) :: 'line', 'quad', 'hex', 'tri', 'tet', 'wedge']
  integer, parameter :: codes(6) = [CUB_LIN, CUB_QUA, CUB_HEX, CUB_TRI, CUB_TET, CUB_WED]
  if (unit == -1) return
  write(unit, '(5(a, ","), 4(i0, ","), a, ",", g0.7)') paradigm, trim(opts%tag), model, cfg, &
    & trim(names(findloc(codes, s%elm, 1))), s%order, s%nnodes, s%npoints, batch, quantity, value
end subroutine csv_row

!-----------------------------------------------------------------------
! Shared by the paradigm packages (jax/, socket/)
!-----------------------------------------------------------------------

subroutine compare_paradigm(nat, rem, q, s, tol)
  !! checks: rem gives the stresses, tangents, R and K of the hand-derived nat
  !! in every configuration, and a batch of 1 equals a batch of all points
  type(material), intent(in) :: nat(:)
  class(material), intent(in) :: rem(:)
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  real(dp), intent(in) :: tol
  real(dp) :: xref(3, s%nnodes), u(3, s%nnodes), f(3, 3, q%npoints)
  ! flat buffers: the callee sees (ns, n) or (3, 3, n), so the first ns*n entries hold the result
  real(dp) :: s1(9*q%npoints), c1(81*q%npoints), s2(9*q%npoints), c2(81*q%npoints)
  real(dp) :: r1(3*s%nnodes), k1(3*s%nnodes, 3*s%nnodes), r2(3*s%nnodes), k2(3*s%nnodes, 3*s%nnodes)
  integer :: m, c, g, n, ns
  character(:), allocatable :: id
  n = q%npoints
  call test_element(s, 0, 0.1_dp, xref, u)
  call element_gradients(q, s, xref, u, f)
  do m = 1, size(nat)
    do c = 1, 3
      id = trim(nat(m)%name) // ' ' // cfgname(c) // ': '
      ns = merge(9, 6, c == 3)
      call evaluate(nat(m), c, n, f, s1, c1)
      call evaluate(rem(m), c, n, f, s2, c2)
      call check(close(s2(:ns*n), s1(:ns*n), tol), id // 'stress equals the closed form')
      call check(close(c2(:ns*ns*n), c1(:ns*ns*n), tol), id // 'tangent equals the closed form')
      do g = 1, n
        call evaluate(rem(m), c, 1, f(1,1,g), s2(ns*(g - 1) + 1), c2(ns*ns*(g - 1) + 1))
      end do
      call check(close(c2(:ns*ns*n), c1(:ns*ns*n), tol), id // 'batch of 1 equals the batch')
      call assemble(c, q, s, xref, u, nat(m), r1, k1)
      call assemble(c, q, s, xref, u, rem(m), r2, k2)
      call check(close(r2, r1, tol) .and. close([k2], [k1], tol), id // 'element R, K equal native')
    end do
  end do
end subroutine compare_paradigm

subroutine evaluate(mat, c, n, f, stress, tangent)
  ! batched call of configuration c; the buffers are passed on by sequence association,
  ! so stress holds (6,n) or (3,3,n) and tangent (6,6,n) or (9,9,n) from their first entry
  class(material), intent(in) :: mat
  integer, intent(in) :: c, n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: stress(9*n), tangent(81*n)
  select case (c)
  case (1)
    call mat%ref(n, f, stress, tangent)
  case (2)
    call mat%cur(n, f, stress, tangent)
  case default
    call mat%mix(n, f, stress, tangent)
  end select
end subroutine evaluate

subroutine bench_paradigm(nat, rem, q, s, opts, clock, ping, label)
  !! table and CSV rows: first-call (JIT) time, material time per point for batches of
  !! 1 point, 1 element and the whole mesh, and the element time, native against rem
  type(material), intent(in) :: nat(:)
  class(material), intent(in) :: rem(:)
  type(cubature), intent(in) :: q
  type(shapefunc), intent(in) :: s
  type(bench_options), intent(in) :: opts
  procedure(clock_interface) :: clock
  procedure(ping_interface) :: ping
  character(*), intent(in) :: label
    !! name of the paradigm (column heads and the CSV paradigm column)
  real(dp), allocatable :: xref(:,:,:), u(:,:,:), fg(:,:,:), st(:,:), tg(:,:)
  real(dp) :: tjit, t0, tn, t1, t8, tall, en, ej, checksum
  integer :: npt, ntot, e, m, c, nb, nelem, ntrial, csv
  character(:), allocatable :: model, cfg
  nelem = opts%nelem
  ntrial = opts%ntrial
  call csv_open(opts, csv)
  npt = q%npoints
  ntot = npt*nelem
  allocate(xref(3, s%nnodes, nelem), u(3, s%nnodes, nelem), fg(3, 3, ntot), st(9, ntot), tg(81, ntot))
  do e = 1, nelem
    call test_element(s, e, 0.05_dp, xref(:,:,e), u(:,:,e))
    call element_gradients(q, s, xref(:,:,e), u(:,:,e), fg(:,:,npt*(e - 1) + 1:npt*e))
  end do
  checksum = 0

  write(*, '(a, i0, a, i0, a, i0, a)') 'element: ', s%nnodes, ' nodes, ', nelem, ' elements, best of ', &
    & ntrial, ' trials, wall time'
  ! the transport alone: a request and a reply of the reference configuration's size
  nb = 1
  t1 = timed('pings')/ntot
  nb = npt
  t8 = timed('pings')/(ntot/npt)
  nb = ntot
  tall = timed('pings')
  write(*, '(a, 3f9.2)') 'transport alone, us per call (1 point, 1 element, mesh):', 1e6_dp*[t1, t8, tall]
  call csv_row(csv, opts, s, label, '-', '-', 1, 'transport_us_per_call', 1e6_dp*t1)
  call csv_row(csv, opts, s, label, '-', '-', npt, 'transport_us_per_call', 1e6_dp*t8)
  call csv_row(csv, opts, s, label, '-', '-', ntot, 'transport_us_per_call', 1e6_dp*tall)
  write(*, '(a)') '                     -------- material, ns per point --------   -- element, us --'
  write(*, '(a5, a4, a9, 5a11, 2a9)') 'model', 'cfg', 'JIT1 s', 'native', label // ' 1', label // ' elem', &
    & label // ' mesh', label // '/nat', 'native', label
  do m = 1, size(nat)
    do c = 1, 3
      model = trim(nat(m)%name)
      cfg = cfgname(c)
      if (.not. (selected(model, opts%models) .and. selected(cfg, opts%cfgs))) cycle
      ! setup compiled for npt points; the first call of another batch size compiles again
      nb = 1
      t0 = clock()
      call batches(rem(m))
      tjit = clock() - t0
      nb = ntot
      call batches(rem(m))
      nb = npt
      tn = timed('native_batches')
      t8 = timed('remote_batches')
      nb = 1
      t1 = timed('remote_batches')
      nb = ntot
      tall = timed('remote_batches')
      en = timed('native_elements')
      ej = timed('remote_elements')
      write(*, '(a5, a4, f9.2, 4f11.0, f11.1, 2f9.1)') model, cfg, tjit, &
        & 1e9_dp*[tn, t1, t8, tall]/ntot, t8/tn, 1e6_dp*[en, ej]/nelem
      call csv_row(csv, opts, s, label, model, cfg, 1, 'first_call_s', tjit)
      call csv_row(csv, opts, s, 'native', model, cfg, npt, 'material_ns_per_point', 1e9_dp*tn/ntot)
      call csv_row(csv, opts, s, label, model, cfg, 1, 'material_ns_per_point', 1e9_dp*t1/ntot)
      call csv_row(csv, opts, s, label, model, cfg, npt, 'material_ns_per_point', 1e9_dp*t8/ntot)
      call csv_row(csv, opts, s, label, model, cfg, ntot, 'material_ns_per_point', 1e9_dp*tall/ntot)
      call csv_row(csv, opts, s, 'native', model, cfg, npt, 'element_us', 1e6_dp*en/nelem)
      call csv_row(csv, opts, s, label, model, cfg, npt, 'element_us', 1e6_dp*ej/nelem)
    end do
  end do
  write(*, '(a, es12.4)') 'checksum ', checksum
  if (csv /= -1) close(csv)

contains

  function timed(what) result(t)
    ! best of ntrial runs of the sweep named what
    character(*), intent(in) :: what
    real(dp) :: t
    type(stopwatch) :: sw
    sw = start_watch(ntrial, clock)
    do while (sw%next())
      select case (what)
      case ('pings')
        call pings()
      case ('native_batches')
        call batches(nat(m))
      case ('remote_batches')
        call batches(rem(m))
      case ('native_elements')
        call elements(nat(m))
      case default
        call elements(rem(m))
      end select
    end do
    t = sw%best
  end function timed

  subroutine pings()
    integer :: k
    do k = 1, ntot, nb
      call ping(nb, fg(1,1,k), st(1,k), tg(1,k))
    end do
  end subroutine pings

  subroutine batches(mt)
    class(material), intent(in) :: mt
    integer :: k
    do k = 1, ntot, nb
      call evaluate(mt, c, nb, fg(1,1,k), st(1,k), tg(1,k))
    end do
    checksum = checksum + tg(1,1)
  end subroutine batches



  subroutine elements(mt)
    class(material), intent(in) :: mt
    real(dp) :: r(3*s%nnodes), k(3*s%nnodes, 3*s%nnodes)
    integer :: e
    do e = 1, nelem
      call assemble(c, q, s, xref(:,:,e), u(:,:,e), mt, r, k)
      checksum = checksum + k(1,1)
    end do
  end subroutine elements



end subroutine bench_paradigm

end module harness
