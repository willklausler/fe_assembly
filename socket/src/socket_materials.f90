! SPDX-License-Identifier: MIT
module socket_materials
  !! Materials evaluated by a separate Python server over a TCP socket (FEAP style).
  !!
  !! Type socket_material extends materials%material and overrides only the batched
  !! methods ref, cur, mix: each is one synchronous request/response for n points.
  !! The client is plain Fortran bound to the POSIX socket API (Linux: MSG_MORE,
  !! MSG_NOSIGNAL). socket_start launches python/server.py and connects;
  !! socket_set creates the server-side handles, compiling and verifying each once
  !! at F = I; socket_stop shuts the server down. The connection is module state:
  !! one server per program.
  use, intrinsic :: iso_c_binding, only: c_int, c_int16_t, c_int32_t, c_int64_t, c_long, &
    & c_size_t, c_intptr_t, c_char, c_ptr, c_loc, c_null_char, c_double
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use tensors, only: eye3
  use materials, only: material, material_set
  implicit none
  private

  public :: socket_material, socket_start, socket_set, socket_stop, socket_ping, wtime

  character(*), parameter :: cfgname(3) = ['ref', 'cur', 'mix']
  integer(c_int), parameter :: AF_INET = 2, SOCK_STREAM = 1, IPPROTO_TCP = 6, TCP_NODELAY = 1
  integer(c_int), parameter :: MSG_NOSIGNAL = int(z'4000'), MSG_MORE = int(z'8000')
  integer(c_int), parameter :: CLOCK_MONOTONIC = 1
  integer(c_int32_t), parameter :: OP_SHUTDOWN = 0, OP_MAKE = 1, OP_EVAL = 2, OP_NOOP = 3
  integer(c_int) :: fd = -1
    ! the connected socket

  type, extends(material) :: socket_material
    !! a material whose batched methods call the server
    integer :: h(3) = -1
      !! server handles for ref, cur, mix; -1 if not available
  contains
    procedure :: ref => socket_ref
    procedure :: cur => socket_cur
    procedure :: mix => socket_mix
  end type socket_material

  type, bind(C) :: sockaddr_in
    integer(c_int16_t) :: sin_family
    integer(c_int16_t) :: sin_port
      ! network byte order
    integer(c_int32_t) :: sin_addr
      ! network byte order
    character(kind=c_char) :: sin_zero(8)
  end type sockaddr_in

  type, bind(C) :: timespec
    integer(c_long) :: tv_sec, tv_nsec
  end type timespec

  interface
    integer(c_int) function c_socket(domain, stype, protocol) bind(C, name='socket')
      import :: c_int
      integer(c_int), value :: domain, stype, protocol
    end function c_socket
    integer(c_int) function c_connect(sockfd, addr, addrlen) bind(C, name='connect')
      import :: c_int, c_ptr
      integer(c_int), value :: sockfd, addrlen
      type(c_ptr), value :: addr
    end function c_connect
    integer(c_int) function c_setsockopt(sockfd, level, optname, optval, optlen) bind(C, name='setsockopt')
      import :: c_int, c_ptr
      integer(c_int), value :: sockfd, level, optname, optlen
      type(c_ptr), value :: optval
    end function c_setsockopt
    ! ssize_t is long on Linux
    integer(c_long) function c_send(sockfd, buf, length, flags) bind(C, name='send')
      import :: c_int, c_long, c_ptr, c_size_t
      integer(c_int), value :: sockfd, flags
      type(c_ptr), value :: buf
      integer(c_size_t), value :: length
    end function c_send
    integer(c_long) function c_recv(sockfd, buf, length, flags) bind(C, name='recv')
      import :: c_int, c_long, c_ptr, c_size_t
      integer(c_int), value :: sockfd, flags
      type(c_ptr), value :: buf
      integer(c_size_t), value :: length
    end function c_recv
    integer(c_int) function c_close(sockfd) bind(C, name='close')
      import :: c_int
      integer(c_int), value :: sockfd
    end function c_close
    integer(c_int16_t) function c_htons(hostshort) bind(C, name='htons')
      import :: c_int16_t
      integer(c_int16_t), value :: hostshort
    end function c_htons
    integer(c_int32_t) function c_inet_addr(cp) bind(C, name='inet_addr')
      import :: c_int32_t, c_char
      character(kind=c_char), intent(in) :: cp(*)
    end function c_inet_addr
    integer(c_int) function c_usleep(usec) bind(C, name='usleep')
      import :: c_int, c_int32_t
      integer(c_int32_t), value :: usec
    end function c_usleep
    integer(c_int) function c_clock_gettime(clk, ts) bind(C, name='clock_gettime')
      import :: c_int, timespec
      integer(c_int), value :: clk
      type(timespec), intent(out) :: ts
    end function c_clock_gettime
  end interface

contains

function wtime() result(t)
  !! monotonic wall time in seconds
  real(dp) :: t
  type(timespec) :: ts
  if (c_clock_gettime(CLOCK_MONOTONIC, ts) /= 0) error stop 'wtime: clock_gettime failed'
  t = real(ts%tv_sec, dp) + 1e-9_dp*real(ts%tv_nsec, dp)
end function wtime

subroutine socket_start(python, port)
  !! launch python/server.py with interpreter python on port, and connect to it
  character(*), intent(in) :: python
  integer, intent(in) :: port
  type(sockaddr_in), target :: addr
  integer(c_int), target :: one
  integer(c_int) :: rc
  character(8) :: portstr
  integer :: try
  write(portstr, '(i0)') port
  call execute_command_line(trim(python) // ' python/server.py ' // trim(portstr), wait=.false.)
  addr%sin_family = int(AF_INET, c_int16_t)
  addr%sin_port = c_htons(int(merge(port - 65536, port, port > 32767), c_int16_t))
  addr%sin_addr = c_inet_addr('127.0.0.1' // c_null_char)
  addr%sin_zero = c_null_char
  ! the server needs a few seconds to import JAX before it listens
  do try = 1, 600
    fd = c_socket(AF_INET, SOCK_STREAM, 0)
    if (fd < 0) error stop 'socket_start: socket failed'
    if (c_connect(fd, c_loc(addr), int(storage_size(addr)/8, c_int)) == 0) exit
    if (c_close(fd) /= 0) error stop 'socket_start: close failed'
    fd = -1
    rc = c_usleep(50000_c_int32_t)
  end do
  if (fd < 0) error stop 'socket_start: no server on port ' // trim(portstr)
  ! request/response traffic: never wait to coalesce small segments
  one = 1
  if (c_setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, c_loc(one), 4_c_int) /= 0) &
    & error stop 'socket_start: TCP_NODELAY failed'
end subroutine socket_start

function socket_set(lam, mu, cfgs, nwarm) result(m)
  !! the native material_set models with server methods for the configurations in cfgs,
  !! compiled and verified for batches of nwarm points
  real(dp), intent(in) :: lam, mu
  character(*), intent(in) :: cfgs(:)
    !! subset of 'ref', 'cur', 'mix'
  integer, intent(in) :: nwarm
    !! batch size of the first call (8 for one hex8 element)
  type(socket_material), allocatable :: m(:)
  type(material), allocatable :: native(:)
  real(dp) :: f(3,3,nwarm), stress(9*nwarm), tangent(81*nwarm)
  integer :: i, c, g, ns
  native = material_set(lam, mu)
  allocate(m(size(native)))
  do g = 1, nwarm
    f(:,:,g) = eye3
  end do
  do i = 1, size(native)
    m(i)%material = native(i)
    do c = 1, 3
      if (.not. any(cfgs == cfgname(c))) cycle
      m(i)%h(c) = make_handle(m(i)%name, cfgname(c), m(i)%par)
      if (m(i)%h(c) < 0) error stop 'socket_set: server make failed for ' // trim(m(i)%name)
      ! compile, then verify: every model is stress-free at F = I with a finite tangent
      ns = merge(9, 6, c == 3)
      call evaluate(OP_EVAL, m(i)%h(c), nwarm, f, stress, ns, tangent, ns*ns)
      if (any(.not. ieee_is_finite(tangent(:ns*ns*nwarm))) .or. maxval(abs(stress(:ns*nwarm))) > 1e-12_dp) &
        & error stop 'socket_set: ' // trim(m(i)%name) // ' ' // cfgname(c) // ' fails at F = I'
    end do
  end do
end function socket_set

subroutine socket_stop()
  !! shut the server down and close the connection
  call send_header(OP_SHUTDOWN, 0, 0, 0, 0_c_int)
  if (c_close(fd) /= 0) error stop 'socket_stop: close failed'
  fd = -1
end subroutine socket_stop

subroutine socket_ping(n, f, stress, tangent)
  !! the transport round trip on n points (ref-sized reply) with nothing computed
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: stress(6,n), tangent(36,n)
  call evaluate(OP_NOOP, 0, n, f, stress, 6, tangent, 36)
end subroutine socket_ping

!-----------------------------------------------------------------------
! Transport
!-----------------------------------------------------------------------

function offset(p, nbytes) result(q)
  ! address p + nbytes, for resuming a partial send or recv
  type(c_ptr), intent(in) :: p
  integer(c_size_t), intent(in) :: nbytes
  type(c_ptr) :: q
  q = transfer(transfer(p, 0_c_intptr_t) + int(nbytes, c_intptr_t), q)
end function offset

subroutine send_all(p, nbytes, flags)
  ! send nbytes from address p; TCP may accept fewer per call
  type(c_ptr), intent(in) :: p
  integer(c_size_t), intent(in) :: nbytes
  integer(c_int), intent(in) :: flags
  integer(c_size_t) :: done
  integer(c_long) :: rc
  done = 0
  do while (done < nbytes)
    rc = c_send(fd, offset(p, done), nbytes - done, ior(flags, MSG_NOSIGNAL))
    if (rc <= 0) error stop 'socket_materials: send failed (server gone?)'
    done = done + int(rc, c_size_t)
  end do
end subroutine send_all

subroutine recv_all(p, nbytes)
  ! receive exactly nbytes into address p
  type(c_ptr), intent(in) :: p
  integer(c_size_t), intent(in) :: nbytes
  integer(c_size_t) :: done
  integer(c_long) :: rc
  done = 0
  do while (done < nbytes)
    rc = c_recv(fd, offset(p, done), nbytes - done, 0_c_int)
    if (rc <= 0) error stop 'socket_materials: recv failed (server gone?)'
    done = done + int(rc, c_size_t)
  end do
end subroutine recv_all

subroutine send_header(op, h, n, nbytes, flags)
  ! int32 op, handle, n, payload bytes
  integer(c_int32_t), intent(in) :: op
  integer, intent(in) :: h, n, nbytes
  integer(c_int), intent(in) :: flags
  integer(c_int32_t), target :: header(4)
  header = [op, int(h, c_int32_t), int(n, c_int32_t), int(nbytes, c_int32_t)]
  call send_all(c_loc(header), 16_c_size_t, flags)
end subroutine send_header

function make_handle(model, cfg, par) result(h)
  ! server handle of fe_jax.make(model, cfg) holding par, or -1
  character(*), intent(in) :: model, cfg
  real(dp), intent(in) :: par(:)
  integer :: h
  character(kind=c_char, len=20 + 8*size(par)), target :: payload
  integer(c_int64_t), target :: reply
  character(16) :: name16
  character(4) :: cfg4
  name16 = model
  cfg4 = cfg
  payload = name16 // cfg4 // transfer(par, repeat(' ', 8*size(par)))
  call send_header(OP_MAKE, 0, 0, len(payload), MSG_MORE)
  call send_all(c_loc(payload), int(len(payload), c_size_t), 0_c_int)
  call recv_all(c_loc(reply), 8_c_size_t)
  h = int(reply)
end function make_handle

subroutine evaluate(op, h, n, f, stress, ns, tangent, nt)
  ! one request/response: F(3,3,n) out, status, stress(ns,n), tangent(nt,n) back
  integer(c_int32_t), intent(in) :: op
  integer, intent(in) :: h, n, ns, nt
  real(dp), intent(in), target :: f(3,3,n)
  real(dp), intent(out), target :: stress(ns,n), tangent(nt,n)
  integer(c_int64_t), target :: status
  ! the header rides in the same TCP segment as F (MSG_MORE)
  call send_header(op, h, n, 72*n, MSG_MORE)
  call send_all(c_loc(f), int(72*n, c_size_t), 0_c_int)
  call recv_all(c_loc(status), 8_c_size_t)
  call recv_all(c_loc(stress), int(8*ns*n, c_size_t))
  call recv_all(c_loc(tangent), int(8*nt*n, c_size_t))
  if (status /= 0) error stop 'socket_materials: server failed to evaluate'
end subroutine evaluate

subroutine call_server(self, c, n, f, stress, ns, tangent, nt)
  ! one evaluation; stops if the configuration has no handle
  class(socket_material), intent(in) :: self
  integer, intent(in) :: c, n, ns, nt
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: stress(ns,n), tangent(nt,n)
  if (self%h(c) < 0) error stop 'socket_material: no server handle for ' // cfgname(c)
  call evaluate(OP_EVAL, self%h(c), n, f, stress, ns, tangent, nt)
end subroutine call_server

subroutine socket_ref(self, n, f, s, c)
  class(socket_material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: s(6,n), c(6,6,n)
  call call_server(self, 1, n, f, s, 6, c, 36)
end subroutine socket_ref

subroutine socket_cur(self, n, f, s, c)
  class(socket_material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: s(6,n), c(6,6,n)
  call call_server(self, 2, n, f, s, 6, c, 36)
end subroutine socket_cur

subroutine socket_mix(self, n, f, p, a)
  class(socket_material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: p(3,3,n), a(9,9,n)
  call call_server(self, 3, n, f, p, 9, a, 81)
end subroutine socket_mix

end module socket_materials
