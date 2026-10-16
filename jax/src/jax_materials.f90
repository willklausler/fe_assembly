! SPDX-License-Identifier: MIT
module jax_materials
  !! Materials evaluated by JAX automatic differentiation through embedded Python.
  !!
  !! Type jax_material extends materials%material and overrides only the batched
  !! methods ref, cur, mix: each is one Fortran -> C -> Python -> JAX call for all
  !! n points. The per-point native routines and the energy stay available.
  !! Call jax_init once before jax_set, and jax_finalize at the end.
  !! jax_set compiles and verifies every handle once (stress-free, finite tangent
  !! at F = I) for the batch size it will see; JAX recompiles for each new n.
  use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char, c_null_char
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use tensors, only: eye3
  use materials, only: material, material_set
  implicit none
  private

  public :: jax_material, jax_init, jax_set, jax_finalize, jax_ping, wtime

  character(*), parameter :: cfgname(3) = ['ref', 'cur', 'mix']
  integer :: hnoop = -1
    ! handle of fe_jax.make(..., 'noop'): the bridge without JAX

  type, extends(material) :: jax_material
    !! a material whose batched methods call JAX
    integer :: h(3) = -1
      !! bridge handles for ref, cur, mix; -1 if not available
  contains
    procedure :: ref => jax_ref
    procedure :: cur => jax_cur
    procedure :: mix => jax_mix
  end type jax_material

  interface
    integer(c_int) function pyb_init(python_exe, module_dir) bind(C)
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: python_exe(*), module_dir(*)
    end function pyb_init
    integer(c_int) function pyb_handle(model, cfg) bind(C)
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: model(*), cfg(*)
    end function pyb_handle
    integer(c_int) function pyb_eval(h, n, f, par, npar, stress, ns, tangent, nt) bind(C)
      import :: c_int, c_double
      integer(c_int), value :: h, n, npar, ns, nt
      real(c_double), intent(in) :: f(*), par(*)
      real(c_double), intent(out) :: stress(*), tangent(*)
    end function pyb_eval
    subroutine pyb_finalize() bind(C)
    end subroutine pyb_finalize
    real(c_double) function wtime() bind(C, name='pyb_wtime')
      !! monotonic wall time in seconds
      import :: c_double
    end function wtime
  end interface

contains

subroutine jax_init(python_exe, module_dir)
  !! start Python as python_exe (a venv python) and import fe_jax from module_dir
  character(*), intent(in) :: python_exe, module_dir
  if (pyb_init(trim(python_exe) // c_null_char, trim(module_dir) // c_null_char) /= 0) &
    & error stop 'jax_init: Python or fe_jax failed to start'
  hnoop = pyb_handle('svk' // c_null_char, 'noop' // c_null_char)
  if (hnoop < 0) error stop 'jax_init: fe_jax.make failed for noop'
end subroutine jax_init

function jax_set(lam, mu, cfgs, nwarm) result(m)
  !! the native material_set models with JAX methods for the configurations in cfgs,
  !! compiled and verified for batches of nwarm points
  real(dp), intent(in) :: lam, mu
  character(*), intent(in) :: cfgs(:)
    !! subset of 'ref', 'cur', 'mix'
  integer, intent(in) :: nwarm
    !! batch size of the first call (8 for one hex8 element)
  type(jax_material), allocatable :: m(:)
  type(material), allocatable :: native(:)
  ! flat buffers: evaluate sees them as (ns, n), so only the first ns*n entries are written
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
      m(i)%h(c) = pyb_handle(trim(m(i)%name) // c_null_char, cfgname(c) // c_null_char)
      if (m(i)%h(c) < 0) error stop 'jax_set: fe_jax.make failed for ' // trim(m(i)%name)
      ! compile, then verify: every model is stress-free at F = I with a finite tangent
      ns = merge(9, 6, c == 3)
      call evaluate(m(i), c, nwarm, f, stress, ns, tangent, ns*ns)
      if (any(.not. ieee_is_finite(tangent(:ns*ns*nwarm))) .or. maxval(abs(stress(:ns*nwarm))) > 1e-12_dp) &
        & error stop 'jax_set: ' // trim(m(i)%name) // ' ' // cfgname(c) // ' fails at F = I'
    end do
  end do
end function jax_set

subroutine jax_ping(n, f, stress, tangent)
  !! the bridge round trip on n points with a Python function that does nothing
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: stress(6,n), tangent(36,n)
  if (pyb_eval(hnoop, n, f, [0._dp], 1, stress, 6, tangent, 36) /= 0) error stop 'jax_ping: failed'
end subroutine jax_ping

subroutine jax_finalize()
  !! release the handles and stop Python
  call pyb_finalize()
end subroutine jax_finalize

subroutine evaluate(self, c, n, f, stress, ns, tangent, nt)
  ! one bridge call; stops if the configuration has no handle or Python raised
  class(jax_material), intent(in) :: self
  integer, intent(in) :: c, n, ns, nt
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: stress(ns,n), tangent(nt,n)
  if (self%h(c) < 0) error stop 'jax_material: no JAX handle for ' // cfgname(c)
  if (pyb_eval(self%h(c), n, f, self%par, size(self%par), stress, ns, tangent, nt) /= 0) &
    & error stop 'jax_material: Python call failed for ' // trim(self%name) // ' ' // cfgname(c)
end subroutine evaluate

subroutine jax_ref(self, n, f, s, c)
  class(jax_material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: s(6,n), c(6,6,n)
  call evaluate(self, 1, n, f, s, 6, c, 36)
end subroutine jax_ref

subroutine jax_cur(self, n, f, s, c)
  class(jax_material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: s(6,n), c(6,6,n)
  call evaluate(self, 2, n, f, s, 6, c, 36)
end subroutine jax_cur

subroutine jax_mix(self, n, f, p, a)
  class(jax_material), intent(in) :: self
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: p(3,3,n), a(9,9,n)
  call evaluate(self, 3, n, f, p, 9, a, 81)
end subroutine jax_mix

end module jax_materials
