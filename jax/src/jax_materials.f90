! SPDX-License-Identifier: MIT
module jax_materials
  !! Materials evaluated by JAX automatic differentiation through embedded Python.
  !!
  !! Type jax_material extends materials%remote_material: make creates an fe_jax
  !! handle, evaluate is one Fortran -> C -> Python -> JAX call for n points.
  !! jax_set compiles and verifies every handle once (remote_setup); JAX
  !! recompiles for each new batch size. Call jax_init once before jax_set and
  !! jax_finalize at the end.
  use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char, c_null_char
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use materials, only: remote_material, remote_setup
  implicit none
  private

  public :: jax_material, jax_init, jax_set, jax_finalize, jax_ping, wtime

  integer :: hnoop = -1
    ! handle of fe_jax.make(..., 'noop'): the bridge without JAX

  type, extends(remote_material) :: jax_material
    !! a material whose batched methods call JAX
  contains
    procedure :: make => jax_make
    procedure :: evaluate => jax_evaluate
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
  !! the material_set models with JAX methods for cfgs, compiled for batches of nwarm
  real(dp), intent(in) :: lam, mu
  character(*), intent(in) :: cfgs(:)
  integer, intent(in) :: nwarm
  type(jax_material) :: m(4)
  call remote_setup(m, lam, mu, cfgs, nwarm)
end function jax_set

subroutine jax_finalize()
  !! release the handles and stop Python
  call pyb_finalize()
end subroutine jax_finalize

subroutine jax_ping(n, f, stress, tangent)
  !! the bridge round trip on n points with a Python function that does nothing
  integer, intent(in) :: n
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: stress(6,n), tangent(36,n)
  if (pyb_eval(hnoop, n, f, [0._dp], 1, stress, 6, tangent, 36) /= 0) error stop 'jax_ping: failed'
end subroutine jax_ping

function jax_make(self, cfg) result(h)
  class(jax_material), intent(in) :: self
  character(*), intent(in) :: cfg
  integer :: h
  h = pyb_handle(trim(self%name) // c_null_char, cfg // c_null_char)
end function jax_make

subroutine jax_evaluate(self, h, n, f, stress, ns, tangent, nt)
  ! the return code is the only run-time check: Python can raise at any call
  class(jax_material), intent(in) :: self
  integer, intent(in) :: h, n, ns, nt
  real(dp), intent(in) :: f(3,3,n)
  real(dp), intent(out) :: stress(ns,n), tangent(nt,n)
  if (pyb_eval(h, n, f, self%par, size(self%par), stress, ns, tangent, nt) /= 0) &
    & error stop 'jax_evaluate: Python call failed for ' // trim(self%name)
end subroutine jax_evaluate

end module jax_materials
