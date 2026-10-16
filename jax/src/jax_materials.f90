! SPDX-License-Identifier: MIT
module jax_materials
  !! Materials evaluated by JAX automatic differentiation through embedded Python.
  !!
  !! Type jax_material extends materials%material and overrides only the batched
  !! methods ref, cur, mix: each is one Fortran -> C -> Python -> JAX call for all
  !! n points. The per-point native routines and the energy stay available.
  !! Call jax_init once before jax_set, and jax_finalize at the end.
  use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char, c_null_char
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use materials, only: material, material_set
  implicit none
  private

  public :: jax_material, jax_init, jax_set, jax_finalize, wtime

  character(*), parameter :: cfgname(3) = ['ref', 'cur', 'mix']

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
end subroutine jax_init

function jax_set(lam, mu, cfgs) result(m)
  !! the native material_set models with JAX methods for the configurations in cfgs
  real(dp), intent(in) :: lam, mu
  character(*), intent(in) :: cfgs(:)
    !! subset of 'ref', 'cur', 'mix'
  type(jax_material), allocatable :: m(:)
  type(material), allocatable :: native(:)
  integer :: i, c
  native = material_set(lam, mu)
  allocate(m(size(native)))
  do i = 1, size(native)
    m(i)%material = native(i)
    do c = 1, 3
      if (any(cfgs == cfgname(c))) then
        m(i)%h(c) = pyb_handle(trim(m(i)%name) // c_null_char, cfgname(c) // c_null_char)
        if (m(i)%h(c) < 0) error stop 'jax_set: fe_jax.make failed for ' // trim(m(i)%name)
      end if
    end do
  end do
end function jax_set

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
