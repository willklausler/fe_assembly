program volume_check
  !! Maps a unit-cube hex8 element through homogeneous deformations x = F X
  !! and integrates the current volume v = sum_g det(dx/dxi) w_g, which must
  !! equal J = det(F) times the reference volume V = 1.
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures, only: cubature, CUB_HEX
  use shapefuncs, only: shapefunc
  use tensors, only: det3
  implicit none
  integer, parameter :: ncase = 4
  character(*), parameter :: names(ncase) = [character(16) :: 'identity', 'uniaxial 1.5', &
                                           & 'simple shear 0.5', 'dilation 1.1']
  real(dp) :: f(3,3,ncase), xref(3,8), x(3,8), jac(3,3), v
  type(cubature) :: q
  type(shapefunc) :: s
  integer :: c, g

  f = 0
  do c = 1, ncase
    f(1,1,c) = 1; f(2,2,c) = 1; f(3,3,c) = 1
  end do
  f(1,1,2) = 1.5_dp
  f(1,2,3) = 0.5_dp
  f(:,:,4) = 1.1_dp*f(:,:,1)

  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  xref = (s%coords + 1)/2

  write(*, '(a16, 2a12)') 'case', 'det F', 'v / V'
  do c = 1, ncase
    x = matmul(f(:,:,c), xref)
    v = 0
    do g = 1, q%npoints
      jac = matmul(x, transpose(s%derv(:,:,g)))
      v = v + det3(jac)*q%weights(g)
    end do
    write(*, '(a16, 2f12.6)') names(c), det3(f(:,:,c)), v
  end do

end program volume_check
