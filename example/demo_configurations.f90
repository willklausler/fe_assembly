program demo_configurations
  !! One Mooney-Rivlin hex8 element in the three configurations: the same R and symmetric K
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures
  use shapefuncs
  use materials
  use elements
  implicit none
  type(cubature) :: q
  type(shapefunc) :: s
  type(material) :: mats(4)
  real(dp) :: x(3,8), u(3,8), r(24,3), k(24,24,3)
  integer :: a

  q = cubature(CUB_HEX, 2)                               ! full integration: 2x2x2 points
  s = shapefunc(q, 1)                                    ! trilinear hexahedron
  mats = material_set(1.2_dp, 0.8_dp)                    ! svk, nh, mnh, mr; Lame 1.2, 0.8
  x = (s%coords + 1)/2                                   ! unit cube
  do a = 1, 8
    u(:,a) = [0.1_dp*x(1,a), -0.05_dp*x(2,a), 0.02_dp*x(3,a)*x(1,a)]
  end do
  call elem_ref(q, s, x, u, mats(4), r(:,1), k(:,:,1))   ! C, S, CC
  call elem_cur(q, s, x, u, mats(4), r(:,2), k(:,:,2))   ! b, sigma, cc
  call elem_mix(q, s, x, u, mats(4), r(:,3), k(:,:,3))   ! F, P, AA
  print '(a, es9.1)', 'R: cur - ref', maxval(abs(r(:,2) - r(:,1)))         ! ~1e-16
  print '(a, es9.1)', 'R: mix - ref', maxval(abs(r(:,3) - r(:,1)))         ! ~1e-16
  print '(a, es9.1)', 'K: mix - ref', maxval(abs(k(:,:,3) - k(:,:,1)))     ! ~1e-16
  print '(a, es9.1)', 'K - K^T', maxval(abs(k(:,:,3) - transpose(k(:,:,3))))   ! ~1e-16
end program demo_configurations
