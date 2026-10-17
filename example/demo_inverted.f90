program demo_inverted
  !! A solver's use of the status flag: halve a displacement step until no point is inverted
  use, intrinsic :: iso_fortran_env, only: dp => real64
  use cubatures
  use shapefuncs
  use materials
  use elements
  implicit none
  type(cubature) :: q
  type(shapefunc) :: s
  type(material) :: mats(4)
  real(dp) :: x(3,8), du(3,8), r(24), k(24,24), step
  integer :: status, cuts

  q = cubature(CUB_HEX, 2)
  s = shapefunc(q, 1)
  mats = material_set(1.2_dp, 0.8_dp)
  x = (s%coords + 1)/2                                   ! unit cube
  du = 0
  du(3,:) = -1.6_dp*x(3,:)                               ! height to -60%: det F = -0.6
  step = 1
  cuts = 0
  do
    call elem_mix(q, s, x, step*du, mats(2), r, k, status)   ! Neo-Hooke, two-point
    if (status == ELEM_OK) exit
    step = step/2                                        ! ELEM_INVERTED: R = K = 0
    cuts = cuts + 1
  end do
  print '(a, i0, a, f6.3)', 'cuts: ', cuts, ', step: ', step   ! cuts: 1, step:  0.500
  print '(a, es10.3)', 'max |R|: ', maxval(abs(r))            ! max |R|:  3.374E+00
end program demo_inverted
