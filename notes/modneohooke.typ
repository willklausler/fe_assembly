#import "defs.typ": *

= Modified Neo-Hooke (volumetric-isochoric split)

$
Psi = mu/2 (macron(I)_1 - 3) + U(J), quad macron(I)_1 = J^(-2/3) I_1, quad
U(J) = kappa/4 (J^2 - 2 ln J - 1)
$
$
p = U'(J) = kappa/2 (J - 1/J), quad
tilde(p) = p + J (dif p)/(dif J) = kappa J
$
$U(1) = U'(1) = 0$ and $U''(1) = kappa$, so $kappa$ is the small-strain bulk modulus.
$U -> infinity$ as $J -> 0^+$ and as $J -> infinity$. Near-incompressibility means $kappa >> mu$.

== Reference configuration: $bold(C)$, $bold(S)$, $CC$

$Dev(bold(bullet)) = (bold(bullet)) - 1/3 [(bold(bullet)) : bold(C)] bold(C)^(-1)$ is the deviator in the
reference configuration. With the fictitious stress $macron(bold(S)) = mu bold(I)$ (and
$macron(CC) = bold(0)$ for Neo-Hooke):
$
bold(S) = J p thin bold(C)^(-1) + underbrace(mu J^(-2/3) (bold(I) - I_1/3 bold(C)^(-1)), bold(S)_"iso")
$
$
CC = & J tilde(p) thin bold(C)^(-1) ot bold(C)^(-1) - 2 J p thin II_(bold(C)^(-1)) \
  & + 2/3 mu J^(-2/3) I_1 (II_(bold(C)^(-1)) - 1/3 bold(C)^(-1) ot bold(C)^(-1))
  - 2/3 (bold(C)^(-1) ot bold(S)_"iso" + bold(S)_"iso" ot bold(C)^(-1))
$
$bold(S)_"iso" : bold(C) = 0$, but $tr bold(S)_"iso" != 0$ in general: "deviatoric" here is not
"trace-free". Writing `S - trace(S)/3*I` in the reference configuration is a bug.

== Current configuration: $bold(b)$, $bold(sigma)$, $cc$

With $macron(bold(b)) = J^(-2/3) bold(b)$ and the ordinary $dev(bold(bullet)) = (bold(bullet)) - 1/3 tr(bold(bullet)) bold(I)$:
$
bold(sigma) = p bold(I) + J^(-1) bold(tau)_"iso", quad bold(tau)_"iso" = mu dev macron(bold(b))
$
$
cc = tilde(p) thin bold(I) ot bold(I) - 2 p II^s
  + J^(-1) [2/3 mu tr macron(bold(b)) (II^s - 1/3 bold(I) ot bold(I))
  - 2/3 (bold(I) ot bold(tau)_"iso" + bold(tau)_"iso" ot bold(I))]
$
$tr bold(tau)_"iso" = 0$, and the pressure $p$ comes from $U$ alone. No inverse is needed: the
cheapest configuration for this model.

== Two-point: $bold(F)$, $bold(P)$, $AA$

$
bold(P) = J p thin bold(F)^(-T) + mu J^(-2/3) (bold(F) - I_1/3 bold(F)^(-T))
$
$
AA = & J tilde(p) thin bold(F)^(-T) ot bold(F)^(-T) - J p thin bold(F)^(-T) ub bold(F)^(-1) \
  & + mu J^(-2/3) [bold(I) ob bold(I) + I_1/3 bold(F)^(-T) ub bold(F)^(-1)
  - 2/3 (bold(F)^(-T) ot bold(F) + bold(F) ot bold(F)^(-T)) + 2/9 I_1 bold(F)^(-T) ot bold(F)^(-T)]
$

== Pitfalls

- *Volumetric locking.* For $kappa >> mu$ the fully integrated 8-node hexahedron enforces
  $J approx 1$ at all eight points and locks. The stress and tangent are still correct; the
  element formulation (F-bar, Q1/P0 mixed) decides whether the results mean anything.
- Same $J > 0$ check as Neo-Hooke.
