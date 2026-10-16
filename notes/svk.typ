#import "defs.typ": *

= Saint Venant-Kirchhoff

$
Psi(bold(E)) = lambda/2 (tr bold(E))^2 + mu tr(bold(E)^2)
$

== Reference configuration: $bold(C)$, $bold(S)$, $CC$

$
bold(S) = lambda (tr bold(E)) bold(I) + 2 mu bold(E), quad
CC = lambda bold(I) ot bold(I) + 2 mu II^s
$
$CC$ is constant.

== Current configuration: $bold(b)$, $bold(sigma)$, $cc$

With $bold(F) bold(E) bold(F)^T = 1/2 (bold(b)^2 - bold(b))$ and $tr bold(E) = 1/2 (tr bold(b) - 3)$:
$
bold(sigma) = J^(-1) [lambda (tr bold(E)) bold(b) + mu (bold(b)^2 - bold(b))], quad
cc = J^(-1) [lambda bold(b) ot bold(b) + mu (bold(b) ob bold(b) + bold(b) ub bold(b))]
$
$cc$ depends on $bold(b)$ and $J$: simple in the reference configuration, expensive in the current one.

== Two-point: $bold(F)$, $bold(P)$, $AA$

$
bold(P) = bold(F) bold(S), quad
AA = bold(I) ob bold(S) + bold(F) dot CC : 1/2 (bold(I) ub bold(F)^T + bold(F)^T ob bold(I))
  = bold(I) ob bold(S) + lambda bold(F) ot bold(F) + mu (bold(F) ub bold(F)^T + bold(b) ob bold(I))
$

== Pitfalls

- *Bounded compression.* $tr bold(E) = 1/2 (lambda_1^2 + lambda_2^2 + lambda_3^2 - 3) >= -3/2$ and
  $bold(E) >= -1/2 bold(I)$, so $Psi$ stays finite as $J -> 0^+$. A 1D bar with stretch
  $ell$ has $P = E_"mod"/2 thin ell (ell^2 - 1)$: compressive force peaks at $ell = 1 \/ sqrt(3)$ and
  returns to zero as $ell -> 0$ (limit point, loss of stability).
- *No volumetric-isochoric split.* Simple shear has $J = 1$ but $tr bold(E) = gamma^2 \/ 2 != 0$: the
  "volumetric" term reacts to shape change.
