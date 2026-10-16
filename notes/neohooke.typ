#import "defs.typ": *

= Neo-Hooke (coupled)

$
Psi = mu/2 (I_1 - 3) - mu ln J + lambda/2 (ln J)^2
$
The term $-mu ln J$ makes the reference configuration stress-free: without it $bold(S)(bold(I)) = mu bold(I)$.
It also couples shape and volume, so this form has no volumetric-isochoric split.

== Reference configuration: $bold(C)$, $bold(S)$, $CC$

$
bold(S) = mu (bold(I) - bold(C)^(-1)) + lambda ln J thin bold(C)^(-1), quad
CC = lambda bold(C)^(-1) ot bold(C)^(-1) + 2 (mu - lambda ln J) II_(bold(C)^(-1))
$
Needs $bold(C)^(-1)$ and a 36-term $II_(bold(C)^(-1))$.

== Current configuration: $bold(b)$, $bold(sigma)$, $cc$

The push-forward of $bold(C)^(-1)$ is $bold(I)$:
$
bold(sigma) = J^(-1) [mu (bold(b) - bold(I)) + lambda ln J thin bold(I)], quad
cc = J^(-1) [lambda bold(I) ot bold(I) + 2 (mu - lambda ln J) II^s]
$
$cc$ depends only on $J$: isotropic in form, with an effective shear modulus $mu - lambda ln J$.
Simple in the current configuration, the opposite of Saint Venant-Kirchhoff.

== Two-point: $bold(F)$, $bold(P)$, $AA$

With $partial bold(F)^(-T) \/ partial bold(F) = -bold(F)^(-T) ub bold(F)^(-1)$:
$
bold(P) = mu (bold(F) - bold(F)^(-T)) + lambda ln J thin bold(F)^(-T), quad
AA = mu bold(I) ob bold(I) + (mu - lambda ln J) bold(F)^(-T) ub bold(F)^(-1)
  + lambda bold(F)^(-T) ot bold(F)^(-T)
$

== Pitfalls

- $Psi -> infinity$ as $J -> 0^+$, as it should; $J <= 0$ (an inverted element) makes $ln J$ undefined,
  so the kernel must check $J > 0$ before evaluating.
- For $lambda >> mu$ (near-incompressible) a pure-displacement element locks; see
  modified Neo-Hooke.
