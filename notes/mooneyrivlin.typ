#import "defs.typ": *

= Mooney-Rivlin (volumetric-isochoric split)

$
Psi = c_1 (macron(I)_1 - 3) + c_2 (macron(I)_2 - 3) + U(J), quad
macron(I)_1 = J^(-2/3) I_1, quad macron(I)_2 = J^(-4/3) I_2, quad
I_2 = 1/2 [(tr bold(C))^2 - tr(bold(C)^2)]
$
$U$, $p$ and $tilde(p)$ are those of modified Neo-Hooke. The small-strain shear modulus is
$mu = 2 (c_1 + c_2)$; $c_2 = 0$ gives modified Neo-Hooke with $mu = 2 c_1$, a built-in check.
With $partial I_2 \/ partial bold(C) = I_1 bold(I) - bold(C)$, the fictitious stress and tangent are
$
macron(bold(S)) = 2 (partial Psi_"iso")/(partial macron(bold(C)))
  = 2 c_1 bold(I) + 2 c_2 (macron(I)_1 bold(I) - macron(bold(C))), quad
macron(CC) = 2 (partial macron(bold(S)))/(partial macron(bold(C))) = 4 c_2 (bold(I) ot bold(I) - II^s)
$
$macron(CC) != bold(0)$ because $macron(I)_2$ is quadratic in $macron(bold(C))$; for Neo-Hooke $macron(I)_1$ is
linear and $macron(CC)$ vanishes.

== Reference configuration: $bold(C)$, $bold(S)$, $CC$

With the projections $PP = II^s - 1/3 bold(C)^(-1) ot bold(C)$ and
$tilde(PP) = II_(bold(C)^(-1)) - 1/3 bold(C)^(-1) ot bold(C)^(-1)$:
$
bold(S) = J p thin bold(C)^(-1) + underbrace(J^(-2/3) Dev macron(bold(S)), bold(S)_"iso"), quad
Dev macron(bold(S)) = macron(bold(S)) - 1/3 (macron(bold(S)) : bold(C)) bold(C)^(-1)
$
$
CC = & J tilde(p) thin bold(C)^(-1) ot bold(C)^(-1) - 2 J p thin II_(bold(C)^(-1))
  + J^(-4/3) PP : macron(CC) : PP^T \
  & + 2/3 J^(-2/3) (macron(bold(S)) : bold(C)) tilde(PP)
  - 2/3 (bold(C)^(-1) ot bold(S)_"iso" + bold(S)_"iso" ot bold(C)^(-1))
$

== Current configuration: $bold(b)$, $bold(sigma)$, $cc$

With $macron(bold(b)) = J^(-2/3) bold(b)$, the fictitious Kirchhoff stress
$macron(bold(tau)) = 2 c_1 macron(bold(b)) + 2 c_2 (macron(I)_1 macron(bold(b)) - macron(bold(b))^2)$,
its push-forward tangent $macron(cc) = 4 c_2 (macron(bold(b)) ot macron(bold(b)) - II_(macron(bold(b))))$ and
$PP^s = II^s - 1/3 bold(I) ot bold(I)$:
$
bold(sigma) = p bold(I) + J^(-1) bold(tau)_"iso", quad bold(tau)_"iso" = dev macron(bold(tau))
$
$
cc = & tilde(p) thin bold(I) ot bold(I) - 2 p II^s \
  & + J^(-1) [PP^s : macron(cc) : PP^s + 2/3 tr macron(bold(tau)) thin PP^s
  - 2/3 (bold(I) ot bold(tau)_"iso" + bold(tau)_"iso" ot bold(I))]
$

== Two-point: $bold(F)$, $bold(P)$, $AA$

With $partial I_2 \/ partial bold(F) = bold(G) = 2 (I_1 bold(F) - bold(b) bold(F))$:
$
bold(P) = J p thin bold(F)^(-T) + c_1 J^(-2/3) (2 bold(F) - 2/3 I_1 bold(F)^(-T))
  + c_2 J^(-4/3) (bold(G) - 4/3 I_2 bold(F)^(-T))
$
$
AA = J tilde(p) thin bold(F)^(-T) ot bold(F)^(-T) - J p thin bold(F)^(-T) ub bold(F)^(-1)
  + c_1 AA_1 + c_2 AA_2
$
$
AA_1 = J^(-2/3) [2 bold(I) ob bold(I) + 2/3 I_1 bold(F)^(-T) ub bold(F)^(-1)
  - 4/3 (bold(F)^(-T) ot bold(F) + bold(F) ot bold(F)^(-T)) + 4/9 I_1 bold(F)^(-T) ot bold(F)^(-T)]
$
$
AA_2 = J^(-4/3) [ & 2 I_1 bold(I) ob bold(I) + 4 bold(F) ot bold(F)
  - 2 (bold(I) ob bold(C) + bold(F) ub bold(F)^T + bold(b) ob bold(I)) \
  & - 4/3 (bold(F)^(-T) ot bold(G) + bold(G) ot bold(F)^(-T))
  + 4/3 I_2 bold(F)^(-T) ub bold(F)^(-1) + 16/9 I_2 bold(F)^(-T) ot bold(F)^(-T)]
$
The longest closed form of the four models; the general relation @eq:A gives the same $AA$ from
$bold(S)$ and $CC$, which the benchmark can time against it.

== Pitfalls

- *Polyconvexity.* $c_1, c_2 >= 0$ makes the coupled form in $I_1, I_2$ polyconvex, but not this
  decoupled one: $macron(I)_2 = J^(-4/3) I_2$ is not polyconvex (Hartmann and Neff, Int. J. Solids
  Struct. 40, 2003), who use $macron(I)_2^(3\/2)$ instead. $c_1 + c_2 > 0$ only gives a positive shear
  modulus. Irrelevant for timing; it matters for existence of solutions at large strain.
- Same volumetric locking and $J > 0$ check as modified Neo-Hooke.
