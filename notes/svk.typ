#set document(title: "Saint Venant-Kirchhoff", author: "Will Klausler")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(size: 10.5pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#let ob = math.overline(sym.times.o)
#let ub = math.underline(sym.times.o)
#let ot = sym.times.o
#let tr = math.op("tr")
#let CC = math.bb("C")
#let cc = math.bb("c")
#let AA = math.bb("A")
#let II = math.bb("I")

= Saint Venant-Kirchhoff

== Notation (Holzapfel)

Upper-case indices are material, lower-case spatial; repeated indices sum.
Fourth-order products are defined by index, so the symbols are unambiguous:
$
(bold(A) ot bold(B))_(i j k l) = A_(i j) B_(k l), quad
(bold(A) ob bold(B))_(i j k l) = A_(i k) B_(j l), quad
(bold(A) ub bold(B))_(i j k l) = A_(i l) B_(j k), quad
II^s = 1/2 (bold(I) ob bold(I) + bold(I) ub bold(I)).
$

Kinematics:
$bold(F) = partial bold(x) \/ partial bold(X)$, $J = det bold(F)$,
$bold(C) = bold(F)^T bold(F)$, $bold(b) = bold(F) bold(F)^T$,
$bold(E) = 1/2 (bold(C) - bold(I))$, and $tr bold(C) = tr bold(b)$.

== Strain energy

$
Psi(bold(E)) = lambda/2 (tr bold(E))^2 + mu tr(bold(E)^2)
$

== Reference configuration: $bold(C)$, $bold(S)$, $CC$

$
bold(S) = (partial Psi)/(partial bold(E)) = lambda (tr bold(E)) bold(I) + 2 mu bold(E), quad
CC = (partial bold(S))/(partial bold(E)) = lambda bold(I) ot bold(I) + 2 mu II^s
$
$CC$ is constant and has major and both minor symmetries (21 independent entries).

== Current configuration: $bold(b)$, $bold(sigma)$, $cc$

Push-forward, $sigma_(i j) = J^(-1) F_(i I) S_(I J) F_(j J)$, with
$bold(F) bold(E) bold(F)^T = 1/2 (bold(b)^2 - bold(b))$ and $tr bold(E) = 1/2 (tr bold(b) - 3)$:
$
bold(sigma) = J^(-1) [lambda (tr bold(E)) bold(b) + mu (bold(b)^2 - bold(b))]
$
$c_(i j k l) = J^(-1) F_(i I) F_(j J) F_(k K) F_(l L) CC_(I J K L)$:
$
cc = J^(-1) [lambda bold(b) ot bold(b) + mu (bold(b) ob bold(b) + bold(b) ub bold(b))]
$
$cc$ depends on $bold(b)$ and $J$. It is $J^(-1)$ times the tangent of the Truesdell rate of the
Kirchhoff stress $bold(tau) = J bold(sigma)$, *not* $partial bold(sigma) \/ partial bold(b)$; the
initial-stress term is added separately (@sec:element).

== Two-point: $bold(F)$, $bold(P)$, $AA$

$
bold(P) = bold(F) bold(S), quad
AA = (partial bold(P))/(partial bold(F))
  = bold(I) ob bold(S) + bold(F) dot CC : 1/2 (bold(I) ub bold(F)^T + bold(F)^T ob bold(I))
$
By the minor symmetry of $CC$, $partial E_(M N) \/ partial F_(k L)$ contracts to one term:
$
A_(i J k L) = delta_(i k) S_(J L) + F_(i I) F_(k N) CC_(I J L N)
$
For Saint Venant-Kirchhoff,
$
AA = bold(I) ob bold(S) + lambda bold(F) ot bold(F) + mu (bold(F) ub bold(F)^T + bold(b) ob bold(I))
$
$bold(I) ob bold(S)$ is the geometric stiffness. $AA$ has major symmetry
($A_(i J k L) = A_(k L i J)$, since $AA = partial^2 Psi \/ partial bold(F) partial bold(F)$);
minor symmetry is undefined for a two-point tensor (45 independent entries).

== Element residual and tangent <sec:element>

With $N_(a, J) = partial N_a \/ partial X_J$ and $N_(a, j) = partial N_a \/ partial x_j$:

#table(
  columns: 3,
  stroke: none,
  table.hline(),
  [], [$R_(a i)$], [$K_(a i b k)$],
  table.hline(),
  [Reference], [$integral_(Omega_0) F_(i I) S_(I J) N_(a, J) dif V$],
    [$integral_(Omega_0) N_(a, J) (F_(i I) F_(k N) CC_(I J L N) + delta_(i k) S_(J L)) N_(b, L) dif V$],
  [Current], [$integral_(Omega) sigma_(i j) N_(a, j) dif v$],
    [$integral_(Omega) N_(a, j) (c_(i j k l) + delta_(i k) sigma_(j l)) N_(b, l) dif v$],
  [Two-point], [$integral_(Omega_0) P_(i J) N_(a, J) dif V$],
    [$integral_(Omega_0) N_(a, J) A_(i J k L) N_(b, L) dif V$],
  table.hline(),
)

*Verification.* For a hyperelastic material all three rows give the same $R$ and the same
symmetric $K$, to round-off. Any difference is a bug in a stress or tangent.

== Pitfalls

- *Bounded compression.* $tr bold(E) = 1/2 (lambda_1^2 + lambda_2^2 + lambda_3^2 - 3) >= -3/2$ and
  $bold(E) >= -1/2 bold(I)$, so $Psi$ stays finite as $J -> 0^+$. A 1D bar with stretch
  $ell$ has $P = E_"mod"/2 thin ell (ell^2 - 1)$: compressive force peaks at $ell = 1 \/ sqrt(3)$ and
  returns to zero as $ell -> 0$ (limit point, loss of stability).
- *No volumetric-isochoric split.* Simple shear has $J = 1$ but $tr bold(E) = gamma^2 \/ 2 != 0$: the
  "volumetric" term reacts to shape change.
- *Derivatives with respect to symmetric tensors.* Automatic differentiation over the nine
  components of $bold(C)$ treats them as independent; symmetrize the result, or $CC$ loses minor
  symmetry and the shear terms are off by a factor of 2.
