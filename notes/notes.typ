#import "defs.typ": *
#set document(title: "Hyperelastic stresses and tangents", author: "Will Klausler")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(size: 10.5pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center, text(16pt, weight: "bold")[Hyperelastic stresses and tangents])

= General relations

== Notation (Holzapfel)

Upper-case indices are material, lower-case spatial; repeated indices sum.
Fourth-order products are defined by index, so the symbols are unambiguous:
$
(bold(A) ot bold(B))_(i j k l) = A_(i j) B_(k l), quad
(bold(A) ob bold(B))_(i j k l) = A_(i k) B_(j l), quad
(bold(A) ub bold(B))_(i j k l) = A_(i l) B_(j k), quad
II_(bold(A)) = 1/2 (bold(A) ob bold(A) + bold(A) ub bold(A)),
$
with $II^s = II_(bold(I))$ and $partial bold(C)^(-1) \/ partial bold(C) = -II_(bold(C)^(-1))$.

Kinematics:
$bold(F) = partial bold(x) \/ partial bold(X)$, $J = det bold(F)$,
$bold(C) = bold(F)^T bold(F)$, $bold(b) = bold(F) bold(F)^T$,
$bold(E) = 1/2 (bold(C) - bold(I))$, $I_1 = tr bold(C) = tr bold(b)$,
$partial J \/ partial bold(C) = J/2 bold(C)^(-1)$, $partial J \/ partial bold(F) = J bold(F)^(-T)$.

== Stresses and tangents in the three configurations

$
bold(S) = 2 (partial Psi)/(partial bold(C)), quad
CC = 2 (partial bold(S))/(partial bold(C)), quad
bold(sigma) = J^(-1) bold(F) bold(S) bold(F)^T, quad
c_(i j k l) = J^(-1) F_(i I) F_(j J) F_(k K) F_(l L) CC_(I J K L),
$
$
bold(P) = bold(F) bold(S), quad
A_(i J k L) = (partial P_(i J))/(partial F_(k L)) = delta_(i k) S_(J L) + F_(i I) F_(k N) CC_(I J L N).
$ <eq:A>

- $CC$ has major and both minor symmetries (21 independent entries).
- $cc$ is $J^(-1)$ times the tangent of the Truesdell rate of the Kirchhoff stress
  $bold(tau) = J bold(sigma)$, *not* $partial bold(sigma) \/ partial bold(b)$; the initial-stress term
  is added separately in the element tangent.
- $AA = partial^2 Psi \/ partial bold(F) partial bold(F)$ has major symmetry
  ($A_(i J k L) = A_(k L i J)$); minor symmetry is undefined for a two-point tensor
  (45 independent entries). Its first term, $bold(I) ob bold(S)$, is the geometric stiffness.

== Element residual and tangent

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

*Derivatives with respect to symmetric tensors.* Automatic differentiation over the nine
components of $bold(C)$ treats them as independent; symmetrize the result, or $CC$ loses minor
symmetry and the shear terms are off by a factor of 2.

*Matching parameters.* The models below linearize to Lamé constants $(lambda, mu)$ or to
$(kappa, mu)$; compare them with $kappa = lambda + 2 mu \/ 3$.

#include "svk.typ"
#include "neohooke.typ"
#include "modneohooke.typ"
