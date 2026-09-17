# Fourier coefficient information for linear inverse imaging

This document derives the Fourier information calculation used by
`bhuq.uq.fourier`. The calculation asks how strongly a linear measurement
system responds to additive image content at each spatial frequency. It does
not estimate uncertainty in neural network parameters.

## 1. Linear image measurements

The geometry and discretization that produce the VLBI measurement matrix are
derived in
[`vlbi_fourier_imaging_foundations.md`](vlbi_fourier_imaging_foundations.md).
This document begins from the resulting linear model

$$
y=Ax+\epsilon,
$$

where

$$
x\in\mathbb R^N,
\qquad
A\in\mathbb C^{M\times N},
\qquad
y\in\mathbb C^M.
$$

Here $N=HW$ is the number of image pixels and $M$ is the number of
measurements.

## 2. Additive Fourier corrections

Suppose an existing method has produced a reference image

$$
x_{\mathrm{ref}}.
$$

Introduce Fourier coefficients

$$
c\in\mathbb C^N
$$

and define an additive image correction by

$$
\delta x=F^{-1}c.
$$

The corrected image is

$$
x(c)=x_{\mathrm{ref}}+F^{-1}c.
$$

This construction changes image intensity. It is not a deformation of image
coordinates. A unit change in coefficient $c_k$ adds the Fourier basis image

$$
F^{-1}e_k,
$$

where $e_k$ is one at mode $k$ and zero elsewhere.

Applying the forward operator gives

$$
\begin{aligned}
\widehat y(c)
&=A x(c)
\\
&=A x_{\mathrm{ref}}+A F^{-1}c.
\end{aligned}
$$

Define

$$
B=AF^{-1}.
$$

Then

$$
\widehat y(c)=A x_{\mathrm{ref}}+Bc.
$$

Column $b_k$ of $B$ is the measurement change caused by one unit of
Fourier mode $k$.

## 3. Remove the fixed reference prediction

Define the data residual relative to the reference image:

$$
d=y-Ax_{\mathrm{ref}}.
$$

The inference problem for the Fourier correction is

$$
d=Bc+\epsilon.
$$

The reference image has disappeared from the derivative with respect to
$c$. This explains why Fourier information depends on the forward operator
and measurement noise, but not on the neural network that produced
$x_{\mathrm{ref}}$.

The observed value $d$ affects the posterior mean of $c$. It does not
affect posterior covariance in this linear Gaussian model.

## 4. Measurement noise

Assume independent Gaussian noise with standard deviations
$\sigma_1,\ldots,\sigma_M$:

$$
\epsilon\sim\mathcal N(0,C_n),
$$

where

$$
C_n
=
\operatorname{diag}
\left(
\sigma_1^2,\ldots,\sigma_M^2
\right).
$$

The noise precision is

$$
\Lambda_n
=
C_n^{-1}
=
\operatorname{diag}
\left(
\sigma_1^{-2},\ldots,\sigma_M^{-2}
\right).
$$

Noise precision and prior precision are different objects. Noise precision
describes measurement reliability and lives in measurement space. Prior
precision describes assumptions about $c$ before seeing the measurements
and lives in Fourier coefficient space.

For complex measurements, the implementation treats real and imaginary
components as independent real observations with the supplied per-component
standard deviation. Equivalently, one may stack them into a real vector. If

$$
B=B_{\mathrm R}+iB_{\mathrm I},
\qquad
c=c_{\mathrm R}+ic_{\mathrm I},
$$

then

$$
\begin{bmatrix}
\operatorname{Re}(Bc)\\
\operatorname{Im}(Bc)
\end{bmatrix}
=
\begin{bmatrix}
B_{\mathrm R} & -B_{\mathrm I}\\
B_{\mathrm I} & B_{\mathrm R}
\end{bmatrix}
\begin{bmatrix}
c_{\mathrm R}\\
c_{\mathrm I}
\end{bmatrix}.
$$

The real derivation below applies directly to this stacked system. The compact
complex notation replaces transposes by conjugate transposes.

## 5. Gaussian likelihood

For real notation, the likelihood is

$$
p(d\mid c)
=
\mathcal N(Bc,C_n).
$$

Ignoring constants that do not depend on $c$, the negative log-likelihood is

$$
\mathcal L(c)
=
\frac{1}{2}
(d-Bc)^{\mathsf T}
\Lambda_n
(d-Bc).
$$

Let

$$
r(c)=d-Bc.
$$

Then

$$
\mathcal L(c)
=
\frac{1}{2}
r(c)^{\mathsf T}\Lambda_n r(c).
$$

## 6. Gradient from differentials

A perturbation $dc$ changes the residual by

$$
dr=-B\,dc.
$$

Differentiate the quadratic:

$$
\begin{aligned}
d\mathcal L
&=
\frac{1}{2}
\left[
(dr)^{\mathsf T}\Lambda_n r
+
r^{\mathsf T}\Lambda_n dr
\right].
\end{aligned}
$$

The noise precision is symmetric, so the two scalar terms are equal:

$$
(dr)^{\mathsf T}\Lambda_n r
=
r^{\mathsf T}\Lambda_n dr.
$$

Therefore

$$
d\mathcal L
=
r^{\mathsf T}\Lambda_n dr.
$$

Substitute $dr=-B\,dc$:

$$
\begin{aligned}
d\mathcal L
&=
-r^{\mathsf T}\Lambda_n B\,dc
\\
&=
\left(
-B^{\mathsf T}\Lambda_n r
\right)^{\mathsf T}
dc.
\end{aligned}
$$

By the definition

$$
d\mathcal L
=
(\nabla_c\mathcal L)^{\mathsf T}dc,
$$

the gradient is

$$
\nabla_c\mathcal L
=
-B^{\mathsf T}\Lambda_n r.
$$

Since $r=d-Bc$,

$$
\boxed{
\nabla_c\mathcal L
=
B^{\mathsf T}\Lambda_n(Bc-d)
}.
$$

## 7. Hessian and Fisher information

Differentiate the gradient:

$$
d(\nabla_c\mathcal L)
=
B^{\mathsf T}\Lambda_n B\,dc.
$$

The likelihood Hessian is

$$
\boxed{
H_{\mathrm{data}}
=
B^{\mathsf T}\Lambda_n B
}.
$$

In complex notation,

$$
\boxed{
H_{\mathrm{data}}
=
B^{\mathsf H}\Lambda_n B
}.
$$

This Hessian is exact. The prediction $Bc$ is linear in $c$, so no
residual-weighted model second derivative appears.

The same matrix follows from Fisher information. The score is

$$
s(c)
=
B^{\mathsf T}\Lambda_n(d-Bc).
$$

Under the likelihood, $d-Bc=\epsilon$. Its score covariance is

$$
\begin{aligned}
\mathbb E[s(c)s(c)^{\mathsf T}]
&=
B^{\mathsf T}
\Lambda_n
\mathbb E[\epsilon\epsilon^{\mathsf T}]
\Lambda_n
B
\\
&=
B^{\mathsf T}
\Lambda_n
C_n
\Lambda_n
B.
\end{aligned}
$$

Because $\Lambda_n=C_n^{-1}$,

$$
\Lambda_n C_n\Lambda_n=\Lambda_n.
$$

Hence

$$
\boxed{
F_c
=
B^{\mathsf T}\Lambda_n B
=
H_{\mathrm{data}}
}.
$$

The Hessian, Gauss Newton matrix, and Fisher information coincide exactly for
this linear Gaussian model.

## 8. Precision pullback and covariance pushforward

The order $B^{\mathsf H}\Lambda_n B$ follows from pulling measurement
precision back to coefficient space. A coefficient perturbation produces

$$
\delta y=B\,\delta c.
$$

Its noise-weighted measurement magnitude is

$$
\begin{aligned}
\delta y^{\mathsf H}\Lambda_n\delta y
&=
(B\delta c)^{\mathsf H}
\Lambda_n
(B\delta c)
\\
&=
\delta c^{\mathsf H}
B^{\mathsf H}\Lambda_n B
\delta c.
\end{aligned}
$$

The matrix between the two copies of $\delta c$ is coefficient-space
precision.

The familiar covariance formula answers the reverse question. If

$$
\operatorname{Cov}(c)=C_c,
$$

then

$$
\operatorname{Cov}(Bc)
=
B C_c B^{\mathsf H}.
$$

Thus

$$
\text{precision pullback}
=
B^{\mathsf H}\Lambda_n B,
$$

while

$$
\text{covariance pushforward}
=
B C_c B^{\mathsf H}.
$$

## 9. Fourier prior

Let the prior over Fourier corrections be

$$
c\sim\mathcal N(0,C_0).
$$

Its precision is

$$
\Lambda_0=C_0^{-1}.
$$

The negative log-prior is

$$
-\log p(c)
=
\frac{1}{2}
c^{\mathsf T}\Lambda_0c
+
\text{constant}.
$$

An isotropic prior uses

$$
\Lambda_0=\alpha I.
$$

A frequency-dependent diagonal prior uses

$$
\Lambda_0
=
\operatorname{diag}
\left(
\alpha_1,\ldots,\alpha_N
\right).
$$

For example, a smooth-image prior could assign larger precision to
high-frequency coefficients. This expresses the belief that large
high-frequency corrections are less plausible before seeing the data.

The measurement noise values $\sigma_m$ do not define this prior. Noise
belongs to $\Lambda_n$; the Fourier prior is a separate modeling choice.

## 10. Posterior by completing the square

The negative log-posterior is

$$
\begin{aligned}
h(c)
&=
\frac{1}{2}
(d-Bc)^{\mathsf T}
\Lambda_n
(d-Bc)
+
\frac{1}{2}
c^{\mathsf T}\Lambda_0c.
\end{aligned}
$$

Expand only the terms that depend on $c$:

$$
h(c)
=
\frac{1}{2}
c^{\mathsf T}
\left(
B^{\mathsf T}\Lambda_n B+\Lambda_0
\right)c
-
c^{\mathsf T}B^{\mathsf T}\Lambda_n d
+
\text{constant}.
$$

Define

$$
Q
=
B^{\mathsf T}\Lambda_n B+\Lambda_0
$$

and

$$
q
=
B^{\mathsf T}\Lambda_n d.
$$

Then

$$
h(c)
=
\frac{1}{2}c^{\mathsf T}Qc-c^{\mathsf T}q+\text{constant}.
$$

Let

$$
\mu_c=Q^{-1}q.
$$

Since $Q\mu_c=q$, expanding

$$
\frac{1}{2}
(c-\mu_c)^{\mathsf T}
Q
(c-\mu_c)
$$

gives

$$
\frac{1}{2}c^{\mathsf T}Qc-c^{\mathsf T}q
+
\frac{1}{2}\mu_c^{\mathsf T}Q\mu_c.
$$

The final term does not depend on $c$, so the posterior has the Gaussian
form

$$
p(c\mid d)
\propto
\exp
\left[
-\frac{1}{2}
(c-\mu_c)^{\mathsf T}
Q
(c-\mu_c)
\right].
$$

Therefore,

$$
\boxed{
\Lambda_{\mathrm{post}}
=
Q
=
B^{\mathsf T}\Lambda_n B+\Lambda_0
},
$$

$$
\boxed{
C_{\mathrm{post}}
=
Q^{-1}
},
$$

and

$$
\boxed{
\mu_c
=
Q^{-1}B^{\mathsf T}\Lambda_n d
}.
$$

The module does not compute $\mu_c$. It studies how much each coefficient
can vary.

## 11. Information for one mode

Let $b_k$ be column $k$ of $B$. The diagonal Fisher entry is

$$
\begin{aligned}
I_k
&=
[B^{\mathsf H}\Lambda_nB]_{k,k}
\\
&=
b_k^{\mathsf H}\Lambda_n b_k
\\
&=
\sum_{m=1}^{M}
\frac{|B_{m,k}|^2}{\sigma_m^2}.
\end{aligned}
$$

This quantity has a direct experimental interpretation. Add one unit of mode
$k$, calculate the resulting measurement changes, divide each change by its
noise standard deviation, square the magnitudes, and add them. Large values
mean that the mode is easy to detect. Small values mean that the mode can move
farther before the change rises above measurement noise.

The image change itself is fixed by the chosen basis vector
$F^{-1}e_k$. The information calculation measures how visible that known
image change is to the instrument.

## 12. Diagonal posterior approximation

The full marginal variance of coefficient $k$ is

$$
[C_{\mathrm{post}}]_{k,k}
=
\left[
\left(
H_{\mathrm{data}}+\Lambda_0
\right)^{-1}
\right]_{k,k}.
$$

The implementation instead computes

$$
\widetilde v_k
=
\frac{1}{
[H_{\mathrm{data}}]_{k,k}+\alpha
}.
$$

These expressions are equal when the posterior precision is diagonal. They
are not equal in general because matrix inversion mixes diagonal and
off-diagonal entries.

The function computes the exact Fisher diagonal for any dense linear
operator. Calling its reciprocal a marginal posterior variance adds the
assumption that coupling between Fourier modes can be neglected.

## 13. Gridded Fourier sampling

For measurements that lie exactly on the discrete Fourier grid,

$$
A=SF.
$$

Then

$$
B
=
AF^{-1}
=
SFF^{-1}
=
S.
$$

The Fisher matrix becomes

$$
H_{\mathrm{data}}
=
S^{\mathsf H}\Lambda_n S.
$$

Because $S$ selects individual Fourier cells, this matrix is diagonal.
Repeated measurements in one cell add their inverse noise variances:

$$
I_k
=
\sum_{\{m:\,m\text{ samples }k\}}
\frac{1}{\sigma_m^2}.
$$

This is the naturally weighted sampling function. Empty cells have zero data
information.

Off-grid VLBI measurements spread sensitivity across nearby discrete modes.
The exact Fisher diagonal still measures per-mode sensitivity, but the full
matrix can contain off-diagonal coupling.

## 14. Real image symmetry

For a real image,

$$
c(-k)=\overline{c(k)}.
$$

Positive and negative frequency coefficients are therefore linked. The
implementation maps every array index to its negative-frequency partner and
averages their information values. This produces a symmetric information map.

A complete posterior over a real image would parameterize independent cosine
and sine amplitudes or explicitly enforce the Hermitian constraint. Pairwise
averaging is sufficient for the current data-blindness diagnostic, but it is
not a substitute for that full constrained posterior.

## 15. Prior scale and calibration

The current module chooses

$$
\alpha
=
\rho\max_k I_k,
$$

where $\rho$ is `prior_precision_fraction`.

This is a relative numerical regularizer. It is not a calibrated prior derived
from an image population or physical source model. It sets the finite ceiling
on variance in modes with little or no data information.

The data-blind mask uses

$$
I_k<\beta\max_j I_j,
$$

where $\beta$ is `blind_information_fraction`. This mask depends only on the
forward operator and measurement noise. It does not depend on the Fourier
prior.

A calibrated Fourier posterior would require a justified $C_0$, such as a
power spectrum estimated from an image population, and the full coupled
precision when off-diagonal terms are not negligible.

## 16. Image-space uncertainty

If the full Fourier posterior covariance were available, its image-space
covariance would be

$$
C_x
=
F^{-1}
C_{\mathrm{post}}
F^{-\mathsf H}.
$$

The current module does not perform this propagation. It reports uncertainty
per Fourier coefficient because its intended output is a map of frequency
coverage.

With a diagonal Fourier covariance and equal-magnitude Fourier basis
functions, the pixelwise diagonal alone can become spatially uniform. The
off-diagonal Fourier covariance carries information needed for more detailed
spatial structure.

## 17. Numerical calculation

The implementation follows these steps:

1. Read the dense image-to-measurement matrix $A$.
2. Apply the inverse two-dimensional DFT along each measurement row to obtain
   $B=AF^{-1}$.
3. Weight row $m$ by $\sigma_m^{-2}$.
4. Sum $|B_{m,k}|^2/\sigma_m^2$ over measurements to obtain $I_k$.
5. Average negative-frequency partners for a real image.
6. Add the isotropic prior precision $\alpha$.
7. Compute the diagonal approximation $1/(I_k+\alpha)$.
8. Mark modes below the relative information threshold as data blind.

The information calculation is exact for the Fisher diagonal. The posterior
variance is exact only when the complete Fourier precision is diagonal and the
prior correctly describes the coefficient distribution.
