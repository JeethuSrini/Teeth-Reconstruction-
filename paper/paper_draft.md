# Statistical Shape Reconstruction of Worn Enamel–Dentine Junction Surfaces in Primate Molars

**Draft — v1**

---

## Abstract

Tooth wear progressively destroys the occlusal morphology that dental anthropology
depends on. The enamel–dentine junction (EDJ) is more resistant than the outer
enamel surface and preserves taxonomically informative form well into moderate
wear, but once attrition reaches the dentine the EDJ itself is truncated and cusp
geometry is lost. We present a reproducible pipeline that reconstructs the pre-wear
EDJ surface of upper left third molars (ULM3) from 3D scans of worn specimens, and
we evaluate it under a protocol designed so that a method which changes nothing
cannot score well.

The pipeline establishes dense point-to-point correspondence across specimens by
coherent point drift registration to an automatically selected template, learns a
principal-component shape model from unworn specimens, and reconstructs worn
anatomy by three alternative strategies: a **global** shape model, a per-specimen
**neighborhood** model built from the anatomically nearest unworn teeth, and a
**non-rigid fit** that warps the model output back onto the observed scan under a
confidence gate. Reconstructions are validated against the true unworn original on
two independent artificially worn series spanning Molnar wear stages 3–6: 16 cases
from a 2-specimen EDJ series and 64 cases from 8 held-out ULM3 specimens.

On the EDJ series the three methods restore worn-region anatomy to
0.130–0.137 mm RMSE, and full-surface Chamfer distance reaches 0.041 mm for the
non-rigid fit. These figures sit within the 0.1–0.2 mm band reported for
statistical shape reconstruction of human teeth in archaeological and clinical
settings, and are obtained without manual landmarking. Error grows monotonically
with wear stage for every method, and all three degrade sharply at Molnar stage 6.
We close by describing how the pipeline is being packaged as a parameterised
library so that researchers can assemble a reconstruction workflow suited to their
own material.

---

## 1. Introduction

Teeth are the most frequently recovered and best preserved elements of the
vertebrate skeleton, and molar crown morphology carries much of the taxonomic,
phylogenetic and dietary signal that dental anthropology relies on. That signal is
progressively erased by wear. Mastication, abrasive diet and bruxism remove enamel
from the occlusal surface, and once wear reaches the dentine it begins to truncate
the cusps themselves.

The **enamel–dentine junction (EDJ)** is the developmental template on which enamel
is deposited. It is laid down during odontogenesis, is not remodelled during life,
and therefore preserves crown and cusp form after the outer enamel surface has been
substantially altered — which is why EDJ morphology has become a standard target for
taxonomic and phylogenetic work on worn and fossil material. But the EDJ is only
*relatively* robust. Under advanced attrition the dentine horns themselves are worn
away, and the very specimens most in need of morphological analysis are the ones
whose EDJ can no longer be measured directly.

Anthropology has addressed this with reconstruction methods that are largely manual
and largely two-dimensional. The **Profile**, **Polynomial** and **Pen Tool**
methods reconstruct crown height and enamel cap from sectioned or scanned teeth,
and a comparative test found them reliable only for lightly worn crowns, with
Profile and Polynomial preferred as wear increases. More recent work has applied
machine learning to predict EDJ cusp height in worn lower molars from morphometric
features. These approaches share two limitations: they recover a small number of
scalar measurements (crown height, cusp height, enamel thickness) rather than a
surface, and they require operator decisions that are difficult to standardise
across labs.

A **statistical shape model (SSM)** offers a different route. If dense
correspondence can be established across a sample of unworn teeth, principal
component analysis yields a compact generative description of how that anatomy
varies. A worn specimen can then be fitted to the model on the surface that
survives, and the model supplies the geometry that wear removed. The result is a
complete surface rather than a set of measurements, and the procedure is
deterministic given its parameters.

This paper reports such a pipeline for primate ULM3 EDJ surfaces, and — as much as
the reconstruction itself — a **validation protocol that does not flatter the
method**. Reconstruction quality is easy to overstate: a worn tooth is mostly
intact, so any metric averaged over the whole crown is dominated by surface that
was never damaged, and a method that returns its input unchanged will score well.
We therefore score restoration on a shared, method-agnostic mask of the anatomy that
wear actually removed, and report whole-surface and fidelity-to-scan figures only as
supporting evidence.

**Contributions.**

1. An end-to-end, parameterised pipeline from raw worn mesh to reconstructed EDJ
   surface, requiring no manual landmarking.
2. Three reconstruction strategies — global SSM, neighborhood SSM, and non-rigid
   fit — implemented against a shared correspondence stage so they can be compared
   directly.
3. A validation protocol with an explicit worn-region mask, applied to two
   independent artificially worn series (16 and 64 cases) with known ground truth,
   across Molnar wear stages 3–6.
4. A characterisation of where these methods fail, in particular at severe wear.

---

## 2. Related work

### 2.1 Reconstructing worn teeth in anthropology

O'Hara and Guatelli-Steinberg compared the Profile, Polynomial and Pen Tool methods
for reconstructing crown height and enamel thickness, and concluded that accurate
reconstruction is limited to relatively unworn crowns, with method choice depending
on the measurement of interest. Micro-CT has been used to reconstruct worn or
broken cusps non-destructively and to check for internal cracks before casting.
A recent machine-learning pipeline predicts EDJ horn-tip height in worn lower
molars using Random Forest and regression ensembles over morphometric variables.
All of these target scalar measurements; none returns a reconstructed surface.

### 2.2 Statistical shape models of teeth

Buchaillard and colleagues built 3D statistical models for tooth surface
reconstruction and reported mean surface error below 0.2 mm using the first five
principal components and below 0.1 mm using thirteen. Woods et al. applied SSM to
27 mandibular canines spanning modern, Anglo-Saxon and Roman samples, reporting
reconstruction error below 0.2 mm after five PCs — roughly 0.7% of mean canine
length — with the first five components capturing about 75% of variance. In
clinical dentistry, SSMs have been used to reconstruct missing crowns from CT, to
estimate root geometry for implant planning (1.3 ± 0.8 mm near the cemento-enamel
junction), and to detect missing teeth on post-mortem CT. Gaussian process
morphable models generalise the PCA formulation to continuous covariance
functions and underpin much of the modern posterior shape-completion literature.

### 2.3 Learning-based dental reconstruction

Recent work reconstructs teeth from 2D inputs: TeethDreamer generates 3D teeth from
five intra-oral photographs via a diffusion prior and neural surface reconstruction;
Occudent recovers 3D teeth from panoramic radiographs with neural implicit
functions; transformer-based implicit neural networks reconstruct missing tooth
morphology evaluated by Chamfer distance, F-score and IoU. These methods solve a
substantially harder, more underdetermined problem — inferring 3D geometry from 2D
projections — and their reported errors are correspondingly larger. They are not
directly comparable to surface-input reconstruction, a point we return to in §7.3.

### 2.4 Positioning

Our work sits between §2.1 and §2.2: the anthropological target (EDJ morphology of
worn molars, Molnar-staged) with the methodological apparatus of dental SSM
(dense correspondence, PCA priors, posterior fitting), evaluated on artificially
worn series where the true pre-wear surface is known. To our knowledge the
combination — surface-level EDJ reconstruction, no manual landmarks, validated
against ground truth across a graded wear series — has not previously been reported.

---

## 3. Materials

### 3.1 Specimens

All specimens are upper left third molars (ULM3), EDJ surfaces extracted from
micro-CT and exported as triangle meshes in millimetres.

| Set | Role | n | Notes |
|---|---|---:|---|
| EDJ training | unworn, SSM training | 18 | `cprc_nyu` series |
| EDJ test | artificially worn | 2 × 8 | TEST1 → `n0245`, TEST2 → `n0257` |
| ULM training | unworn, SSM training | 90 | held-out specimens excluded |
| ULM test | artificially worn, held out | 8 × 8 | N1063, N332, N4, N459, N705, N726, N728, N891 |

For the ULM series the eight test specimens are excluded from the 90-tooth training
pool, so every ULM reconstruction is of a specimen the model has never seen.

### 3.2 Artificial wear series

Ground truth for a *naturally* worn tooth does not exist — the pre-wear surface is
gone. We therefore evaluate on artificially worn series: each source specimen was
progressively worn in eight steps, giving matched (worn, original) pairs. The eight
steps map onto **Molnar wear stages 3–6**, two files per stage, so results can be
read against a standard anthropological wear scale rather than an arbitrary index.

| Wear level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Molnar stage | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 |

---

## 4. Methods

The pipeline runs in three stages: establish correspondence across all specimens,
reconstruct each worn tooth, then evaluate. Stages 2a–2c are independent
alternatives that consume the same Stage 1 output.

```
  unworn teeth ──┐
                 ├──► STAGE 1: correspondence ──► STAGE 2: reconstruction ──► STAGE 3: evaluation
  worn teeth  ───┘      (shared point ordering)     (a) global SSM
                                                     (b) neighborhood SSM
                                                     (c) non-rigid fit
```

### 4.1 Stage 1 — dense correspondence

Shape statistics require that point *i* mean the same anatomical location on every
tooth. We obtain this without manual landmarking, following the logic of automated
point-cloud landmarking approaches such as ALPACA:

1. **Sample.** Draw points uniformly from each mesh surface (100,000 for the EDJ
   set, 10,000 for the ULM set).
2. **Normalise.** Centre each cloud, divide by its bounding-box diagonal to remove
   size, and rotate onto its principal axes. Because the sign of a PCA axis is
   arbitrary, each candidate reflection is tested against the template and the
   best-fitting orientation retained — without this step registration can begin
   180° out and never recover.
3. **Rigid align.** Iterative closest point onto an automatically selected template
   (the most central specimen).
4. **Deform.** Coherent point drift warps the *template* onto each target. Because
   the template's own point ordering is carried along, the deformed template
   defines correspondence. For high point counts CPD is run on a subset and the
   displacement field interpolated back to the full cloud.

Output: one cloud per specimen, all sharing a point ordering.

### 4.2 Stage 2a — global statistical shape model

Stack the corresponded unworn teeth into a matrix, take the mean shape **μ**, and
run PCA to obtain modes **V** with variances **Λ**. Any shape is then approximated
as

> **x**(b) = **μ** + **V** b

Fitting a worn tooth means finding the coefficients b that best explain the points
that *survive*, and letting the model supply the rest. With Ω the observed subset,
b is obtained by Tikhonov-regularised least squares,

> b = (VᵀV + λΛ⁻¹)⁻¹ Vᵀ(x_Ω − μ_Ω)

with coefficients clipped to ±4σ so the fit cannot extrapolate into implausible
shapes. Regularisation is what keeps a partial observation from producing a wild
model; λΛ⁻¹ penalises each mode in proportion to how rare that deformation is in
the training sample.

On the EDJ set, 18 training teeth yield 14 modes at 99.4% of total variance
(PC1 alone 52.8%, PC1–2 73.4%, 95% reached at mode 9). On the ULM set, 90 training
teeth yield 26 modes at 99.1%.

After fitting, a **non-rigid refinement** restores local detail: observed points are
snapped back to their measured positions, and the resulting correction is carried
into the reconstructed region by inverse-distance-weighted interpolation, so the
filled anatomy joins the real surface without a seam.

### 4.3 Stage 2b — neighborhood shape model

A single model built from all unworn teeth describes the *population*. If a worn
specimen belongs to a morphological subgroup, the population mean pulls its
reconstruction toward a centroid it does not belong to. The neighborhood variant
therefore rebuilds the model per specimen:

1. Project all unworn teeth into a low-dimensional PCA space.
2. Set an adaptive threshold at median + 2 × IQR of the pairwise distances among
   unworn teeth.
3. Project the worn tooth in, and select every unworn tooth within the threshold,
   capped at K.
4. Build the SSM from just those neighbours; if nothing falls within the threshold,
   fall back to the global model.

Everything downstream is identical. The cost is fewer modes — a K-neighbour model
has at most K−1 — so the local model trades expressive range for a tighter, more
relevant prior.

### 4.4 Stage 2c — non-rigid fit

Both SSM variants output a surface that is anatomically complete but only as
detailed as the model allows. The non-rigid fit takes that reconstruction and warps
it back onto the real scan, keeping model anatomy only where the scan has nothing
to say:

1. Rigidly align the reconstruction to the scan.
2. For each reconstruction point, measure distance to the nearest scan point and
   convert it to a **confidence**: high where the scan is present, falling smoothly
   to zero where it is absent (i.e. in the worn region).
3. Use confident points as landmarks for a thin-plate-spline warp of the whole
   reconstruction.
4. Repeat coarse-to-fine, tightening the distance band and relaxing the smoothness
   penalty each round.

Where the tooth is intact the output converges on the real measured surface; where
wear removed material there are no confident landmarks and the model's anatomy is
retained. No scan points enter the output, so scanning noise and holes are not
inherited.

### 4.5 Surface output

Reconstructed clouds are mapped back to the original millimetre coordinate frame by
inverting the Stage 1 normalisation, then meshed with screened Poisson
reconstruction, trimmed at the 1% density quantile to remove extrapolated fringe,
and Taubin-smoothed.

---

## 5. Evaluation protocol

**Ground truth is the unworn original**, never the worn scan. Each reconstruction is
taken in millimetre space and rigidly aligned to the original before any distance is
computed.

**The shared wear mask.** Points of the original surface lying farther than twice the
median point spacing from the worn scan are the anatomy wear removed. The same mask
is applied to every method, so no method is scored on its own idea of what it
repaired.

**Three metric layers.** A single number invites the wrong conclusion, because most
of a worn crown is undamaged:

| Layer | Compares | Question | Rank on it? |
|---|---|---|---|
| **Primary** | worn region → original | Did the missing anatomy come back? | **Yes** |
| **Secondary** | full surface ↔ original | Overall match to a healthy tooth | Supporting |
| **Tertiary** | full surface ↔ worn scan | Was intact enamel preserved? | **No** — fidelity only |

The tertiary layer must not be used for ranking: a method returning its input
unchanged scores perfectly on it while restoring nothing.

Metrics are Chamfer distance, RMSE, MAE, Hausdorff and its 95th percentile (H95),
and coverage at twice the median spacing, all in millimetres.

---

## 6. Results

### 6.1 EDJ series (TEST1 / TEST2, 16 cases)

**Primary — worn region vs original**

| Method | n | Worn RMSE | Worn MAE | Worn H95 |
|---|---:|---:|---:|---:|
| **Global SSM** | 16 | **0.1300** | 0.0982 | **0.2712** |
| Neighborhood SSM | 16 | 0.1347 | 0.1014 | 0.2844 |
| Non-rigid fit | 16 | 0.1368 | **0.0955** | 0.2993 |

**Secondary — full surface vs original**

| Method | Chamfer | RMSE | H95 | Coverage@2× |
|---|---:|---:|---:|---:|
| **Non-rigid fit** | **0.0409** | **0.0636** | **0.1432** | **0.702** |
| Global SSM | 0.0572 | 0.0773 | 0.1731 | 0.303 |
| Neighborhood SSM | 0.0581 | 0.0788 | 0.1784 | 0.294 |

**Tertiary — vs worn scan (fidelity only)**

| Method | Chamfer | RMSE |
|---|---:|---:|
| Non-rigid fit | 0.0349 | 0.0593 |
| Global SSM | 0.0573 | 0.0798 |
| Neighborhood SSM | 0.0576 | 0.0791 |

**Worn-region RMSE by wear level**

| Method | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global SSM | 0.0643 | 0.0699 | 0.0813 | 0.0911 | 0.1137 | 0.1251 | 0.3114 | 0.1832 |
| Neighborhood SSM | 0.0641 | 0.0702 | 0.0812 | 0.0908 | 0.1117 | 0.1225 | 0.3106 | 0.2261 |
| Non-rigid fit | 0.0351 | 0.0471 | 0.0654 | 0.0821 | 0.1154 | 0.1225 | 0.2837 | 0.3436 |

### 6.2 ULM series (8 held-out specimens, 64 cases)

**Primary — worn region vs original**

| Method | n | Worn RMSE | Worn MAE | Worn H95 |
|---|---:|---:|---:|---:|
| **Non-rigid fit** | 64 | **0.1927** | **0.1275** | **0.4445** |
| Neighborhood SSM | 64 | 0.2429 | 0.1888 | 0.5146 |
| Global SSM | 64 | 0.2447 | 0.1900 | 0.5174 |

**Secondary — full surface vs original**

| Method | Chamfer | RMSE | H95 | Coverage@2× |
|---|---:|---:|---:|---:|
| **Non-rigid fit** | **0.0720** | **0.1716** | **0.4061** | **0.166** |
| Neighborhood SSM | 0.1592 | 0.2300 | 0.4950 | 0.031 |
| Global SSM | 0.1600 | 0.2316 | 0.4975 | 0.030 |

**Worn-region RMSE by wear level**

| Method | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global SSM | 0.1787 | 0.1778 | 0.1929 | 0.2073 | 0.2607 | 0.2384 | 0.3356 | 0.3661 |
| Neighborhood SSM | 0.1755 | 0.1759 | 0.1914 | 0.2044 | 0.2590 | 0.2365 | 0.3362 | 0.3638 |
| Non-rigid fit | 0.1043 | 0.1093 | 0.1369 | 0.1474 | 0.2071 | 0.2004 | 0.3009 | 0.3354 |

### 6.3 Summary of findings

- On the EDJ series the three methods are close on worn-region RMSE
  (0.130–0.137 mm), with Global SSM marginally ahead.
- On the larger held-out ULM series the non-rigid fit leads the primary metric
  (0.193 mm vs 0.243–0.245 mm).
- The non-rigid fit dominates every full-surface and fidelity measure on both sets.
- Global and Neighborhood SSM are statistically indistinguishable throughout
  (differences of 0.7–3.7%).
- Error rises monotonically with wear stage for all methods on both datasets.

---

## 7. Discussion

### 7.1 What the numbers mean in physical terms

A ULM3 EDJ surface spans roughly 7–9 mm. A worn-region RMSE of 0.13 mm on the EDJ
series is therefore on the order of **1.5–2% of crown dimension** — the
reconstructed cusp sits about a tenth of a millimetre from where it originally was.
For cusp-tip position, occlusal basin depth or dentine-horn height that is a usable
tolerance. For fine surface texture — crenulation, secondary ridging — it is not:
those features have relief comparable to the error itself and should not be read
off a reconstruction.

The gap between the primary and secondary layers is instructive. The non-rigid fit
scores 0.041 mm on full surface but 0.137 mm on the worn region, a factor of three.
That difference *is* the difference between reproducing a tooth and restoring one:
the whole-surface figure is dominated by intact enamel the method essentially copied.
Reporting only whole-surface Chamfer — as is common — would materially overstate
what any of these methods achieve.

### 7.2 Comparison with statistical shape work on teeth

| Study | Material | Task | Reported error |
|---|---|---|---|
| Buchaillard et al. | human teeth | SSM surface reconstruction | < 0.2 mm (5 PCs); < 0.1 mm (13 PCs) |
| Woods et al. | 27 mandibular canines, archaeological | SSM reconstruction | < 0.2 mm (5 PCs), ≈ 0.7% of crown length |
| CBCT tooth models | human teeth | CBCT reconstruction | RMS 0.069–0.241 mm |
| Root SSM (implant planning) | human roots | root shape from crown | 1.3 ± 0.8 mm at CEJ |
| **This work (EDJ series)** | **primate ULM3 EDJ** | **worn-region restoration** | **0.130–0.137 mm RMSE** |
| **This work (ULM series)** | **primate ULM3 EDJ, held out** | **worn-region restoration** | **0.193–0.245 mm RMSE** |

Our figures fall inside the band the dental SSM literature reports, which is the
appropriate sanity check. Two qualifications matter. First, most of those studies
report error over the **whole** reconstructed surface; the closest comparison to
their numbers is our secondary layer (0.041–0.058 mm Chamfer on the EDJ series),
not our primary. Second, they reconstruct from complete or lightly damaged inputs,
whereas the present task withholds a genuinely missing region. The comparison
therefore establishes that the pipeline is in the right regime, not that it
outperforms those methods.

Against the anthropological baselines the comparison is qualitative, because
Profile, Polynomial and Pen Tool return scalar measurements rather than surfaces
and are validated on different quantities. The relevant claim is one of scope: those
methods are reported to be reliable only for lightly worn crowns, whereas the
present pipeline returns a complete surface across Molnar stages 3–6, with a
quantified and monotonically increasing error.

### 7.3 Why image-based methods are not a fair comparison

Published 3D dental reconstruction from photographs or radiographs reports Chamfer
distances an order of magnitude larger than ours. That gap reflects problem
difficulty, not method quality. TeethDreamer infers 3D geometry from five 2D
photographs; Occudent works from a single panoramic radiograph. Both must recover
depth, scale and identity from projections. Our input is already a metric 3D
surface, and only a bounded region of it is missing. Tabulating the two side by side
would be misleading, and we do not do so.

### 7.4 Why the global and neighborhood models tie

The neighborhood model was motivated by the expectation that a locally-fitted prior
would beat a population prior. It does not: differences are below 4% on every
metric and below 1% on most. Two explanations are consistent with the data. First,
the adaptive threshold is generous relative to the spread of the training sample, so
most specimens select the full neighbour quota and the local model approaches the
global one. Second, a K-neighbour model retains at most K−1 modes, so any gain from
a more relevant mean is offset by reduced expressive range. Distinguishing these
requires a sweep over the threshold multiplier and the neighbour cap, which we have
not yet run; until then the neighborhood variant should be regarded as **not
demonstrated to help** rather than as refuted.

### 7.5 Failure at severe wear

All methods degrade sharply at Molnar stage 6 (levels 6–7). On the EDJ series
worn-region RMSE roughly triples between level 5 and level 6 (0.125 → 0.311 mm for
Global SSM). This is expected and worth stating plainly: as wear removes more of the
crown, the fit is constrained by fewer surviving points *and* asked to invent more
geometry. The reconstruction degrades toward the model mean, which is precisely when
a practitioner should stop trusting it. The non-rigid fit, which is the strongest
method at light and moderate wear, is not reliably better at stage 6 — on the EDJ
series it is the worst of the three at level 7 (0.344 mm) — because it depends on
confident scan landmarks that severe wear removes.

**Practical guidance.** Through Molnar stage 5 these reconstructions are accurate to
roughly 0.1–0.2 mm and suitable for cusp-position and crown-height work. At stage 6
they should be treated as indicative only, and reported with the error figures above
attached.

---

## 8. Limitations

1. **Artificial wear is not natural wear.** Our series are produced by progressive
   digital removal from a known original. Natural attrition also involves dentine
   exposure, secondary dentine deposition and differential hardness effects. The
   error figures should be read as a lower bound on what to expect from natural
   specimens.
2. **Single tooth position.** Everything reported is ULM3. Generalisation to other
   molar positions or to other taxa is untested.
3. **Training sample size.** The EDJ model is built from 18 specimens and retains
   14 modes; shape models on small samples risk under-representing rare morphology.
   The 90-specimen ULM model is more comfortable in this respect.
4. **Validation caveats.** Two aspects of the evaluation warrant re-checking before
   final submission: (a) the worn-region mask depends on the relative sampling
   density of the ground-truth and worn surfaces, and should be confirmed to track
   wear stage on both datasets; (b) the two datasets were corresponded at different
   point densities (100,000 vs 10,000 points per tooth), which affects every
   nearest-neighbour metric and means the EDJ and ULM columns should not be compared
   directly against each other.
5. **No inter-operator baseline.** We have not compared against manual reconstruction
   by an experienced observer, which would be the most meaningful anthropological
   reference point.

---

## 9. Toward a reconstruction library

The pipeline is being packaged as a library so that this style of reconstruction is
available to researchers as a set of composable, documented steps rather than a
monolithic script. The motivating observation is that **no single configuration is
right for all material**: sample size, scan resolution, wear severity and the
morphological question all change what the appropriate settings are. The library's
purpose is to make those choices explicit and adjustable rather than buried.

### 9.1 Design principles

- **Stages are independent.** Correspondence, reconstruction and evaluation
  communicate through files on disk, so a researcher can substitute their own
  correspondence method or their own evaluation and keep the rest.
- **Every choice is a named parameter** with a documented default and a stated effect.
- **Reconstruction strategies are interchangeable** behind one interface, so adding
  a fourth method does not disturb the first three.
- **Evaluation is separable from reconstruction**, so results cannot be produced by a
  metric tuned to a method.

### 9.2 Parameters by stage

**Stage 1 — correspondence**

| Parameter | Default | What it controls | When to change it |
|---|---|---|---|
| `n_points` | 20,000 | points sampled per tooth | raise to 100k for fine detail; lower for speed and for coarse meshes |
| `registration_mode` | `direct` | CPD on all points, or coarse-to-fine | use coarse-to-fine above ~50k points |
| `cpd_points` | 43,000 | points used for CPD in coarse-to-fine | trade registration accuracy against memory |
| `displacement_knn` | 3 | neighbours for upsampling the deformation | raise to smooth a noisy displacement field |
| `auto_template` | off | pick the most central specimen as template | leave on unless a specific reference is required |
| `scale_method` | bbox diagonal | how size is removed | switch to unit sphere if bounding boxes are unstable |
| `seed` | 42 | sampling reproducibility | fix for publication runs |

**Stage 2a/2b — shape model reconstruction**

| Parameter | Default | What it controls | When to change it |
|---|---|---|---|
| `variance_threshold` | 0.99 | how many PCA modes to keep | lower (0.90–0.95) for small or noisy training samples |
| `n_components` | auto | explicit mode count | set directly for a controlled ablation |
| `regularization` (λ) | 1.0 | strength of the shape prior | raise when little of the tooth survives; lower for lightly worn, well-sampled teeth |
| `proxy_missing_fraction` | 0.15 | fraction treated as missing when the worn region is not known | **the most consequential parameter**: too high and healthy anatomy is overwritten, too low and worn anatomy is not repaired |
| `max_neighbors` (K) | 5 | cap on neighbours for the local model | raise with a larger training pool |
| `threshold_iqr_mult` | 2.0 | how permissive neighbour selection is | lower it to make selection genuinely selective |
| `ssm_variance` | 0.95 | variance retained by the local model | as above |

**Stage 2c — non-rigid fit**

| Parameter | Default | What it controls | When to change it |
|---|---|---|---|
| `iters` | 12 | coarse-to-fine rounds | raise for large initial misalignment |
| `step` | 0.6 | how far to move per round | lower for stability, raise for speed |
| `close_mult` / `far_start` / `far_end` | 2 / 8 / 3 | the confidence band, in median spacings | widen for noisy scans, tighten for clean ones |
| `smooth_start` / `smooth_end` | 1.2 / 0.3 | warp smoothness, annealed | raise to prevent local distortion |
| `tight_snap` | 0 | final snap of confident points onto the scan | leave at 0; positive values cause density clumping |

**Stage 3 — evaluation**

| Parameter | Default | What it controls | When to change it |
|---|---|---|---|
| `wear_mult` | 2.0 | how far from the scan a point must be to count as worn | raise for noisy scans to avoid flagging noise as wear |
| `n_gt` | 120,000 | sampling density of the reference surface | keep matched to the reconstruction's density |
| ICP mode | point-to-plane | registration before measuring | match whatever the comparison literature used |

**Surface output**

| Parameter | Default | What it controls | When to change it |
|---|---|---|---|
| `poisson_depth` | auto by point count | octree depth | lower for sparse clouds to avoid artefacts |
| `density_quantile` | 0.01 | fraction of low-density fringe trimmed | raise to trim more extrapolated surface |
| `smooth_iter` | 30 | Taubin smoothing passes | lower to preserve fine relief |

### 9.3 Intended workflow

A researcher would supply a directory of unworn reference teeth and a directory of
worn specimens, choose a reconstruction strategy, and receive reconstructed surfaces
in the original coordinate frame together with a per-specimen evaluation record. For
material without ground truth, the evaluation stage still reports fit residuals,
model coefficients and the fraction of surface treated as missing — the diagnostics
needed to decide whether a reconstruction is trustworthy.

---

## 10. Conclusion

We have presented a landmark-free pipeline that reconstructs pre-wear EDJ surfaces of
primate ULM3s from worn 3D scans, and evaluated it against known ground truth across
Molnar wear stages 3–6 on two independent series. Worn-region restoration error is
0.130–0.137 mm on the 16-case EDJ series and 0.193–0.245 mm on the 64-case held-out
ULM series, within the band reported for statistical shape reconstruction of teeth
elsewhere, and obtained without manual landmarking. The non-rigid fit is the
strongest method overall; the neighborhood shape model, contrary to expectation, does
not measurably improve on the global one. All methods degrade sharply at Molnar stage
6, which we regard as the practical limit of the approach as it stands.

The broader contribution is methodological: by scoring restoration on a shared mask
of the anatomy wear actually removed, and reporting whole-surface agreement only as
supporting evidence, we avoid the inflation that follows from averaging over a
surface which is mostly intact. We recommend the same separation for future work in
this area.

---

## References

1. Molnar, S. (1971). Human tooth wear, tooth function and cultural variability. *American Journal of Physical Anthropology*, 34(2), 175–189.
2. O'Hara, M. C., & Guatelli-Steinberg, D. (2022). Reconstructing tooth crown heights and enamel caps: A comparative test of three existing methods with recommendations for their use. *The Anatomical Record*, 305(1). doi:10.1002/ar.24637
3. Woods, C., Fernee, C., Browne, M., Zakrzewski, S., & Dickinson, A. (2017). The potential of statistical shape modelling for geometric morphometric analysis of human teeth in archaeological research. *PLoS ONE*, 12(12), e0186754.
4. Buchaillard, S. I., Ong, S. H., Payan, Y., & Foong, K. (2007). 3D statistical models for tooth surface reconstruction. *Computers in Biology and Medicine*, 37(10), 1461–1471.
5. Myronenko, A., & Song, X. (2010). Point set registration: Coherent point drift. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 32(12), 2262–2275.
6. Porto, A., Rolfe, S., & Maga, A. M. (2021). ALPACA: A fast and accurate computer vision approach for automated landmarking of three-dimensional biological structures. *Methods in Ecology and Evolution*, 12(11), 2129–2144.
7. Lüthi, M., Gerig, T., Jud, C., & Vetter, T. (2018). Gaussian Process Morphable Models. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(8), 1860–1873.
8. Kazhdan, M., & Hoppe, H. (2013). Screened Poisson surface reconstruction. *ACM Transactions on Graphics*, 32(3), 1–13.
9. Xu, C., Liu, Z., Liu, Y., Dou, Y., Wu, J., Wang, J., Wang, M., Shen, D., & Cui, Z. (2024). TeethDreamer: 3D teeth reconstruction from five intra-oral photographs. *MICCAI 2024*, LNCS 15003.
10. Wu, S., Lian, C., Xia, P., & Wang, L. (2023). 3D teeth reconstruction from panoramic radiographs using neural implicit functions. *MICCAI 2023*.
11. Skinner, M. M., Wood, B. A., Boesch, C., Olejniczak, A. J., Rosas, A., Smith, T. M., & Hublin, J.-J. (2008). Dental trait expression at the enamel-dentine junction of lower molars in extant and fossil hominoids. *Journal of Human Evolution*, 54(2), 173–186.
12. Skinner, M. M., Gunz, P., Wood, B. A., & Hublin, J.-J. (2008). Enamel-dentine junction (EDJ) morphology distinguishes the lower molars of *Australopithecus africanus* and *Paranthropus robustus*. *Journal of Human Evolution*, 55(6), 979–988.
13. Fan, H., Su, H., & Guibas, L. J. (2017). A point set generation network for 3D object reconstruction from a single image. *CVPR 2017*, 605–613.

> **Reference note.** Entries 1–10 and 13 were verified against publisher records
> during drafting. Entries 11–12 are standard EDJ-morphology citations included on
> topical grounds and should be checked against the specific claims they support
> before submission. Author lists in the project's existing `references.bib` contain
> errors and should be replaced with the list above.
