# Correlated errors and graph decoding on calibration-realistic noise

**Format:** three notebooks. A small imported utility library for the parts nobody
learns anything from; everything a reader should judge is written inline.

## 1. Question

> Decomposing a circuit's detector error model into a graphlike form discards
> correlations between the X and Z sectors. On heterogeneous, calibration-derived
> noise, how much does that cost minimum-weight matching, and can a small learned
> decoder recover any of it?

A `Y` error on a data qubit fires an X-check and a Z-check together. Standard MWPM
decodes the two sectors independently and throws that link away. This is a known
effect with a known sign — correlated matching buys roughly a 10% effective
distance improvement in the literature — which makes it a good target: the
experiment can succeed, and if it fails the failure is diagnosable rather than
mysterious.

The heterogeneity comes from a frozen IBM calibration snapshot, so the noise
profile is real rather than a flat depolarizing parameter.

## 2. What this is not

- **No hardware run.** Simulation only, from one frozen snapshot.
- **No claim that this circuit runs on that device.** IBM's heavy-hex lattice is
  degree ≤ 3; a rotated surface code needs degree-4 ancillas. Notebook 02 asserts
  this rather than assuming it. The snapshot supplies *error rates*, not a layout.
- No threshold or pseudo-threshold estimate.
- No decoder-superiority claim beyond the tested distances and noise settings.
- No claim the snapshot captures drift, leakage, crosstalk, or non-Markovian effects.

## 3. Fixed scope

| | |
| --- | --- |
| Code | Rotated surface code, `d ∈ {3, 5}`, `rounds = d` |
| Experiment | Logical `Z` memory; logical `X` failure channel |
| Circuits | Stim `surface_code:rotated_memory_z`, noise applied per instruction |
| Noise | Calibration-derived heterogeneous, plus a matched-mean uniform control |
| Scaling | `λ ∈ {0.5, 1.0, 1.5}`, `q_λ = 1 − (1 − q)^λ` |
| Decoders | none · MWPM (graphlike, Z-sector) · learned (both sectors) |
| Seeds | 3 model seeds |
| Metric | Logical failure probability, 95% Wilson interval |

**The control that makes it an experiment:** heterogeneous noise is compared against
uniform noise *at matched mean error rate*. Without that, any difference is just
"one setting had more errors."

## 4. Where code lives

The rule: **if a reviewer would judge you on it, it is in the notebook. If they would
scroll past it, it is in the library.**

**Imported** — `pip install git+https://github.com/LukeJamesMiller/qec-utils.git@v0.1.0`
in cell 1 of each notebook, pinned to a tag so the notebooks stay reproducible:

- `RotatedSurfaceCode` — long, already written, already verified against Stim's
  coordinate convention at `d = 3, 5, 7`. The geometry figure is what a reader wants,
  not 400 lines of construction.
- Backend-snapshot loading and normalisation — boilerplate.
- Plot palette and one save helper.

**Inline, copied into the notebook that uses it:**

- Wilson interval — ~10 lines, and writing it shows you know why the normal
  approximation is wrong at small `p`.
- Noise scaling `q_λ = 1 − (1 − q)^λ` — two lines, and it is a design decision.
- The rule assigning calibration entries to circuit locations — this is the
  modelling choice.
- Detector-graph feature construction — this is the modelling choice.
- The decoder model — this is the model.
- Logical failure counting — three lines.

Duplication between notebooks is fine. A notebook that cannot be read top to bottom
has failed at its only job.

## 5. Notebooks

### `01_code_and_detectors.ipynb`
*What object does a decoder actually see?*

Rotated-code geometry and stabilizers; the Stim memory circuit; measurement rounds
and detection events; the detector error model, decomposed and undecomposed.
Ends by showing one `Y` fault firing detectors in both sectors, and the same fault
after graphlike decomposition — the picture the rest of the project is about.

Outputs: `figures/code_geometry.png`, `figures/detector_graph.png`.

### `02_calibration_noise_and_mwpm.ipynb`
*Does realistic heterogeneity change the baseline?*

Load the frozen snapshot; show the spread in two-qubit and readout error across the
device (this is the motivating plot). Assert `max degree ≤ 3` and state the
embedding limitation. Assign calibration entries to circuit locations; build the
heterogeneous noisy circuit and the matched-mean uniform control. MWPM and
no-correction across both, at all three `λ`.

Outputs: `results/mwpm.csv`, `figures/calibration_spread.png`, `figures/mwpm_vs_noise.png`.

### `03_correlated_decoding.ipynb`
*Can a small model recover what decomposition discarded?*

One model, fixed before results are seen. Trained at `d = 3`, heterogeneous,
`λ = 1.0`, on both detector sectors. Evaluated in-domain, at the other two `λ`, and
zero-shot at `d = 5`. Compared against MWPM on identical shots. Final table and
limitations.

Outputs: `results/decoders.csv`, `figures/decoder_comparison.png`.

## 6. Grid

| Axis | Values | Count |
| --- | --- | ---: |
| Distance | 3, 5 | 2 |
| Noise profile | heterogeneous, matched-mean uniform | 2 |
| Scaling `λ` | 0.5, 1.0, 1.5 | 3 |

12 conditions. Each evaluated by no-correction, MWPM, and 3 model seeds on identical
shots. Training happens once: `d = 3`, heterogeneous, `λ = 1.0`.

**Shots:** 50,000 per test condition, 20,000 train, 5,000 validation. Stim samples
these in seconds; only training scales. Extend a condition to 200,000 if it shows
fewer than 200 logical failures. That rule is fixed now, before any comparison.

**Reporting:** every logical failure probability carries a Wilson interval. Model-seed
spread is reported as min/max separately, never folded into the binomial interval.

## 7. Stopping points

| Stop | You have | Worth showing? |
| --- | --- | --- |
| After 01 | Correct code, circuit and detector semantics, with the correlation problem stated | Yes — a competent QEC explainer |
| After 02 | Real calibration data, an honest embedding caveat, MWPM under matched controls | Yes — a complete non-ML result |
| After 03 | The full comparison | The intended project |

Notebook 02 is the minimum respectable stop. No later notebook may paper over a
failure in an earlier one.

## 8. Repository

```text
surface-code-correlated-decoding/
├── README.md              ← the deliverable; a reader should not need the notebooks
├── EXPERIMENTAL_PLAN.md
├── requirements.txt
├── data/ibm_<backend>_<date>.json
├── notebooks/
│   ├── 01_code_and_detectors.ipynb
│   ├── 02_calibration_noise_and_mwpm.ipynb
│   └── 03_correlated_decoding.ipynb
├── results/*.csv
└── figures/*.png
```

No `src/`, no test directory, no config files. Assertions live in the notebook cell
that earns them. The one exception is `qec-utils`, which is a separate repository
with its own tests because it is meant to outlive this project.

Every notebook opens in Colab from the README badge and runs end to end.

## 9. Done when

1. Notebook 01 shows a `Y` fault firing both sectors, and shows what decomposition does to it.
2. Notebook 02 states the heavy-hex limitation as a checked assertion, not a caveat paragraph.
3. Heterogeneous and uniform noise are compared at matched mean error rate.
4. All decoders are scored on identical shots per condition.
5. The `d = 5` result is zero-shot, with no `d = 5` labels touched.
6. Every number in the README traces to a committed CSV.
7. The limitations section distinguishes real calibration data from the approximate
   Pauli simulation built on top of it.

The success criterion is not that the model wins. It is that the question is sharp,
the control is real, and the answer is reported either way.
