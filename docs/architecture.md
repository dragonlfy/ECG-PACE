# Architecture and methodological alignment

## Components

| Paper concept | Source module | Runtime behavior |
| --- | --- | --- |
| Frozen ECG model | `backbone.py` | Initial report and diagnosis-level log odds, evidence-conditioned belief query, narrow edit proposals |
| Diagnostic state | `models.py` | Beliefs, acquired evidence, remaining tool budget |
| Prospective evidence valuation | `policy.py` | Outcome distribution, expected quality gain, action cost, failure-risk penalty |
| Tool-grounded evidence | `tools.py` | Lead-specific measurements and nonpatient clinical-criterion retrieval |
| Belief updating | `agent.py` | Blend current and newly conditioned beliefs by evidence reliability |
| Adaptive stopping | `agent.py` | Stop at budget, no candidate, or maximum value not above threshold |
| Selective revision | `agent.py` | Keep unrelated report content; require known accepted evidence and benefit threshold |
| Offline summary statistics | `evaluation.py`, `judge.py` | Diagnosis-set metrics, structured judge aggregation and paired bootstrap intervals |

The inference loop never requests a measurement result before selecting that action.
The policy observes diagnostic beliefs and the already collected evidence, never
future measurements. `EmpiricalPolicy` is the provided simple training option;
it learns distributions grouped by action ID. The method allows a richer estimator
with `outcomes` and `unreliable_risk` methods conditioned on the entire state.

## Decision functions

1. `s_t = (b_t, E_t, B_t)`
2. `V(a) = Σ_z p(z|s_t,a) * ΔQ(s_t,a,z) − λ*C(a) − μ*R(a)`
3. `a* = argmax_a V(a)`
4. `b_(t+1) = (1 − q_t)*b_t + q_t*b_evidence`
5. Terminate at `max_a V(a) ≤ δ`, exhausted budget or no remaining action.

The default configuration is `λ=0.10`, `μ=0.20`, `δ=0.05`, the edit threshold
`0.15`, and a maximum of four acquisition actions. A confidence value `b(c)`
is represented as a log-odds score, not an independently calibrated probability.

## Evidence trust boundary

- Patient measurements must identify their lead, record, measurement source,
  window, numerical value, and units.
- Missing fiducials or invalid signal windows lead to `unavailable` evidence.
- Clinical-criterion records are reference material, not measurements of the
  current ECG. They cannot directly update diagnosis beliefs or support edits.
- Accepted observations are passed to the frozen model, and new beliefs are
  reliability-weighted using only the latest observation's quality coefficient.
- Proposed edits are rejected unless their evidence IDs resolve to accepted
  patient observations; exact one-off substring replacement preserves other text.

## Extension points

- Add a new `FrozenBackbone` subclass for the original model's native input format.
- Replace `EmpiricalPolicy` with an estimator trained on features of the current
  diagnostic state, using only development/training cases.
- Replace annotation-based measurements with separately validated waveform
  delineators. Do not treat waveform summaries as validated diagnoses.
- Replace local criterion snippets with attributable and versioned clinical sources.
- Extend `evaluation.py` with project-specific judge interfaces and aggregation.

## Practical limitations

The supplied waveform measurements require fiducial annotations for interval
and atrial-feature queries. The image path carried by `ECGRecord` is metadata;
a custom model provider must load or render image content if required by the
model. `HTTPBackbone` sends raw signals in millivolts as arrays, not a rendered
12-lead ECG image. Signal quality checks here are basic sanity tests, not
clinical measurement validation. The lightweight action-conditional estimator
is one executable policy implementation, not released trained paper weights.
