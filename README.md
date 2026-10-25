<div align="center">

# ECG-PACE

### Hypothesis-Guided Evidence Acquisition for ECG Reasoning with Large Language Models

**Sequential ECG evidence search · Prospective action valuation · Frozen-backbone reasoning**

[Quick start](#quick-start) · [Method](#method-at-a-glance) · [Real ECGs](#working-with-ecg-records) · [Model integration](#connect-a-frozen-ecg-model) · [Evaluation](#evaluation) · [Documentation](docs/architecture.md)

</div>

---

ECG-PACE is a modular research framework for choosing **which ECG evidence to
inspect next**. Rather than following a fixed checklist, an agent maintains
competing diagnostic hypotheses, estimates the diagnostic value of candidate
measurements, acquires selected observations, and updates its beliefs before
deciding whether additional evidence is worthwhile.

The ECG reasoning backbone remains **frozen**. Adaptation happens in the
external evidence-selection and decision process, not through parameter
updates of the foundation model.

> **Scope.** This repository contains the source code for the agent loop,
> waveform-measurement interfaces, evidence-value policy, model-provider
> interface, conservative report editing, evaluation utilities, and tests.
> The bundled example uses generated ECG-like signals and a scripted model.
> It is intended for software verification only. It is not an ECG-R1 model
> checkpoint, a trained value model from the manuscript, or a clinical study.
> See [Research artifacts and data](#research-artifacts-and-data) for what
> needs to be provided for a full research run.

## Highlights

- **Hypothesis-guided acquisition:** maintains diagnosis-specific beliefs and
  dispatches lead-specific ECG queries based on unresolved hypotheses.
- **Prospective evidence-value estimation:** balances predicted improvement,
  tool cost, and estimated observation risk **before** acquiring the result.
- **Tool-grounded measurement:** provides annotated RR, PR, QRS, QT/QTc,
  atrial-activity, axis, ST-level and waveform-summary actions with provenance.
- **Evidence-conditioned belief updates:** uses accepted, patient-specific
  observations to obtain revised diagnostic beliefs from the frozen model.
- **Adaptive stopping:** stops when no candidate is useful or budget remains
  insufficient, rather than performing every available measurement.
- **Selective report revision:** accepts only edits anchored to previously
  validated patient evidence and preserves all untouched report content.
- **Inspectable trajectories:** saves ranked action values, tool observations,
  reliability, belief changes, proposed edits and the stopping decision.

## Method at a glance

```text
                  +--------------------------------------+
                  |   Frozen ECG Reasoning Backbone      |
12-lead ECG ----> |   Initial Report + Diagnosis Beliefs| 
                  +------------------+-------------------+
                                     |
                                     v
                     Competing Diagnostic Hypotheses
                                     |
                                     v
                     Prospective Evidence Valuation
                     Expected Gain - Cost - Risk
                                     |
                                     v
                       Select Next ECG / Criterion Tool
                                     |
                                     v
                       Validate Acquired Observation
                                     |
                                     v
                        Evidence-Conditioned Beliefs
                                     |
                        +------------+------------+
                        |                         |
                 Useful action                Stop / budget
                  remains?                      exhausted
                        |                         |
                        +---- back to search     v
                                         Selective Report Revision
                                                   |
                                                   v
                                          Evidence-Grounded Report
```

At step `t`, the agent state is `s_t = (b_t, E_t, B_t)`: diagnosis beliefs,
collected evidence, and remaining action budget. It ranks available actions by

```text
V_t(a) = E_z[ Q(s_(t+1) | a,z) - Q(s_t) ] - lambda * C(a) - mu * R(a)
```

where the expectation is computed from an **outcome model for observations not
yet made**. Once action `a` returns a valid result, a frozen-model query provides
new evidence-conditioned beliefs. The update is

```text
b_(t+1)(c) = (1 - q_t) * b_t(c) + q_t * b_evidence(c)
```

where `q_t` is the accepted observation's reliability. Clinical reference
retrieval can guide action choice, but it is never itself considered a
patient-specific ECG measurement.

See [Architecture](docs/architecture.md) for component-level details.

## Quick start

**Requirements:** Python 3.10+ and NumPy. GPU access is **not required** for
running the package's generated example and tests.

```bash
# From the downloaded or cloned ECG-PACE directory:
cd ECG-PACE
python -m pip install -e '.[dev]'

# Execute the complete agent loop on generated waveforms.
ecg-pace demo --out outputs/demo_trace.json

# Run the automated tests.
pytest -q
```

The demo prints a concise summary and writes a complete JSON trace to
`outputs/demo_trace.json`. The generated example contains a deliberate change
in RR timing, and the scripted backbone uses an explicitly coded response to
the observed RR measurement. No interpretation in this example should be read
as medical advice or benchmark evidence.

Example trace fields:

```json
{
  "record_id": "generated-example-001",
  "stop_reason": "insufficient_expected_value",
  "trajectory": [
    {
      "step": 1,
      "action": "rr_ii",
      "action_value": {
        "expected_gain": 0.39,
        "cost_penalty": 0.015,
        "reliability_penalty": 0.008,
        "value": 0.367
      },
      "evidence_id": "ev:rr_ii"
    }
  ]
}
```

The fragment above is a **format illustration**, not a complete saved result;
all available fields are written by the running program.

## Repository structure

```text
ECG-PACE/
├── ecgpace/
│   ├── agent.py           # Search, reliability-weighted updates, revision gate
│   ├── backbone.py        # Frozen model interface / local HTTP adapter
│   ├── cli.py             # Public command-line interface
│   ├── config.py          # Search and edit thresholds
│   ├── demo.py            # Generated signal + scripted model demonstration
│   ├── evaluation.py      # Label metrics, paired bootstrap confidence intervals
│   ├── io.py              # 12-lead record and action-catalog loaders
│   ├── judge.py           # Aggregate seven-criterion structured judge outputs
│   ├── models.py          # Data contracts and trace structures
│   ├── policy.py          # Prospective action valuation and offline fitting
│   └── tools.py           # Lead-specific signal tools and evidence records
├── examples/
│   ├── action_catalog.json
│   ├── criteria.json
│   ├── policy_training_rows.jsonl
│   └── frozen_backend_service.py
├── docs/
│   └── architecture.md
├── tests/
├── pyproject.toml
└── LICENSE
```

## Working with ECG records

For standard signal-based experiments, provide one `.npz` file with the **12
canonical leads** and a JSON metadata sidecar. Each lead is a one-dimensional
NumPy array in **millivolts**, sampled at the same frequency and length:

```python
import numpy as np

np.savez(
    "record.npz",
    I=lead_I, II=lead_II, III=lead_III,
    aVR=lead_aVR, aVL=lead_aVL, aVF=lead_aVF,
    V1=lead_V1, V2=lead_V2, V3=lead_V3,
    V4=lead_V4, V5=lead_V5, V6=lead_V6,
)
```

Corresponding `record.json`:

```json
{
  "record_id": "local-record-0001",
  "fs": 500,
  "annotations": {
    "II": {
      "r_peaks": [350, 820, 1280],
      "qrs_onsets": [330, 800, 1260],
      "qrs_offsets": [370, 840, 1300],
      "p_onsets": [260, 730, 1190],
      "p_peaks": [290, 760, 1220],
      "t_offsets": [540, 1010, 1460]
    }
  }
}
```

All fiducial locations are **zero-based sample indices**. The shown numbers
are illustrative annotations, not extracted annotations from the paper.
Atrial-activity and interval tools require the corresponding fiducial events.
If necessary events are unavailable, the tool returns an `unavailable`
observation, rather than guessing a clinical value. The lightweight waveform
checks supplied here are not a substitute for a validated ECG delineation
pipeline.

Action catalogs are JSON arrays of actions specifying `id`, `kind`, `lead`,
`cost`, optional hypothesis targets, and optional acquisition windows in
seconds. Supported action kinds include:

| Action | Output | Required input |
| --- | --- | --- |
| `rr_intervals` | RR summary and successive interval change, ms | R-peak annotations |
| `pr_interval` | Median PR, ms | P onset and QRS onset |
| `qrs_duration` | Median QRS width, ms | QRS onset and offset |
| `qt_interval` | Median QT, ms | QRS onset and T offset |
| `qtc_interval` | Bazett QTc, ms | RR, QRS onset and T offset |
| `p_wave_presence` | Fraction with annotated preceding P-wave peaks | R and P peaks |
| `st_level` | ST displacement approximation, mV | QRS onset and offset |
| `frontal_axis` | Approximate frontal QRS axis, degrees | QRS boundaries in leads I/aVF |
| `waveform_summary` | Min/max/RMS voltages, mV | Raw lead samples |
| `clinical_criterion` | Sourced reference entry | Local/versioned criterion record |

An optional `support_rule` converts an **observed numeric feature** into
`support` / `oppose` for policy bookkeeping. Its threshold is research
configuration, not a medical-grade diagnosis. It is evaluated only after a
measurement finishes.

## Connect a frozen ECG model

The `FrozenBackbone` interface isolates all model-specific operations:

1. **`interpret(record)`** — initial ECG report and diagnosis-level beliefs.
2. **`conditioned_beliefs(record, initial_report, current, evidence)`** —
   evidence-conditioned beliefs in the same diagnosis vocabulary.
3. **`propose_edits(record, initial_report, beliefs, evidence)`** — conservative
   text replacements, each naming supporting evidence IDs.

Beliefs follow the manuscript's diagnosis-level convention:
`log P(Present) - log P(Absent)`. They are **log odds**, not calibrated
probabilities. The code stores these scores without normalizing across
hypotheses, since diagnoses need not be mutually exclusive.

For a separately served ECG-R1-8B-RL model, connect a local REST bridge with
three endpoints:

| Endpoint | Input | Required output |
| --- | --- | --- |
| `POST /interpret` | `record` | `report`, `beliefs` |
| `POST /beliefs` | `record`, `initial_report`, `current_beliefs`, `evidence` | `beliefs` |
| `POST /edits` | `record`, `initial_report`, `beliefs`, `evidence` | `edits` |

A record payload contains `record_id`, `fs`, 12 lead arrays under `signals_mv`,
optional fiducials, and optional local image-path metadata. The service is
responsible for converting this payload to the model's required modalities,
formatting belief queries, and parsing token log-probabilities. The package
**does not** assume that an image file path is a model-ready image.

A small provider-contract example is included in
[`examples/frozen_backend_service.py`](examples/frozen_backend_service.py).
It hosts the **scripted example backend** for interface testing, not ECG-R1.

```bash
# Terminal 1: start an example JSON model service.
python examples/frozen_backend_service.py

# Terminal 2: create local example inputs and fit the included teaching data.
ecg-pace export-example --directory outputs/generated_record
ecg-pace fit-policy --input examples/policy_training_rows.jsonl --out outputs/local_policy.json

# For your own study, replace these with curated ECG data and fitted parameters.
ecg-pace run \
  --npz outputs/generated_record/record.npz \
  --metadata outputs/generated_record/record.json \
  --catalog examples/action_catalog.json \
  --policy outputs/local_policy.json \
  --criteria examples/criteria.json \
  --endpoint http://127.0.0.1:8765 \
  --max-actions 4 \
  --out outputs/record_trace.json
```

By default, the HTTP adapter permits only loopback hosts. Remote transport
requires `--allow-remote` and should be used only after data-protection and
access-control review. The source ECG samples are transmitted to the configured
service; do not send identifiable patient data to an unapproved endpoint.

## Fit an evidence-value policy

The policy module exposes expected outcome distributions and outcome-dependent
quality changes. The included **action-conditional empirical estimator** can
be fitted from offline observations with these JSONL fields:

```json
{"action_id":"rr_ii","outcome":"support","delta_quality":0.12,"unreliable":false}
```

- `action_id`: stable identifier in the candidate-action catalog.
- `outcome`: `support`, `oppose`, `uncertain`, or `unavailable`.
- `delta_quality`: observed change in quality, normalized to `[-1, 1]`.
- `unreliable`: whether the action returned unusable or untrustworthy evidence.

```bash
ecg-pace fit-policy \
  --input /path/to/training_action_observations.jsonl \
  --out /path/to/policy.json
```

The values in [`examples/policy_training_rows.jsonl`](examples/policy_training_rows.jsonl)
are **made-up teaching data only**. Use observations drawn from the designated
training/development split to fit a research policy. Keep evaluation records
and reference diagnoses inaccessible to online action selection.

**Estimator note.** The paper describes state-dependent outcome and diagnostic
quality models. `EmpiricalPolicy` is a simpler action-conditional implementation
that makes the full search protocol executable. Researchers may substitute a
state-conditioned predictor without modifying `PACEAgent`, so long as the
candidate-action evaluation remains prospective and respects the data split.
The paper's trained estimator weights are not packaged here.

## Settings

Default values are aligned to the implementation configuration in the paper:

| Parameter | Default | Role |
| --- | ---: | --- |
| Frozen backbone | ECG-R1-8B-RL (external) | ECG interpretation and belief queries |
| Maximum tool calls | 4 | Hard upper bound (allowed: 1–4) |
| `cost_weight` (`λ`) | 0.10 | Penalize acquisition cost |
| `reliability_weight` (`μ`) | 0.20 | Penalize unreliable observations |
| `stop_threshold` (`δ`) | 0.05 | Terminate when max action value is not greater |
| `revision_threshold` | 0.15 | Minimum claimed expected benefit for an edit |
| `minimum_evidence_reliability` | 0.50 | Basic local observation acceptance gate |

The additional minimum-reliability gate is an implementation safeguard, not a
reported tuning result. Cost units, expected gains, uncertainty estimates,
and quality scales must be aligned when training the value policy; changing
their scales changes the meaning of the thresholds.

## Evaluation

[`ecgpace/evaluation.py`](ecgpace/evaluation.py) and [`ecgpace/judge.py`](ecgpace/judge.py) provide:

- Set-based diagnosis precision, recall and micro-F1 (judge-free).
- Paired bootstrap confidence intervals for per-record score differences;
  default 10,000 resamples.
- Validation and per-record aggregation of the seven structured ECG judge criteria
  (`DA`, `Credit`, `AC`, `AR`, `LEV`, `GEU`, `EBR`, `RDP`).

```python
from ecgpace.evaluation import diagnosis_scores, paired_bootstrap_ci

scores = diagnosis_scores(
    predicted=[{"pac"}, {"rbbb", "sinus_rhythm"}],
    reference=[{"pac"}, {"rbbb"}],
)
ci = paired_bootstrap_ci(
    a=[0.9, 0.8, 0.7, 0.9],
    b=[0.8, 0.7, 0.7, 0.8],
    samples=10000,
)
```

The manuscript additionally reports diagnosis accuracy/credit, analysis
completeness/relevance, lead evidence validity, grounded understanding,
evidence-based reasoning, and realistic diagnostic process using external
judges. The judge-call service and benchmark test records are not included; this
package accepts structured judge output and computes its defined record scores.
Any final comparative scores require an explicitly configured evaluation
protocol, the appropriate judge service, and the authorized benchmark data.

## Research artifacts and data

The code is intentionally separate from third-party checkpoints and data.
For a research run with the original model and benchmark, users need:

1. Authorized access to the ECG-Grounding benchmark and the corresponding
   original waveform records, where permitted by the data license.
2. A working **frozen** ECG-R1-8B-RL inference service that supports the input
   signal/image format, the report prompt, and diagnosis-level log-probability
   queries used in the paper.
3. ECG feature annotations or validated lead-specific measurement components.
4. An action-value estimator fitted on an appropriate non-test split.
5. The intended evaluation model(s), scoring templates, and approval to process
   the selected dataset with those services.

This repository **does not contain** private clinical data, proprietary model
weights, benchmark predictions, or verified performance tables. Running the
included example is not evidence that a particular performance gain holds on
ECG-Grounding.

## Responsible use

ECG-PACE is research software. It is **not a medical device**, does not provide
validated medical diagnoses, and must not be used for patient care or triage.
A non-empty evidence record does not prove a diagnosis. In clinical research,
validate acquisition functions, waveform segmentation, measurement criteria,
model calibration, privacy protections, and the complete reporting workflow
before interpreting outputs.

## Testing and contribution

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Pull requests are welcome for additional waveform tools, model integrations,
policy estimators, evaluations, and documentation. Prefer tests with small,
fully generated signals. Avoid including patient records, access tokens,
restricted model weights, or licensed materials in pull requests.

## Citation

If this repository supports your research, cite the associated manuscript:

> **ECG-PACE: Hypothesis-Guided Evidence Acquisition for ECG Reasoning with
> Large Language Models.** Manuscript, 2026.

Formal publication and author metadata will be added once available.

## License

The source code is distributed under the [MIT License](LICENSE). External ECG
data, ECG-R1 weights, clinical criterion texts, and hosted model services may
carry separate access requirements and licenses.
