# Included examples

`policy_training_rows.jsonl` contains fabricated teaching values solely to show
how the fitting command accepts action-level training observations. They are
**not** ECG-Grounding results, observed clinical measurements, or paper metrics.

`action_catalog.json` is a minimal action registry, including optional feature
rules. The numerical comparison values are configurable research-side flags,
not independently validated diagnostic criteria.

`criteria.json` shows the retrieval data structure. Replace the local
checklist with appropriately sourced, versioned clinical content.

For the quickest self-contained tour run `ecg-pace demo` from the repository
root. See the main README for the real-record and frozen-model workflows.
