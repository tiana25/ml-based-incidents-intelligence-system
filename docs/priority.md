# Priority in the Dataset Schema

Schema: `id, incident_group_id, source_type, text, label, priority, timestamp`

## Why `priority` is in the schema

`priority` is a **ground-truth label generated at data creation time**, not something the ML model predicts. It serves two purposes:

1. **Rule consistency verification** — after the pipeline runs `score_priority()` on each incident, we compare its output against the stored `priority` to confirm the rule-based scorer is self-consistent (target: ≥ 90% match). This is the "evaluation" for the priority component.

2. **Correlation escalation input** — when the correlation layer merges signals into a fused incident report, it reads `priority` values across the cluster to decide whether to escalate.

## How `priority` is defined

It is derived from the **same keyword rules** used by `score_priority()` at runtime, applied during synthetic data generation in `src/data/generate.py`:

```
HIGH_KEYWORDS   = ["critical", "outage", "down", "failed", "fatal", "exceeded threshold", "unavailable", "crash", "oom"]
MEDIUM_KEYWORDS = ["warning", "slow", "degraded", "latency", "timeout", "retry", "elevated", "error"]
```

Source type also contributes a base priority:

```
alert  → base Medium
ticket → base Medium
log    → base Low
```

Final priority = `max(keyword_match_priority, source_base_priority)`

## Why this is not circular ML

The priority component is intentionally **rule-based**, not a trained classifier. We define the rules, generate labels using those rules during data generation, then verify that `score_priority()` reproduces the same labels at runtime. That is a rule consistency check — not training a model to re-learn the rules.
