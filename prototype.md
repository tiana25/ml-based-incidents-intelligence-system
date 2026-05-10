# ML-Based Incident Intelligence System - Prototype Guide

## What this prototype does

Large organisations receive incident signals from multiple isolated systems simultaneously. A single real incident - such as an authentication service failure - generates a ServiceNow ticket, a log entry, and a monitoring alert within minutes of each other. Without a system to correlate these signals, engineers treat each in isolation, delaying diagnosis.

This prototype ingests multi-source incident data, classifies each signal by incident type, scores its priority, detects similarity between signals, and merges correlated signals into a single incident report with a confidence score.

---

## How everything connects

```
src/data/generate.py              ← Step 1: create synthetic data
src/features/embed.py             ← Step 2: extract DistilBERT embeddings
src/models/classifier.py          ← Step 3: train the classifier     (logs to MLflow)
src/evaluation/eval_classifier.py ← Step 4: evaluate classification  (logs to MLflow)
src/evaluation/eval_similarity.py ← Step 5: evaluate clustering

src/pipeline/classify.py          ← runtime: incident type inference
src/pipeline/prioritize.py        ← runtime: priority scoring
src/pipeline/similarity.py        ← runtime: cosine similarity + DBSCAN
src/dashboard/app.py              ← Streamlit UI
```

The split is intentional. **Training scripts** (`src/models/`, `src/features/`, `src/evaluation/`) run once to produce artefacts on disk. **Pipeline modules** (`src/pipeline/`) are the runtime inference layer called live by the dashboard.

---

## Full pipeline - how to run it

To run all steps in one go:

```bash
python train.py
```

`train.py` runs all 5 steps in order, prints which step is running and how long each took, and stops immediately if any step fails.

To run steps individually, execute them from the project root in order. Each step produces a file the next step depends on.

### Step 1 - Generate synthetic data

```bash
python src/data/generate.py
```

Produces `data/raw/synthetic_incidents.csv` - 450 rows representing 50 incident groups across 3 incident classes. Each group contains 3 signals (ticket + log + alert) describing the same underlying real incident from different source systems.

Example incident groups:

**Authentication failure**
- Ticket: `"Users cannot log in to VPN. Error: token validation failed"`
- Log: `"TokenValidationException: JWT expired at 2024-01-15 03:42:11"`
- Alert: `"auth_error_rate exceeded threshold 5% → 23% over 10min window"`

**Network issue**
- Ticket: `"Intermittent packet loss reported by multiple office locations"`
- Log: `"DNS resolution timeout for host db.internal after 3 retries"`
- Alert: `"ALERT: packet_loss_rate > 15% on edge router eth0"`

**Deployment issue**
- Ticket: `"Service unavailable after latest release, rollback requested"`
- Log: `"Pod crash-looping: OOMKilled, memory limit 512Mi exceeded"`
- Alert: `"health_check failed for payment-service 3/3 replicas down"`

Schema: `id, incident_group_id, source_type, text, label, priority, timestamp`

### Step 2 - Extract embeddings

```bash
python src/features/embed.py
```

Downloads `distilbert-base-uncased` on the first run (cached locally after that). Passes every row's text through DistilBERT and extracts the `[CLS]` token output - a 768-dimensional vector encoding the semantic meaning of the text.

Produces:
- `data/processed/embeddings.npy` - shape (450, 768)
- `data/processed/labels.npy` - integer-encoded incident type labels

**`embeddings.npy` - shape (450, 768)**

Each row is one incident signal converted into a 768-number vector by DistilBERT. The numbers capture semantic meaning - we never read them directly, the model uses them for similarity comparison and classification.

```
row 0 (ticket: "Users cannot log in to VPN"):
[0.23, -0.81, 0.44, 0.12, ..., -0.33]  ← 768 floats

row 1 (log: "TokenValidationException: JWT expired"):
[0.21, -0.79, 0.41, 0.15, ..., -0.30]  ← 768 floats, very close to row 0
                                           because same incident, similar meaning

row 2 (alert: "auth_error_rate exceeded threshold"):
[0.19, -0.77, 0.39, 0.11, ..., -0.28]  ← 768 floats, also close

row 3 (ticket: "Packet loss on edge router"):
[-0.54, 0.33, -0.12, 0.67, ..., 0.91]  ← 768 floats, very different
                                           network issue → different region of space
```

**`labels.npy` - integer-encoded incident type**

The string labels from the CSV encoded as integers so sklearn can use them:

```
0 → authentication_failure
1 → deployment_issue
2 → network_issue

[0, 0, 0, 2, 2, 2, 1, 1, 1, ...]
 ↑ticket ↑log ↑alert  ↑ticket ↑log ↑alert
 group 1 (auth)        group 2 (network)
```

### Step 3 - Train the classifier

```bash
python src/models/classifier.py
```

**Frozen DistilBERT embeddings** means DistilBERT's weights are not updated during training - it is used purely as a feature extractor. Each text was run through it once in Step 2, the 768-dim vectors saved to disk, and DistilBERT is never touched again. The Logistic Regression trains only on those saved vectors.

1. **Load artefacts** - reads `embeddings.npy` (450 × 768) and `labels.npy` (450 integers) from disk.

2. **Stratified split** - splits 80/20 with `random_state=42`, stratified by label so all 3 classes are proportionally represented in both train and test sets.
   ```
   train: 360 rows  (120 per class)
   test:   90 rows  (30 per class)
   ```

3. **Train LogisticRegression** - fits a linear boundary in the 768-dimensional embedding space. Each class gets its own set of 768 weights that votes for or against that label. Logistic Regression trains by repeatedly adjusting its weights in small steps to reduce prediction error. **Converging** means those adjustments become so small that the algorithm decides it has found the best weights it can and stops. `max_iter=1000` is a safety cap - stop after 1000 steps even if not converged yet. On this data it converges much earlier (typically ~50–100 steps) because the 3 classes are well-separated in embedding space. If it hits the cap without converging, sklearn prints a warning indicating the model may be slightly suboptimal - but on this dataset it won't happen.

4. **Why Logistic Regression on top of DistilBERT** - DistilBERT already did the hard work of mapping text into a space where similar meanings are geometrically close. Logistic Regression just needs to draw decision boundaries between the 3 clusters - a simple task since the clusters are well-separated.

DistilBERT converts every text into a point in 768-dimensional space. Similar texts land close together - all authentication failures cluster in one region, all network issues in another, all deployment issues in a third.

After training, classifier has learned the boundaries between those regions. When we give it a new text, it checks which region the point falls in and returns that incident type.

5. **Accuracy** - measured as percentage of correct predictions (`correct / total`). Two values are logged:
   - `train_accuracy` - on the 360 rows the model trained on; expected ≥ 0.99 since it has already seen them
   - `test_accuracy` - on the 90 unseen rows; this is the number that actually matters.
   If `train_accuracy` is high but `test_accuracy` is low, the model memorised instead of learning - that's called **overfitting**. On this dataset both should be high because the 3 classes are semantically distinct.

6. **Save model** - serialises the trained `LogisticRegression` object to `models/classifier.pkl`. This is what `classify.py` loads at runtime for live inference.

6. **MLflow logging** - logs hyperparameters (`C`, `max_iter`, `test_size`) and train/test accuracy as a new run under the `incident-classifier` experiment - see the MLflow section below.

### Step 4 - Evaluate classification

```bash
python src/evaluation/eval_classifier.py
```

Reloads the saved model, evaluates on the held-out 20% test set, prints accuracy and weighted F1, saves a confusion matrix PNG to `data/processed/confusion_matrix.png`, and logs everything to MLflow.

Expected results: accuracy ≥ 0.98, weighted F1 ≥ 0.98 on this synthetic dataset (classes are semantically distinct).

`evaluate()` answers the question: **how good is the trained model?**

1. **Recreates the same test split** - uses `train_test_split` with the same `random_state=42` as training, so it gets the exact same 90 unseen rows.

2. **Makes predictions** - `clf.predict(x_test)` runs each of the 90 embeddings through the trained model and gets a predicted label for each.

3. **Compares predictions to reality** - compares `y_pred` (what the model guessed) against `y_test` (the correct answers) and computes:
   - `accuracy` - % of correct predictions overall
   - `weighted_f1` - similar to accuracy but accounts for class imbalance
   - `classification_report` - precision/recall/F1 broken down per class
   - `confusion_matrix` - a 3×3 grid showing where the model got confused (e.g. predicted network_issue when it was actually authentication_failure)

4. **Returns all results** as a dict so the caller can log them to MLflow and save the confusion matrix as a PNG.

---

## Runtime pipeline

The pipeline modules are not training scripts - they run live at inference time, called by the dashboard for each new signal.

### Classification (`src/pipeline/classify.py`)

`classify_incident(text)` takes a raw incident text string and returns the predicted incident type and confidence:

```python
classify_incident("Token validation failed for user admin")
# → {"label": "authentication_failure", "confidence": 0.97}
```

Steps:
1. **Load models once** - DistilBERT tokenizer, DistilBERT model, and the trained `classifier.pkl` are loaded on the first call and cached in memory (`@lru_cache`) so subsequent calls are fast.
2. **Embed the text** - runs the text through DistilBERT, extracts the `[CLS]` token → a (1, 768) vector.
3. **Classify** - passes the vector to the Logistic Regression model, gets probabilities for all 3 classes, picks the highest.
4. **Return** label and confidence score (the winning class probability).

---

### Priority scoring (`src/pipeline/prioritize.py`)

`score_priority(text, source_type)` is rule-based - no model involved. It scores text against two keyword lists:

- **High keywords** - `critical`, `outage`, `down`, `failed`, `crash`, `oom`, etc.
- **Medium keywords** - `warning`, `slow`, `degraded`, `latency`, `timeout`, `error`, etc.

Each source type also has a base priority:
- `alert` → medium
- `ticket` → low
- `log` → low

Final priority = whichever is higher: the keyword match or the source base priority.

```python
score_priority("CRITICAL: auth service outage", "alert")  → {"priority": "high",   "score": 0.08}
score_priority("Retry attempt 2 for DNS lookup", "log")   → {"priority": "medium",  "score": 0.03}
score_priority("Scheduled maintenance window", "ticket")  → {"priority": "low",     "score": 0.0}
```

**Escalation** - if a cluster contains signals from ≥ 2 different source types, all signals in that cluster are escalated to `high` regardless of their individual scores. A ticket + log + alert all reporting the same incident is a stronger signal than any one of them alone.

---

### Similarity and clustering (`src/pipeline/similarity.py`)

Similarity and clustering are handled by `src/pipeline/similarity.py`. The same DistilBERT embeddings computed in Step 2 are reused directly - no second model is loaded.

**How signals get grouped**

For each pair of signals, cosine similarity is computed between their embeddings. Two signals are correlated if:
- cosine similarity ≥ 0.80 (text is semantically close)
- timestamps are within 60 minutes of each other

The 60-minute constraint is enforced by `run_temporal_dbscan` - it prepends the timestamp to each embedding and uses a custom distance function that returns a distance beyond any threshold if two signals are more than 3600 seconds apart, blocking them from clustering regardless of how similar their text is. This means a network issue from Monday and a network issue from Friday will never be merged, even if they sound identical.

**DBSCAN**

DBSCAN clusters all 450 signals in one pass using `eps=0.20` (equivalent to similarity threshold 0.80, since distance = 1 − similarity) and `min_samples=2`. Signals that don't match anything above the threshold get `cluster_id = -1` - these are singletons with no correlated partners.

**Fused incident report**

When a cluster has ≥ 2 members, a fused report is produced:

```
cluster_id:   3
type:         authentication_failure   (majority vote across signals in the cluster)
priority:     high                     (escalated if ≥ 2 different source types present)
sources:      ticket, log, alert
signal_count: 3
confidence:   0.93                     (max pairwise cosine similarity in the cluster)
```

---

### Evaluation (`src/evaluation/eval_similarity.py`)
Runs DBSCAN over all embeddings, compares the resulting clusters against the ground-truth `incident_group_id` column using NMI score, and reports within-group vs cross-group cosine similarity.

**NMI (Normalized Mutual Information)** measures how well two groupings agree with each other, on a scale from 0 to 1. In this project:
- **Grouping A** - clusters produced by DBSCAN from the embeddings
- **Grouping B** - ground-truth `incident_group_id` from the CSV

**NMI = 1.0** - perfect match, every DBSCAN cluster maps exactly to one incident group  
**NMI = 0.0** - no agreement, clusters are random relative to ground truth

```
Ground truth (incident_group_id):   [1, 1, 1, 2, 2, 2, 3, 3, 3]
DBSCAN clusters:                    [A, A, A, B, B, B, C, C, C]
→ NMI ≈ 1.0  (perfect)

DBSCAN clusters:                    [A, B, C, A, B, C, A, B, C]
→ NMI ≈ 0.0  (useless - each cluster mixes all groups)
```

NMI is used instead of accuracy because accuracy requires a direct match between labels. DBSCAN names its clusters with numbers (0, 1, 2...) but those numbers are arbitrary - cluster 0 might contain authentication failures in one run and network issues in another run. So we can't check "did DBSCAN label this row 1 and does ground truth also say 1?" - the numbers mean nothing.

NMI sidesteps this. It doesn't look at label names at all. It only asks: **are the same rows grouped together?**

```
Ground truth:    [A, A, A, B, B, B]
DBSCAN:          [9, 9, 9, 3, 3, 3]   ← different numbers, same structure → NMI = 1.0
DBSCAN:          [9, 3, 9, 3, 9, 3]   ← rows are mixed up               → NMI = 0.0
```

The first case is a perfect result even though the numbers don't match.

Target metrics:
- Within-group mean cosine similarity ≥ 0.85
- Cross-group mean cosine similarity ≤ 0.50
- NMI vs ground-truth groups ≥ 0.80

---

## MLflow - experiment tracking

MLflow records every training and evaluation run so we can compare experiments over time.

### Start the UI

```bash
mlflow ui
# open http://127.0.0.1:5000
```

### What we see

One experiment named `incident-classifier` with runs inside it:

| Run name | What is logged |
|---|---|
| `logistic-regression` | **Params:** `C`, `max_iter`, `test_size`, `n_samples`, `n_classes` / **Metrics:** `train_accuracy`, `test_accuracy` / **Artefact:** serialised sklearn model |
| `evaluation` | **Metrics:** `accuracy`, `weighted_f1`, per-class `precision_*` / `recall_*` / `f1_*` for all 3 labels / **Artefact:** confusion matrix PNG |

### What MLflow is for

Every time we change a hyperparameter (e.g. try `C=0.1` or add the source-type feature engineering), re-running the training script creates a new row in the table with its own metrics. Click any two runs and press **Compare** to see a side-by-side diff of every metric and parameter. This is how we confirm whether a change actually improved the model.

---

## Streamlit dashboard - how to use it

```bash
streamlit run src/dashboard/app.py
# open http://localhost:8501
```

The first load takes 20–30 seconds while DistilBERT loads into memory. After that everything is cached until we restart.

### Tab 1 - Correlation Dashboard

The main operational view. DBSCAN has grouped the 450 signals into clusters of correlated incidents. Use the **selectbox** to browse clusters - sorted so high-priority ones appear first, with incident type and signal count in the label.

For the selected cluster:

- **Priority badge** - colour-coded: red = HIGH, orange = MEDIUM, green = LOW
- **Incident type** - majority vote across all signals in the cluster
- **Confidence** - max cosine similarity between any two signals in the cluster; higher means the model is more certain they describe the same event
- **Sources** - which source types are present (ticket, log, alert)
- **PCA scatter plot** - all 450 signals in 2D space; the selected cluster is highlighted in red so we can see whether the group is geometrically tight or scattered
- **Signal text areas** - the raw text of each signal sorted by timestamp; read these to decide if the correlation is real

At the bottom, submit analyst feedback:
- **Accept** - confirms the ML correlation is correct
- **Reject** (expandable) - pick a reason: `false_positive`, `wrong_type`, or `wrong_priority`

All feedback is persisted to `data/feedback.jsonl`. If we have already reviewed a cluster, a notice appears so we do not double-log.

### Tab 2 - Analyze Incident

Live inference for a single new signal. Type any incident text, pick source type, click **Analyze**.

Results:
- Predicted incident type and confidence percentage
- Per-class probability bar chart - useful for spotting when the model is uncertain between two classes
- Priority badge from the rule-based keyword scorer
- Table of the 8 most similar rows from the dataset (cosine similarity ≥ 0.80) - shows what known incidents our new signal resembles

### Tab 3 - Dataset

Static view of the 450 synthetic training rows. Three distribution charts (by incident type, source type, priority) and a filterable table. Useful for understanding what the model was trained on and verifying data balance.

### Tab 4 - Feedback Log

Running audit trail of every Accept/Reject decision with timestamps. Shows total accepted and rejected counts. Includes a **Clear all feedback** button that wipes `data/feedback.jsonl`.

---

## Full picture

```
generate.py → synthetic_incidents.csv
                       ↓
embed.py    → embeddings.npy + labels.npy
                       ↓
classifier.py → classifier.pkl ──────────→ MLflow (params + metrics + model artefact)
                       ↓
eval_classifier.py ───────────────────────→ MLflow (F1, per-class metrics, confusion matrix PNG)

At runtime (dashboard):
  user input text
         ↓
  classify.py   → DistilBERT embed → classifier.pkl → incident type + confidence
  prioritize.py → keyword scoring  → priority + score
  similarity.py → cosine search    → similar incidents + DBSCAN clusters
         ↓
  app.py (Streamlit) → Correlation Dashboard, Analyze Incident, Dataset, Feedback Log
                                          ↓
                                   feedback.jsonl
```

---

## Project structure

```
ml-based-incidents-intelligence-system/
├── data/
│   ├── raw/
│   │   └── synthetic_incidents.csv       # generated by generate.py
│   ├── processed/
│   │   ├── embeddings.npy                # (450, 768) DistilBERT CLS vectors
│   │   ├── labels.npy                    # integer-encoded labels
│   │   └── confusion_matrix.png          # saved by eval_classifier.py
│   └── feedback.jsonl                    # analyst accept/reject decisions
├── models/
│   └── classifier.pkl                    # trained LogisticRegression
├── mlruns/                               # MLflow tracking store (do not commit)
├── src/
│   ├── data/generate.py                  # synthetic data generation
│   ├── features/embed.py                 # DistilBERT embedding extraction
│   ├── models/classifier.py              # LogisticRegression training
│   ├── evaluation/
│   │   ├── eval_classifier.py            # classification metrics
│   │   └── eval_similarity.py            # NMI + cosine similarity metrics
│   ├── pipeline/
│   │   ├── classify.py                   # incident type inference
│   │   ├── prioritize.py                 # rule-based priority scoring
│   │   └── similarity.py                 # cosine similarity + DBSCAN
│   └── dashboard/
│       └── app.py                        # Streamlit UI
├── requirements.txt
├── CLAUDE.md
└── prototype.md                          # this file
```

---

## Limitations

**No singleton signals in the dataset**
Every row in the generated data belongs to an incident group of exactly 3 - one ticket, one log, one alert. Every single row belongs to an incident_group_id with exactly one ticket, one log, and one alert. There are no singleton signals in the dataset. In a real system the majority of signals would be noise: routine log entries, one-off alerts that resolve on their own, tickets unrelated to anything else. The DBSCAN threshold and escalation logic have never been tested against this scenario, so false positive clustering (grouping unrelated signals together) is not measured.

**Synthetic data only**
The classifier and similarity model are trained and evaluated on data generated by the same templates they were designed around. Performance on real incident data from ServiceNow, production logs, or monitoring systems is unknown.

**Embeddings are not discriminative enough between different incidents**
Within-group mean cosine similarity is 0.93 (signals from the same incident are similar - good) but cross-group mean cosine similarity is 0.92 (signals from different incidents are almost equally similar - bad). The gap between the two is nearly zero, meaning DistilBERT embeddings alone cannot reliably separate different incidents from each other. Plain DBSCAN without a time constraint merges everything into one cluster because of this. The temporal constraint in `run_temporal_dbscan` compensates by blocking incidents more than 60 minutes apart from clustering together, which is what produces correct results (NMI 0.88). In a real system with incidents happening at similar times, this lack of embedding discrimination would cause false positive correlations.

**`correlate` function sits in an awkward middle ground**
`correlate` does two things: find similar incidents within the 60-minute window, and look up the pre-assigned cluster ID from DBSCAN. But point 2 only works if DBSCAN has already been run on the full dataset beforehand and its results (`cluster_assignments`) are passed in. So `correlate` doesn't determine the cluster ID itself, it just reads it from a pre-computed array. That makes it a hybrid: half real-time lookup (find similar incidents), half batch result reader (what cluster did DBSCAN already put this in?). If we're running DBSCAN in batch anyway, we already have all the cluster assignments and don't need to look them up per-incident. And if we're doing real-time streaming, we don't have pre-computed DBSCAN results to read from. It sits in an awkward middle ground between the batch approach (`build_fused_reports`) and a pure real-time approach (`find_similar`).

**Batch-only correlation, no real-time streaming**
The dashboard loads all 450 signals at once and runs DBSCAN in a single batch. There is no streaming ingestion - new signals arriving one at a time cannot be slotted into an existing cluster on the fly. A real system would need a per-incident lookup (compare the new signal against recent incidents within the time window and assign it to an existing cluster or open a new one). The batch approach is sufficient for a prototype but would not scale to a live incident feed.

**All clusters show high priority and high confidence**
Because every incident group in the synthetic dataset contains exactly one ticket, one log, and one alert, every cluster will always have ≥ 2 source types present - triggering escalation to `high` for all clusters. Similarly, confidence scores are artificially high because synthetic texts within a group share injected identifiers (service name, error code, IP) that DistilBERT picks up strongly. In a real system we would expect a spread of priorities and lower confidence scores for ambiguous signals.


**The embeddings are not discriminative enough between different incidents**
Within-group: 0.93 - good. Signals from the same incident (ticket + log + alert) are very similar to each other in embedding space. DistilBERT is picking up that they describe the same event.

Cross-label: 0.92 - this is the problem. Signals from different incidents are almost as similar to each other as signals from the same incident. There's barely any gap (0.93 vs 0.92). Ideally cross-label should be much lower (target was ≤ 0.50) - meaning different incidents should look clearly different to the model.

This explains why plain DBSCAN merged everything into one cluster - from the model's perspective, everything looks nearly the same.

NMI: 0.88 - good. Despite the similarity problem above, the temporal constraint saves it. By limiting clustering to the 60-minute window, the temporal DBSCAN still correctly recovers the incident groups.

Bottom line: the embeddings are not discriminative enough between different incidents - the synthetic texts across different groups are too similar in semantic space. The temporal constraint is doing most of the heavy lifting to get correct clusters.

**The timestamp collision**
The time window is doing the real clustering work. Without it, DBSCAN would likely merge most of the 450 signals into a handful of giant clusters (or one) because the cosine distances between all IT-ops texts are small enough to fall within eps=0.20.

The current 50 clusters of 9 signals: 50 time slots × (3 label classes × 3 signals) = 50 × 9 = 450. The timestamp collision is merging 3 incident groups (one per label class) into each cluster.

DBSCAN clusters by both time proximity AND cosine similarity. Since the 9 signals all land in the same 60-minute window, the temporal gate passes them all as candidates. Then it checks if their embeddings are within cosine distance 0.20 (similarity ≥ 0.80).

The issue: DistilBERT without fine-tuning produces embeddings where all these IT-ops texts sit fairly close together in embedding space - they all share vocabulary like ERROR, ALERT:, failed, service, exceeded threshold. The model never learned to separate "JWT expired" from "packet loss" at a fundamental semantic level without task-specific training.

So signals from 3 different incident types end up within 0.20 cosine distance of each other → they all merge into one cluster.

**The time window does the real clustering work, not the embeddings**
The temporal constraint in `run_temporal_dbscan` is what actually separates clusters - not embedding similarity. DistilBERT without fine-tuning produces embeddings where all IT-ops texts sit close together in embedding space because they share the same vocabulary (`ERROR`, `ALERT:`, `failed`, `service`, `exceeded threshold`). It never learned to separate "JWT expired" from "packet loss" at a fundamental level without task-specific training. As a result, cosine similarity between signals from entirely different incident types is nearly as high as between signals from the same incident (within-group: 0.93, cross-group: 0.92 - a gap of only 0.01). Without the 60-minute time gate, DBSCAN would merge most of the 450 signals into one or a few giant clusters. The time window is what breaks the dataset into meaningful groups; embedding similarity is only confirming what the time window already selected. To make embeddings do real discriminative work, fine-tuning on labeled incident data or a contrastive training objective (pulling same-class embeddings together and pushing different-class embeddings apart) would be required.

---

## Addressing Limitations

### 1. Discriminative embeddings via DistilBERT fine-tuning

**Problem:** with the base `distilbert-base-uncased` model, within-group cosine similarity was 0.93 and cross-group cosine similarity was 0.92 - a gap of just 0.01. The embedding space had no meaningful separation between incident types. All IT-ops texts share vocabulary (`ERROR`, `ALERT:`, `failed`, `exceeded threshold`) so the base model placed them all in the same neighbourhood regardless of incident class. DBSCAN's cosine similarity check was therefore useless - it was the 60-minute time window doing all the clustering work. The result was 50 clusters of 9 mixed signals (3 incident groups × 3 signals, one per class, sharing the same time slot) instead of 150 clusters of 3 same-class signals.

**What was done:** `distilbert-base-uncased` was fine-tuned on the 450 labeled samples using a 3-class classification head (`network_issue`, `authentication_failure`, `deployment_issue`) on Google Colab T4 GPU. Fine-tuning adds a classification head on top of the backbone and trains the entire model end-to-end with cross-entropy loss. The gradient from the loss propagates back through all 66M DistilBERT parameters, adjusting every weight so that the CLS token embedding for each text shifts toward a region of the 768-dimensional space that is distinct for its incident class. This is fundamentally different from the Logistic Regression trained earlier - that classifier drew a boundary in the existing embedding space but could not move the embeddings. Fine-tuning moves the embeddings themselves.

**Why the same 450 samples are sufficient:** DistilBERT was pre-trained on Wikipedia and books, giving it general language understanding. It does not need to re-learn language from scratch - it only needs to learn the relatively small adjustment: map "JWT expired" and "token validation failed" toward one region, "packet loss" and "BGP dropped" toward another, "OOMKilled" and "rollback triggered" toward a third. 450 examples with clear class boundaries is enough for this narrow task. Training converges in 10 epochs (~3 minutes on T4).

**Results after fine-tuning:**

```
Metric                                       Base   Fine-tuned
--------------------------------------------------------------
Within-group cosine similarity             0.9310       0.9361
Cross-group cosine similarity              0.9239      -0.0377
Discrimination gap (within - cross)        0.0071       0.9739
--------------------------------------------------------------
Gap improvement: 137x
```

Cross-group similarity dropped from 0.92 to **-0.04** - negative cosine similarity means signals from different incident types now point in opposite directions in embedding space. They are not just separated; they actively repel each other. Within-group similarity stayed at 0.93, meaning signals from the same real incident (ticket + log + alert) are still tightly grouped.

**Impact on clustering:** with fine-tuned embeddings, signals from different incident classes that happen to fall in the same 60-minute window have a cosine distance of ~1.04 (1 − (−0.04)), which far exceeds the DBSCAN eps of 0.20. They no longer cluster together. The result is ~150 clusters of 3 same-type signals each - one cluster per real incident group - without any change to timestamps or DBSCAN parameters. The timestamp collision that previously forced 9 signals into one cluster is self-corrected because the embedding check now actively blocks cross-class clustering.

**Fine-tuning configuration:**
- Base model: `distilbert-base-uncased`
- Classification head: linear layer → 3 output classes
- `lr = 2e-5`, `epochs = 10`, `batch_size = 16`, `weight_decay = 0.01`
- HuggingFace `Trainer` API
- Only the backbone weights are saved (not the classification head) - `embed.py` loads these to extract CLS embeddings

**How to reproduce (Google Colab T4):**

```
Step 1 - open research/finetune_distilbert.ipynb in Colab
         set runtime: Runtime → Change runtime type → T4 GPU
Step 2 - run the install cell, then Runtime → Restart session from the menu
Step 3 - upload data/raw/synthetic_incidents.csv when prompted
Step 4 - run all remaining cells (~3 min on T4)
         Step 6 prints the comparison table - confirm the gap is large
Step 5 - Step 7 downloads distilbert-finetuned.zip
Step 6 - extract the zip, place the folder at models/distilbert-finetuned/
Step 7 - run locally: python src/features/embed.py --model finetuned
         this saves data/processed/embeddings_finetuned.npy
Step 8 - open the dashboard - the model toggle appears in the sidebar
```

**Dashboard model toggle:**

The dashboard sidebar has a radio button: **Base DistilBERT (no fine-tuning)** / **Fine-tuned DistilBERT**. Switching reloads embeddings from `embeddings_base.npy` or `embeddings_finetuned.npy` and re-runs DBSCAN. The toggle is only shown when `embeddings_finetuned.npy` exists; otherwise the sidebar shows instructions for running the Colab notebook. This allows side-by-side comparison during a demo:

| Model | Clusters | Signals per cluster | Incident types per cluster |
|---|---|---|---|
| Base DistilBERT | 50 | 9 (mixed) | 3 (all classes merged) |
| Fine-tuned DistilBERT | ~150 | 3 | 1 (correctly separated) |

**`embed.py` model flag:**

```bash
python src/features/embed.py                  # base model → embeddings_base.npy
python src/features/embed.py --model finetuned  # fine-tuned  → embeddings_finetuned.npy
```

Both commands also copy their output to `embeddings.npy` for backward compatibility with the classifier and evaluation scripts.

---

### 2. Singleton signals in the dataset *(next step - not yet implemented)*

**Problem:** every row in the dataset belongs to a 3-signal incident group. DBSCAN's noise detection (cluster_id = -1) is never exercised - there is nothing to reject. In a real system the majority of signals are noise: routine health checks, scheduled maintenance alerts, one-off logs that resolve on their own.

**Fix:** Add ~12 singleton signals to `src/data/generate.py` via a `generate_singletons()` function. These are benign, standalone texts with no related signals:

```
"Scheduled maintenance window starting at 02:00 UTC"
"Health check passed for payment-api - all replicas healthy"
"Daily backup completed successfully for db-primary"
"Certificate renewed successfully for api.internal"
"Disk usage on log-archive-01 at 74% - within normal range"
```

Schema: `incident_group_id = -1`, `label = "none"`, `priority = "low"`, `source_type` varied, timestamps spread across the 60-day range so they don't accidentally land inside a real incident's time window.

**Expected outcome:** all 12 singletons appear as cluster_id = -1 in DBSCAN output. `eval_similarity.py` reports a singleton isolation metric: `12/12 singletons correctly isolated as noise`. This demonstrates that DBSCAN's noise detection works and is not just a theoretical claim.