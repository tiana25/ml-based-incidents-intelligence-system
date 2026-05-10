# Decisions, Lessons Learned, and Potential Improvements

---

## Design Decisions

### 1. DistilBERT as the backbone, not a larger model
We chose `distilbert-base-uncased` over larger alternatives (BERT-large, RoBERTa, GPT-based models) because the prototype runs entirely on CPU locally and must complete the full pipeline in under 5 minutes. DistilBERT is 40% smaller and 60% faster than BERT-base while retaining 97% of its performance on NLP benchmarks. For a prototype with 450 synthetic samples this is the right trade-off: it demonstrates the transformer-based approach without requiring GPU infrastructure for inference.

### 2. Frozen embeddings + Logistic Regression for classification, not end-to-end fine-tuning from the start
The classifier (Step 3) trains a Logistic Regression on top of frozen DistilBERT embeddings rather than fine-tuning the full model. This was the right starting point for a prototype: it is fast (trains in seconds on CPU), interpretable, and reveals a key insight - the classifier achieves ≥ 0.98 F1 even on frozen embeddings because the classes are semantically distinct enough for a linear boundary. The limitation this exposed (embeddings not discriminative for similarity) led directly to the fine-tuning decision later.

### 3. Rule-based priority scoring, not a trained model
Priority (high / medium / low) is scored by keyword matching rather than a trained classifier. The reason: training a priority classifier on synthetic data generated from the same keyword rules would be circular - the model would simply re-learn the rules it was trained from, adding no value. Rule-based scoring is honest, fully interpretable, and sufficient for a prototype. It also makes the scoring logic auditable, which matters in an operational context where engineers need to trust and override decisions.

### 4. DBSCAN for clustering, not K-Means or hierarchical clustering
DBSCAN was chosen because it does not require specifying the number of clusters in advance and it natively labels outliers as noise (cluster_id = -1). In a real incident system the number of active incidents at any moment is unknown. K-Means requires k upfront; hierarchical clustering does not handle noise well. DBSCAN's eps parameter (cosine distance threshold) maps directly onto the similarity threshold from the specification (0.80), making it interpretable.

### 5. Temporal constraint baked into DBSCAN, not as a pre-filter
The 60-minute time window is enforced inside a custom distance function passed to DBSCAN rather than as a separate pre-filtering step. This keeps the clustering logic in one place and means DBSCAN reasons about both dimensions simultaneously. The trade-off is that DBSCAN with a custom metric uses `algorithm="ball_tree"` which is slower than the default for high-dimensional data - acceptable for 450 samples, not for millions.

### 6. Batch DBSCAN over the full dataset, not real-time per-signal correlation
The dashboard runs DBSCAN once over all 450 signals in a single batch. This was a deliberate prototype simplification - it avoids the complexity of maintaining a running cluster state and updating assignments as new signals arrive. The batch approach is sufficient to demonstrate the concept but would not work in a live production feed.

### 7. Fine-tuning done externally on Colab, not as part of the local pipeline
Fine-tuning DistilBERT requires GPU to complete in reasonable time (3 min on T4 vs ~15 min on CPU). Rather than requiring GPU infrastructure as a dependency, the fine-tuning is a one-off step done in `research/finetune_distilbert.ipynb` on Google Colab. The output (backbone weights) is saved locally and the rest of the pipeline runs on CPU. This keeps the core pipeline hardware-independent while still enabling the improvement.

---

## Lessons Learned

### 1. Pre-trained embeddings are not always discriminative for domain-specific tasks
The most significant finding: `distilbert-base-uncased` without fine-tuning produced embeddings where the within-group cosine similarity (0.93) and cross-group cosine similarity (0.92) were nearly identical - a gap of just 0.01. All IT-ops texts share vocabulary (`ERROR`, `ALERT:`, `failed`, `exceeded threshold`) and the base model placed them all in the same neighbourhood of embedding space regardless of incident class. This meant the cosine similarity check in DBSCAN was doing almost nothing - the time window was the only thing separating clusters. This would be a serious problem in production where unrelated incidents happen simultaneously.

**Key takeaway:** using a pre-trained model as a feature extractor without any domain adaptation is not sufficient when the downstream task requires fine-grained semantic discrimination within a narrow domain. General language models encode general semantic meaning; they need task-specific signal to encode task-specific meaning.

### 2. Fine-tuning on a small dataset is highly effective when the task is narrow
After fine-tuning on the same 450 samples with a 3-class objective, cross-group similarity dropped from 0.92 to -0.04 - a 137x improvement in the discrimination gap. Negative cosine similarity means signals from different incident types now point in opposite directions in the 768-dimensional space. This confirms that DistilBERT already had the representational capacity to separate the classes; it just needed the right training signal to activate it. A narrow, well-defined 3-class problem with clearly distinct texts is an easy fine-tuning target even with limited data.

### 3. The clustering result depends more on the data distribution than on the algorithm
Before fine-tuning, the synthetic data generation assigned the same time slots to all three label classes (network_issue group 0, auth_failure group 0, and deployment_issue group 0 all started at day 0, hour 0). This caused 3 incident groups to fall in every 60-minute window, producing 50 clusters of 9 mixed signals instead of 150 clusters of 3. Fixing the algorithm parameters would not have solved this - the root cause was in the data. After fine-tuning made the embeddings discriminative, the same DBSCAN parameters with the same timestamps naturally produced the correct 150 clusters. The lesson: always inspect what the clustering algorithm is actually doing, not just whether it runs.

### 4. Evaluation metrics can be misleading if the data has structural problems
NMI of 0.88 looked like a success before fine-tuning - the clusters appeared to match ground-truth groups. But the high NMI was partly a result of the data structure (consistent time slot collisions) rather than genuine semantic clustering. The within/cross-group similarity gap of 0.01 exposed the real problem. A single headline metric (NMI) masked a fundamental limitation that only became visible when examining the embedding space directly.

### 5. The `correlate` function is architecturally misaligned
`correlate()` is a hybrid that does two things: find similar incidents within the time window (real-time), and read back a pre-computed DBSCAN cluster ID (batch). These two approaches are incompatible in production. If DBSCAN is run in batch, the cluster IDs are already known for all signals and `correlate` is redundant. If a new signal arrives in real-time, there is no pre-computed DBSCAN to look up. The function was kept because it is part of the prototype specification, but it represents a design that would need to be resolved before productionisation.

---

## Potential Improvements

### Embeddings and model

**Fine-tune with contrastive loss instead of classification loss**
Classification fine-tuning optimises for label prediction, which indirectly improves embedding separation. Contrastive training (e.g. SimCSE, triplet loss) directly optimises cosine similarity: same-class pairs are pulled together, different-class pairs are pushed apart. This would likely produce an even larger discrimination gap and would not require class labels - useful if incident types are not well-defined in advance.

**Use a domain-adapted base model**
`distilbert-base-uncased` was pre-trained on Wikipedia and books. A model pre-trained on IT operations text (server logs, incident reports, monitoring data) would already have better representations before fine-tuning. Models like `microsoft/codebert-base` or a log-specific BERT variant would be worth evaluating.

**Normalise embeddings before similarity computation**
Cosine similarity is equivalent to dot product on L2-normalised vectors. Explicitly normalising embeddings before storing them would make the similarity computation faster and ensure consistent behaviour across different model outputs.

### Clustering and correlation

**Real-time streaming correlation**
Replace the batch DBSCAN with a per-signal lookup: when a new signal arrives, compare its embedding against all signals within the last 60 minutes, assign to the best-matching existing cluster or open a new one. This is the architecture needed for a live incident feed.

**Adaptive time window**
The 60-minute window is fixed. Different incident types propagate at different speeds: a network outage generates correlated signals within minutes; a slow memory leak might generate correlated signals over hours. A learned or per-type time window would improve recall.

**Hierarchical clustering for large-scale datasets**
DBSCAN with a custom metric scales poorly to large datasets (O(n²) comparisons with `ball_tree` and a non-Euclidean metric). For production scale, approximate nearest-neighbour search (FAISS, ScaNN) combined with a graph-based clustering approach would be needed.

### Data and evaluation

**Real incident data**
All evaluation is on synthetic data generated from the same templates used to design the system. Performance on real ServiceNow tickets, production logs, or PagerDuty alerts is unknown. The most important next step before any production claim is evaluation on real data, ideally with human-annotated correlation ground truth.

**Singleton signals**
The current dataset has no standalone signals - every row belongs to a 3-signal incident group. DBSCAN's noise detection (cluster_id = -1) is therefore never stress-tested. Adding ~12 benign, standalone signals (routine health checks, scheduled maintenance, backup completions) with `incident_group_id = -1` would verify that DBSCAN correctly isolates them as noise rather than forcing them into a cluster.

**Class imbalance and partial incident groups**
In reality, not every incident generates all three signal types. A network outage might produce only an alert and a log, with no ticket opened. The model has never seen a 2-signal or 1-signal incident group. Testing partial groups would reveal whether the priority escalation and cluster confidence logic degrades gracefully.

### Priority and escalation

**Source-type weighting based on reliability**
Alerts are pre-filtered signals (a human set the threshold), tickets are human-authored, logs are raw and noisy. The current escalation logic treats all source types equally once 2+ are present. A weighted scheme - where an alert+ticket combination carries more weight than a log+log - would produce more nuanced priority scores.

**Feedback loop integration**
The dashboard collects analyst accept/reject decisions in `feedback.jsonl`. Currently this data is never used. In a production system, accepted correlations would be positive examples and rejected correlations would be negative examples for a re-training loop that improves the model over time.
