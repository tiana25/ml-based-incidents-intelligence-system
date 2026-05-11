# Next Steps: Moving to a Real System

The prototype is complete and working. Here's how I'd approach moving this into a real, production-grade system - roughly in the order I'd tackle it.


## 1. Replace synthetic data with real incident data

This is the most important step and the biggest unlock. The pipeline is sound, but the model's quality depends entirely on real signal text.

- Connect to actual ServiceNow, Splunk/ELK logs, PagerDuty/Prometheus alerts
- Build an ingestion adapter per source and normalize to the existing schema: `source_type`, `text`, `timestamp`
- The rest of the pipeline works unchanged - the only thing that changes is the input

## 2. Fine-tune DistilBERT on real labeled incidents

Right now the classifier uses frozen DistilBERT embeddings + Logistic Regression trained on synthetic text. With real data I'd:

- Gather a few hundred labeled real incidents - SREs can label a backlog
- Fine-tune DistilBERT end-to-end on the 3-class problem - this will significantly outperform the frozen-embedding + LR approach
- The `src/features/embed.py` + `src/models/classifier.py` structure is already the right separation; I'd replace the frozen model with a fine-tuned one

## 3. Add a real-time ingestion loop

Right now the pipeline is batch - run once on a CSV. For production I'd need:

- A queue or stream (Kafka, Redis Streams, or even a simple REST endpoint) that accepts incoming signals
- The pipeline triggered per-event, not per-file
- The sliding window in `find_similar()` would need to query a store (Redis/Postgres), not an in-memory array

## 4. Persist state and serve results

- Store incident clusters, classifications, and priorities in a database - Postgres is a natural fit
- The `src/dashboard/app.py` is a starting point; it needs to read from the database, not from local files
- Expose a REST API (FastAPI is a good minimal choice) so other systems can query active incidents

## 5. Tune the correlation thresholds on real data

The `similarity ≥ 0.80` and `60-minute window` thresholds were calibrated for synthetic data. With real incidents I'd validate these against actual SRE feedback - too low and you get false groupings, too high and real correlations are missed.

## 6. Add observability and a feedback loop

- Log every classification decision with its confidence score
- Track when SREs override or reject a correlation - that's the training signal for retraining
- Add a model staleness check: if real-world accuracy drops, trigger a retrain
