# Prototype Architecture

> Render with the [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) VS Code extension, or push to GitHub — both render Mermaid natively.

---

## Training pipeline

Runs once. Each step produces a file the next step depends on.

```mermaid
flowchart TD
    A([generate.py]) --> B[(synthetic_incidents.csv\n450 rows · 50 incident groups)]

    B --> C([embed.py])
    C --> D[(embeddings.npy\n450 × 768)]
    C --> E[(labels.npy\ninteger-encoded)]

    D --> F([classifier.py])
    E --> F
    F --> G[(classifier.pkl\nLogisticRegression)]
    F --> H[[MLflow\nparams · train/test accuracy · model artefact]]

    D --> I([eval_classifier.py])
    G --> I
    I --> J[[MLflow\nF1 · per-class precision/recall · confusion matrix PNG]]

    D --> K([eval_similarity.py])
    K --> L[[NMI vs ground-truth groups\nwithin-group · cross-group cosine similarity]]

    subgraph OPT [Optional — Google Colab T4]
        M([finetune_distilbert.ipynb]) --> N[(models/distilbert-finetuned/\nfine-tuned backbone weights)]
        N -->|--model finetuned flag| C
    end

    style OPT fill:#f5f5f5,stroke:#bbb,stroke-dasharray:5
    style H fill:#fff3cd,stroke:#e6ac00
    style J fill:#fff3cd,stroke:#e6ac00
    style L fill:#fff3cd,stroke:#e6ac00
```

---

## Runtime pipeline

Called live by the dashboard for each new signal or on page load.

```mermaid
flowchart TD
    INPUT([User input text\n+ source type]) --> CL([classify.py])
    INPUT --> PR([prioritize.py])
    INPUT --> SIM([similarity.py])

    DISTIL[(distilbert-base-uncased\nor distilbert-finetuned)] -->|embed text → 768-dim vector| CL
    PKL[(classifier.pkl)] --> CL
    EMB[(embeddings.npy\n450 stored vectors)] --> SIM

    CL --> R1[incident type\n+ confidence score]
    PR --> R2[priority: high / medium / low\n+ keyword score]
    SIM --> R3[top-8 similar incidents\n+ DBSCAN cluster assignment]

    R1 --> APP([app.py · Streamlit])
    R2 --> APP
    R3 --> APP

    APP --> T1[Tab 1\nCorrelation Dashboard]
    APP --> T2[Tab 2\nAnalyze Incident]
    APP --> T3[Tab 3\nDataset]
    APP --> T4[Tab 4\nFeedback Log]

    T1 -->|accept / reject| FB[(feedback.jsonl)]
```

---

## Full picture

Both pipelines together, showing how training artefacts feed the runtime layer.

```mermaid
flowchart TD
    subgraph TRAIN [Training — runs once]
        direction TB
        G1([generate.py]) --> CSV[(synthetic_incidents.csv)]
        CSV --> EM([embed.py])
        EM --> NPY[(embeddings.npy)]
        EM --> LBL[(labels.npy)]
        NPY --> CLS([classifier.py])
        LBL --> CLS
        CLS --> PKL[(classifier.pkl)]
        CLS --> MLF1[[MLflow run: logistic-regression]]
        NPY --> EVAL([eval_classifier.py])
        PKL --> EVAL
        EVAL --> MLF2[[MLflow run: evaluation]]
        NPY --> EVALSIM([eval_similarity.py])
        EVALSIM --> NMI[[NMI · cosine similarity report]]

        subgraph COLAB [Optional · Google Colab T4]
            FT([finetune_distilbert.ipynb]) --> FTW[(distilbert-finetuned/)]
        end
        FTW -.->|--model finetuned| EM
    end

    subgraph RUNTIME [Runtime — per request]
        direction TB
        IN([incident text\n+ source type])
        IN --> CLASSIFY([classify.py])
        IN --> PRIO([prioritize.py])
        IN --> SIM([similarity.py])
        CLASSIFY --> OUT1[type + confidence]
        PRIO --> OUT2[priority + score]
        SIM --> OUT3[similar incidents\n+ cluster]
        OUT1 & OUT2 & OUT3 --> DASH([app.py · Streamlit])
        DASH --> FB[(feedback.jsonl)]
    end

    PKL -->|loaded once, cached| CLASSIFY
    NPY -->|cosine search| SIM

    style COLAB fill:#f5f5f5,stroke:#bbb,stroke-dasharray:5
    style MLF1 fill:#fff3cd,stroke:#e6ac00
    style MLF2 fill:#fff3cd,stroke:#e6ac00
    style NMI fill:#fff3cd,stroke:#e6ac00
    style FB fill:#d4edda,stroke:#28a745
```
