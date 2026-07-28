# FEMTO Predictive Maintenance — Project Execution Plan

*4-person team, 3-4 week build*

## Team roles

The four MLOps lifecycle stages in the project brief map naturally onto four owners. Each person leads one stage end-to-end but stays involved in integration points with neighboring stages — this isn't a "build in isolation" split, since each stage's output is the next stage's input.


| Role                            | Owns                                                               | Also collaborates on                                             |
| ------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| **A — Data & Features (Grace)** | Ingestion, RUL labeling, feature extraction, EDA, train/test split | Feeds format/schema to Person B's orchestrator                   |
| **B — Pipeline & Tracking**     | Airflow/Prefect DAG, MLflow experiment tracking, model registry    | Consumes A's feature pipeline, hands registered model to C       |
| **C — Deployment**              | Docker + FastAPI inference service                                 | Consumes B's registered model, hands live endpoint to D          |
| **D — Monitoring & Drift**      | EvidentlyAI dashboard, drift simulation, anomaly verification      | Hits C's live endpoint, owns final documentation/slides assembly |


---



## Week 1 — Foundations

**Goal: clean, labeled, feature-engineered data with a baseline model, sitting in a shared repo.**

- [ ] Repo scaffolding: folder structure, `requirements.txt`/environment file, README skeleton, branch strategy (A leads, whole team reviews)
- [ ] Download FEMTO data, verify integrity, set up shared storage (cloud bucket or shared drive — don't rely on local laptops only) (A)
- [ ] RUL label construction — decide and document the labeling scheme (life percentage vs. raw time-to-failure) (A)
- [ ] Feature extraction script — time-domain (RMS, kurtosis, skewness, crest factor) + frequency-domain (FFT band energies) (A, with B reviewing output schema)
- [ ] EDA notebook — class/condition distributions, degradation trend visualizations, justify the RMSE + PHM12 scoring metric choice (A, D contributes drift-relevant EDA)
- [ ] Bearing-level train/test split, isolate test set (A)
- [ ] Baseline model (simple linear/tree regression, no tuning) to sanity-check the pipeline end-to-end (A + B)
- [ ] Kickoff meeting: confirm roles, tooling choices (Airflow vs Prefect, Docker+FastAPI vs BentoML), meeting cadence (whole team)

**End of week 1 checkpoint:** everyone can run the feature extraction script locally and get the same output. This is the single most important sync point — if schemas don't match, week 2 stalls.

---



## Week 2 — Orchestration & Experiment Tracking

**Goal: automated pipeline producing a versioned, tracked model in a registry.**

- [ ] Stand up Airflow/Prefect locally (or lightweight cloud instance) (B)
- [ ] Build DAG: ingest → feature extraction → train → evaluate → register, wrapping A's week-1 scripts as tasks (B)
- [ ] Set up MLflow tracking server, log runs (params, metrics, artifacts) (B)
- [ ] Run hyperparameter sweep / AutoML (Optuna, FLAML, or MLflow's built-in tools) across XGBoost/LightGBM/Random Forest (B, A helps interpret results)
- [ ] Optional stretch: pretrain on conditions 1+2, fine-tune on condition 3 (transfer learning) — only if core pipeline is ahead of schedule (A + B)
- [ ] Register best model to MLflow Model Registry with semantic versioning (B)
- [ ] Start Dockerfile / API skeleton in parallel so C isn't blocked waiting for a "final" model (C)
- [ ] Start monitoring dashboard skeleton against baseline test data in parallel (D)

**End of week 2 checkpoint:** a registered model artifact exists that C can pull into a container. Confirm the model's expected input format matches what C is building the API around.

---



## Week 3 — Deployment & Monitoring Integration

**Goal: a live, containerized inference endpoint with a working monitoring baseline.**

- [ ] Build FastAPI service wrapping the registered model, containerize with Docker (C)
- [ ] Test endpoint locally with sample requests, confirm real-time prediction latency is reasonable (C)
- [ ] Deploy container (local Docker, or cloud if your course expects it) (C)
- [ ] Set up EvidentlyAI (or chosen tool) monitoring baseline using the clean test set run through the live API (D, needs C's endpoint live)
- [ ] Baseline validation: pass clean test data through, confirm dashboard reflects expected "no drift" state (D)
- [ ] Begin drift simulation design: decide corruption strategies (out-of-bounds amplitude spikes, channel swaps, schema changes) (D + A, since A understands the feature semantics best)
- [ ] Mid-week integration check: full pipeline dry run, A→B→C→D, fix any broken handoffs (whole team)

**End of week 3 checkpoint:** clean data flows all the way from raw files to a monitored prediction with no manual intervention. This is your "happy path" — get this rock solid before introducing drift.

---



## Week 4 — Drift Testing, Docs, and Presentation

**Goal: documented drift detection, polished repo, rehearsed presentation.**

- [ ] Run drift simulation against live endpoint, capture dashboard's anomaly response (screenshots/recordings for slides) (D)
- [ ] Tune alerting thresholds if the dashboard doesn't flag corruption clearly enough (D)
- [ ] Finalize README: setup instructions, architecture overview, how to reproduce locally (D coordinates, everyone writes their section)
- [ ] Code cleanup pass: comments, remove dead code, confirm requirements file is accurate (whole team, one PR each)
- [ ] Build slide deck: problem/EDA (A), architecture + tracking (B), deployment + monitoring (C), drift analysis (D) — each person builds their own section
- [ ] Full dry-run presentation (10-15 min), time it, prep for Q&A (whole team)
- [ ] Confirm each member can clearly state their individual contribution for the presentation requirement (whole team)

**End of week 4:** repo is clean and reproducible, slides are done, everyone can speak to both their own stage and the pipeline as a whole.

---



## Risk notes

- **Biggest risk is the week 1→2 handoff** (feature schema mismatch between A and B). Lock the feature output format (column names, types, units) in writing by end of week 1, even if it changes later.
- **If the transfer learning stretch goal starts eating into week 3**, drop it — a working core pipeline beats an ambitious model with a broken deployment/monitoring stage. The rubric weighs all four MLOps stages, not just modeling.
- **Build the "unhappy path" (drift simulation) as early as week 2's spare cycles if possible** — D shouldn't be blocked until week 4 to even think about what corruption strategies to use.
- **Keep a shared doc of decisions** (labeling scheme, metric choice, tool versions) as you go, so the README in week 4 is assembly, not a scramble to remember what you did in week 1.

