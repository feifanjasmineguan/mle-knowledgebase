# Real-Time Machine Learning: Challenges and Solutions

## Summary

Real-time machine learning is the approach of using real-time data to generate more accurate predictions and adapt models to changing environments. Rather than precomputing predictions in batch, real-time ML systems generate predictions on-demand after requests arrive, incorporating fresh data and enabling continuous model adaptation. The field encompasses two main challenges: **(1) online prediction** — generating predictions with fresh, real-time data; and **(2) continual learning** — updating models in response to changing data distributions rather than on fixed schedules. This is fundamentally an infrastructure problem requiring mature streaming systems, feature stores, and model management solutions.

## Key Concepts

### Online Prediction Stages

**Stage 1: Batch Prediction**
- Predictions precomputed at fixed intervals (e.g., every 4 hours or daily)
- All computation happens offline; predictions retrieved at request time
- Typical use cases: collaborative filtering, recommendations (Netflix, DoorDash, Reddit)
- Limitation: Cannot personalize for new/unlogged-in users; predictions become stale between batch jobs
- Legacy approach; not a prerequisite for online prediction

**Stage 2: Online Prediction with Batch Features**
- Predictions generated *after* requests arrive, not before
- User activities collected in real-time but only used to lookup pre-computed embeddings
- No real-time feature computation; session-based adaptation via embedding retrieval
- Typical pattern: Retrieve embeddings → Combine into session embedding → Candidate generation (retrieval) → Ranking
- Use cases: Session-based recommendations, in-session personalization
- Minimal streaming overhead; candidates filtered via retrieval (e.g., k-NN) before expensive ranking

**Stage 3: Online Prediction with Online Features**
- Both batch features (historical, static) and online features (fresh, dynamic) used at prediction time
- Two types of online features:
  - **Real-time (RT) features**: Computed synchronously at prediction time (low latency ~milliseconds but hard to scale; computation latency adds to user-facing latency)
  - **Near real-time features**: Precomputed asynchronously via streaming, retrieved at prediction time (staleness in order of seconds; complex features feasible; preferred approach)
- Near RT features computed using streaming processing engine; recomputed frequently only for active users
- Hundreds to thousands of features per prediction possible
- Companies: Stripe, Uber, Faire (fraud detection, credit scoring, delivery estimation, recommendations)

### Continual Learning Stages

**Stage 1: Manual, Stateless Retraining**
- Updates happen ad-hoc when model performance degrades and team has time
- Full retraining from scratch on historical + new data
- Feature/model code often mismatches between development and production
- Common in companies with <3 years ML adoption and no ML platform team

**Stage 2: Automated Retraining**
- Retraining frequency automated via scheduler (e.g., Cron, Airflow)
- Still stateless (retraining from scratch)
- Frequency often set by intuition, not data-driven optimization
- Requires model store to version code/artifacts
- Dependencies among models complicate scheduling

**Stage 3: Automated, Stateful Training**
- **Fine-tuning on new data rather than retraining from scratch**
- Reduces training cost dramatically (Grubhub: 45x reduction)
- Requires fewer data samples per update cycle
- Enables data privacy: samples only needed twice (prediction + training)
- Needs model lineage tracking and streaming feature reproducibility

**Stage 4: Continual Learning**
- Model updates triggered by:
  - Time-based schedules
  - Performance degradation detection
  - Data distribution shift detection (drift)
- Adapts to changing environments without fixed schedules
- Holy grail: Edge deployment (models ship with devices, update locally without centralized server)
- Requires drift detection, continuous evaluation (backtest, progressive evaluation, shadow deployment, canary analysis, bandits), and orchestration for model updates

### Feature Types

- **Batch features**: Extracted from historical data via batch processing; static; can be arbitrarily complex; may be days old
- **Online features**: Extracted from fresh data
  - **Real-time**: Computed synchronously at request time; fresh but hard to scale; latency-sensitive
  - **Near real-time**: Precomputed async via streaming; staleness ~seconds; scalable; complexity not latency-constrained

### Bandit Algorithms for Online Evaluation

**Bandits for Model Evaluation**
- Replace stateless A/B testing with stateful exploration
- Dynamically route traffic to higher-performing models during evaluation
- More data-efficient than A/B testing (Google study: Thompson Sampling needed 12K samples vs. 630K for 95% confidence A/B test)
- Requirements: online prediction, short feedback loops, performance tracking mechanism
- Rarely adopted outside big tech due to implementation complexity

**Contextual Bandits as Exploration Strategy**
- Balance exploitation (show likely-liked items) vs. exploration (test unpopular items) to avoid feedback loops
- Used to optimize individual prediction actions (e.g., which item to recommend)
- Hard to implement; depends on model architecture; well-researched but underdeployed

### Training Paradigms

- **Stateless retraining**: Model trained from scratch each cycle; dominant approach
- **Stateful training (fine-tuning)**: Model continues learning on new data; enables continual learning; requires model lineage tracking

## Details

### Infrastructure Requirements by Stage

**Online Prediction (Stage 2-3) requires:**

1. **Streaming Transport** (e.g., Kafka, AWS Kinesis, GCP Dataflow)
   - Moves real-time user activity events to prediction service
   - Most companies prefer managed services over self-hosted

2. **Streaming Computation Engine** (e.g., Flink SQL, KSQL, Spark Streaming)
   - Processes streaming data in real-time
   - For session-based: divides activities into sessions and maintains state
   - Flink SQL and KSQL preferred for SQL abstraction familiar to data scientists

3. **Feature Store**
   - Manages materialized features
   - Ensures consistency between training and prediction
   - Current solutions often lack feature computation management or source code tracking
   - Stage 3: must also manage streaming feature reproducibility for time-travel debugging

4. **Model Store**
   - Versions and stores all code/artifacts needed to reproduce models
   - Simple: S3 bucket with structured model blobs
   - Mature: SageMaker (managed, limited) or MLFlow (open-source, more features)
   - Stage 3+: must track model lineage (which models fine-tune on which)

5. **Inference Service**
   - Retrieves candidate items via retrieval algorithms (collaborative filtering, k-NN)
   - Scores/ranks candidates with ranking model
   - Handles embedding lookups and session embeddings

### Continual Learning Infrastructure

**Stage 2 (Automated Retraining):**
- Scheduler (Airflow, Argo, Cron)
- Data warehouse with full lineage
- Feature extraction scripts
- Label processing/annotation pipeline
- Model versioning system

**Stage 3 (Stateful Training):**
- Everything from Stage 2, plus:
- Model lineage tracking
- Streaming feature reproducibility for time-travel
- Mechanism to log predictions and extracted features for reuse ("log and wait")

**Stage 4 (Continual Learning):**
- Everything from Stage 3, plus:
- **Drift detection**: Monitor for data distribution shifts (not just feature statistics but actionable changes)
- **Continuous evaluation mechanisms**: Backtest, progressive evaluation, shadow deployment, canary analysis, bandits
- **Orchestrator**: Automatically spin up instances to update/evaluate models without interrupting prediction service

### Efficiency Considerations

**Online vs. Batch Prediction Cost:**
- Myth: Online prediction is less efficient than batch
- Reality: Online prediction avoids wasted computation on inactive users (e.g., only 2% of Grubhub's 31M users log in daily)
- Trade-off: Higher per-prediction cost but far fewer total predictions needed

**Log and Wait Pattern:**
- Reuse features already extracted during prediction for training
- Reduces compute and maintains training-serving consistency
- Emerging best practice (Faire, others)

## Interview Relevance

This concept appears in ML systems design interviews with questions such as:

1. **Architecture & Design:**
   - "Design a real-time recommendation system for an e-commerce platform. What are the key components?"
   - "Walk us through the evolution from batch to online prediction. What trade-offs would you consider?"
   - "How would

## Backlinks
- [[_index]]