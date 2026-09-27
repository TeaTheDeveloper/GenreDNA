# GenreDNA roadmap

## Phase 1 — acoustic prototype
- Upload and analyze audio
- Genre DNA
- Explainable acoustic fingerprint
- Sonic trajectory
- Hybrid detection

## Phase 2 — trained classifier
1. Download MTG-Jamendo.
2. Build artist-disjoint train/validation/test splits.
3. Generate 10-second mel-spectrogram windows.
4. Train multi-label genre/instrument/mood heads.
5. Calibrate probabilities on validation data.
6. Aggregate window predictions with attention or robust pooling.

## Phase 3 — differentiation
### Genre DNA
A fixed acoustic fingerprint that can be compared across tracks.

### Genre evolution
Classify windows separately and draw a timeline:
intro -> verse -> hook -> bridge -> outro.

### Hybrid detector
Do not force a single genre. Detect mixtures and show the closest competing styles.

### Genre distance
Learn an embedding where tracks can be searched by sonic similarity, independent of artist metadata.

### Regional genre intelligence
Maintain a taxonomy layer for regional genres and subgenres. The acoustic representation remains universal while labels can evolve.

### Evidence mode
Show which measurable features contributed to a prediction. Never present model confidence as objective truth.

### Counterfactual mode
Experimental feature: perturb selected acoustic features and report how classification changes. This helps users understand what the model learned.

### Producer mode
Add arrangement-oriented tags:
- sparse/dense
- live/electronic
- acoustic/synthetic
- vocal/instrumental
- percussion prominence
- bass prominence
- dynamic range
- section energy

## Phase 4 — production
- Async processing
- object storage
- PostgreSQL metadata
- pgvector similarity search
- model versioning
- feature caching
- rate limits
- waveform/spectrogram visualization
- API keys
- batch analysis
