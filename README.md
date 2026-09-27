# GenreDNA

A music-analysis platform that classifies the *audio itself* and exposes the evidence behind the classification.

## What makes it different

GenreDNA is designed around a **Genre DNA** rather than a single genre label:

- Multi-label genre/subgenre probabilities
- Audio-derived fingerprint: rhythm, spectrum, timbre, harmony, energy
- BPM and beat statistics
- Tonal/key/chroma analysis
- Instrument and vocal presence hooks
- Section-aware analysis (intro/verse/chorus/outro can evolve differently)
- Genre trajectory: how the song's sound changes over time
- Similarity fingerprint for finding sonically related tracks
- Explainable evidence: "why did the model call this Afrobeats?"
- Regional/custom taxonomies can be added without replacing the acoustic feature layer
- Confidence and "uncertain/hybrid" states instead of forcing a bad label

## Architecture

    audio
      |
      v
    decode/resample
      |
      +--> spectral features
      +--> rhythm/beat features
      +--> tonal/harmonic features
      +--> timbral features
      +--> loudness/dynamics
      +--> embeddings (optional pretrained model)
      |
      v
    feature fusion
      |
      +--> genre head
      +--> subgenre head
      +--> mood head
      +--> instrument head
      +--> similarity embedding
      |
      v
    Genre DNA JSON

## Quick start

    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate

    pip install -r requirements.txt
    uvicorn app.main:app --reload

Open http://127.0.0.1:8000

## Current prototype

The included analyzer is deliberately useful before training a custom classifier. It extracts real acoustic features from the uploaded track and returns a Genre DNA profile.

The classifier interface is separated from feature extraction so a trained neural model can be plugged in later without rewriting the product.

## Training direction

Start with MTG-Jamendo for multi-tag training. It contains 55k+ full tracks and 195 tags spanning genre, instrument and mood/theme; its genre subset contains 95 genre tags. See:
https://github.com/MTG/mtg-jamendo-dataset

For a stronger model, use a pretrained audio embedding model and fine-tune a multi-task classification head. Essentia publishes pretrained feature extractors and classification heads, including Discogs-derived genre models:
https://essentia.upf.edu/models/

Do not train/test on artist- or album-overlapping tracks. Split by artist to reduce leakage.

## Suggested production stack

Frontend: React/Next.js or Flutter Web
API: FastAPI
Workers: Celery/RQ + Redis
Storage: S3-compatible object storage
Metadata: PostgreSQL
Vector search: pgvector
ML: PyTorch + ONNX Runtime
Audio: FFmpeg + librosa/Essentia
Deployment: Docker

## License

Prototype code: MIT. Dataset/model licenses remain those of their respective providers.
