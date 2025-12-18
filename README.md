# Face Recognition (10 People) — Embedding-based

Pipeline (target):
1) Face detection → crop/alignment
2) Face embedding (pretrained)
3) Gallery match (cosine/L2) + threshold for Unknown
4) Evaluation: ROC, FAR/FRR, confusion matrix
5) Streamlit demo

## Repo structure
- `src/face_recognition/` : reusable code
- `notebooks/` : experiments / learning
- `scripts/` : runnable commands
- `data/` and `models/` are not committed (stored on Drive)
