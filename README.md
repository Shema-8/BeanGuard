# 🌿 BeanGuard — bean leaf disease detection, an end-to-end MLOps project

> Diagnose bean leaf diseases (healthy · angular leaf spot · bean rust) from a phone photo — and **keep
> improving the model** as agronomists contribute new labelled images.

| | |
|---|---|
| 🎥 **Video demo** | `<<YOUTUBE LINK — camera on>>` |
| 🌐 **Live dashboard** | `<<http://YOUR-SERVER-IP:8501>>` |
| 🔌 **Live API docs (Swagger)** | `<<http://YOUR-SERVER-IP/docs>>` |
| 📓 **Notebook** | [`notebook/rwanda_heritage_classifier.ipynb`](notebook/rwanda_heritage_classifier.ipynb) |
| 📦 **Model files** | [`models/`](models/) (`.keras` served by the API, `.h5` export) |

---

## 1. The problem

Beans are a staple crop in Rwanda and across East Africa. Leaf diseases such as **angular leaf spot** and
**bean rust** can destroy a harvest if they are not caught early, but agronomists are few and fields are many.
BeanGuard lets a farmer or extension officer photograph a leaf and get an instant diagnosis with practical advice.
Because new labelled photos can be uploaded at any time, the model **keeps learning from the field** instead of
going stale.

**Classes:** `healthy`, `angular_leaf_spot`, `bean_rust`.
**Dataset:** iBean — ~1,300 field photos of bean leaves from Makerere AI Lab (Uganda), see `data/SOURCES.md`.
Low-confidence predictions are flagged in the UI ("this may not be a bean leaf").

## 2. The ML lifecycle in this repo

```
 Data acquisition ─► Preprocessing ─► Model creation ─► Evaluation ─► FastAPI ─► Dashboard
 scripts/split_      src/preprocessing  notebook/ +       src/model.py   api/        ui/ (Streamlit)
 dataset.py          .py                src/train.py                     main.py        │
        ▲                                                                   ▲           │ upload + button
        │                                                                   │ hot-reload▼
        └──────────── data/train ◄── merge ◄── src/retrain.py ◄── /upload, /retrain  (+ auto trigger)
                                                      │
                          Docker Compose: nginx ─► N × api containers ─► Locust load test & monitoring
```

| Stage | Where | Notes |
|---|---|---|
| 1. Data acquisition | `scripts/download_beans.py`, `scripts/split_dataset.py` | download iBean, validate, dedupe, split into train/test/demo |
| 2. Data processing | `src/preprocessing.py` | EXIF fix, resize 224², in-model rescale, class weights, `tf.data` |
| 3. Model creation | `notebook/…ipynb`, `src/model.py`, `src/train.py` | MobileNetV2 transfer learning, 2-phase training |
| 4. Model testing | notebook §6–7, `src/model.evaluate` | accuracy, loss, precision, recall, F1, confusion matrix |
| 5. **Retraining** | `src/retrain.py` | upload → DB → preprocess → fine-tune **existing** model → compare → promote |
| 6. API | `api/main.py` | FastAPI: `/predict /upload /retrain /status /metrics /insights` |
| UI | `ui/app.py` | Diagnose · Data insights · Upload & Retrain · Monitoring |
| Deployment | `Dockerfile.*`, `docker-compose.yml`, `nginx/` | scale API replicas behind nginx |
| Load test | `locust/` | 1 / 2 / 3 container experiment |

## 3. Model & optimisation techniques

MobileNetV2 (ImageNet) → GlobalAveragePooling → Dropout(0.3) → L2-regularised softmax.

* Transfer learning, data augmentation (flip/rotation/zoom/contrast), Dropout, L2 weight decay
* Adam optimiser; **two-phase schedule**: frozen backbone (lr 1e-3) → fine-tune top 40 layers (lr 1e-5)
* EarlyStopping, ReduceLROnPlateau, balanced class weights, small hyper-parameter search
* A vanilla CNN baseline is trained to show what the optimisations buy

### Results (held-out test set) — `<<fill in from notebook §6>>`

| Model | Accuracy | Loss | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| Baseline CNN | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| MobileNetV2 (optimised) | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` |

## 4. The retraining pipeline

1. **Upload** labelled images (one class per upload) or a ZIP with one folder per class → validated, de-duplicated,
   stored in `data/uploads/` and recorded in the SQLite database (`uploads` table).
2. **Trigger** — either the **Retrain** button / `POST /retrain`, or automatically once `RETRAIN_THRESHOLD`
   (default 30) new images are waiting.
3. **Preprocess** — uploads are merged into `data/train/<class>/` and run through the same `tf.data` pipeline.
4. **Retrain** — the *existing, already-trained* model is loaded and fine-tuned (low learning rate) on the
   enlarged dataset: a custom pre-trained model, not ImageNet from scratch.
5. **Evaluate & gate** — old and new models are scored on the **untouched test set**; the new model is promoted
   only if macro-F1 is not worse. The old model is archived (`models/archive/`).
6. **Deploy** — the model file is replaced atomically; every API replica detects the new file and hot-reloads it
   (no restart). The version history is shown in the Monitoring tab.

A lock file prevents concurrent retraining jobs across replicas; a failed run leaves the live model untouched.

## 5. Setup

### Option A — Docker (recommended, same as production)
```bash
git clone <<YOUR_REPO_URL>> && cd <<YOUR_REPO_FOLDER>>
# models/ must contain the trained model (heritage_mobilenetv2.keras, class_names.json, metrics.json)
docker compose up -d --build --scale api=2
```
* Dashboard: http://localhost:8501  · API docs: http://localhost/docs

### Option B — local Python
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r ui/requirements.txt
uvicorn api.main:app --reload                           # terminal 1
API_URL=http://localhost:8000 streamlit run ui/app.py   # terminal 2
```

### Train from scratch
```bash
# 1. download the iBean dataset into raw_images/<class>/ and split it
python scripts/download_beans.py
python scripts/split_dataset.py raw_images --out data
# 2a. notebook:  jupyter lab notebook/rwanda_heritage_classifier.ipynb    (Colab GPU recommended)
# 2b. or CLI:    python -m src.train --head-epochs 15 --ft-epochs 15
```
Run tests: `pytest -q` · Demo ZIP for the retraining upload: `python scripts/make_demo_zip.py`

### Configuration (environment variables)
`RETRAIN_THRESHOLD` (30) · `RETRAIN_EPOCHS` (8) · `RETRAIN_LR` (1e-5) · `CONFIDENCE_THRESHOLD` (0.5) ·
`PROMOTION_TOLERANCE` (0.0) · `MAX_UPLOAD_MB` (50) — see `src/config.py`.

## 6. API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | classify one image → label, confidence, top-3, model version, latency |
| `POST /upload` | bulk labelled images (`label` form field) or ZIP of class folders |
| `POST /retrain` · `GET /retrain/status` | trigger / follow retraining |
| `GET /status` · `GET /health` | uptime, model version, pending uploads, serving container |
| `GET /metrics` | test-set metrics, live latency stats, model version history |
| `GET /insights` | dataset statistics + auto-written interpretations for the dashboard |

## 7. Cloud deployment

Deployed on `<<AWS EC2 t3.xlarge / GCP e2-standard-4 — region>>` with Docker Compose.
Full step-by-step: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Public URL: `<<…>>`.

## 8. Flood-request simulation (Locust)

Same test (`<<50>>` users, spawn `<<5>>`/s, `<<120>>` s) against 1, 2 and 3 API containers:

```bash
pip install -r locust/requirements.txt
./locust/run_experiments.sh 50 5 120s     # writes locust/results/*.csv and summary.md
```

| Containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Req/s |
|---|---|---|---|---|---|---|---|
| 1 | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| 2 | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` |
| 3 | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` | `<<>>` |

![latency and throughput](locust/results/latency_throughput.png)

**Reading the results:** `<<2–3 sentences: how latency and throughput change as containers increase; where it
stops scaling (CPU cores of the VM); failures>>`

## 9. Repository layout

```
├── README.md
├── notebook/rwanda_heritage_classifier.ipynb
├── src/            preprocessing.py  model.py  prediction.py  retrain.py  train.py  database.py  config.py
├── api/main.py     FastAPI service
├── ui/app.py       Streamlit dashboard
├── data/           train/  test/  uploads/ (runtime)  SOURCES.md
├── models/         heritage_mobilenetv2.keras / .h5 · class_names.json · metrics.json
├── locust/         locustfile.py  run_experiments.sh  summarize_results.py  results/
├── scripts/        download_beans.py  split_dataset.py  make_demo_zip.py
├── nginx/ Dockerfile.api Dockerfile.ui docker-compose.yml
├── tests/          unit + API tests (run in CI)
└── docs/           DEPLOYMENT.md  DEMO_SCRIPT.md  SUBMISSION_CHECKLIST.md
```

## 10. Limitations & ethics
* The dataset has only three classes and field photos from one region; the model may not generalise to other bean
  diseases, other crops or very different cameras. Low-confidence predictions are flagged, not eliminated.
* BeanGuard supports, and does not replace, an agronomist: the advice cards are general guidance.
* The iBean dataset is credited in `data/SOURCES.md`; contribute only photos you have the right to share.
* SQLite on a shared volume is fine for one VM; production would use Postgres + object storage.
* One-VM scaling is bounded by CPU cores (see the load-test discussion).
