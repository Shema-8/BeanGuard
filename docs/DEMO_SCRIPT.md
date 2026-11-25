# Video demo script (5–7 min · camera ON · clear audio)

Rubric: *"prediction and retraining process in a video with camera on"*. Keep your face visible (picture-in-picture)
and speak while you click.

| Time | Do | Say |
|---|---|---|
| 0:00 | Camera on, introduce yourself, show README | The problem: bean leaf diseases hurt Rwandan farmers; one system from notebook to monitoring |
| 0:30 | Notebook: EDA plots, interpretations | What the data says (3 features), class balance |
| 1:15 | Notebook: baseline vs optimised table, confusion matrix | 4+ metrics, optimisations used |
| 2:15 | Open the **live URL** (not localhost) → Diagnose tab | Upload a test leaf → correct diagnosis + confidence + advice card. Try a random photo → "low confidence" |
| 3:00 | Visualisations tab | Interpretations, model metrics of the deployed version |
| 3:30 | Upload & Retrain: upload `new_data_demo.zip` | Files are validated and stored in the database; pending counter rises |
| 4:15 | Click **Retrain model**, show live status | Preprocess → fine-tune the existing model → compare on the untouched test set → promote/reject |
| 5:00 | Predict again; Monitoring → version table | Model v1 → v2, F1 before/after, hot reload without restart |
| 5:30 | Monitoring + terminal `docker compose ps` | Uptime, latency, requests per container; Locust results table from README |
| 6:15 | Wrap up | Limitations and next steps |

**Before recording:** pick 3 test images the model gets right; make sure `data/new_data_demo` contains images of the
classes the model is weakest on; do one dry run; check the microphone.
