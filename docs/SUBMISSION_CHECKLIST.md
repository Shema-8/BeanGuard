# Submission checklist (45 marks)

**Video demo (5)**
- [ ] Camera on, clear sound · shows prediction **and** retraining · YouTube link in README

**Retraining process (10)** — script + model file present
- [ ] Data upload saved to the database (`uploads` table) · [ ] preprocessing of uploaded data · [ ] retrains from the saved custom model
- [ ] Trigger button works **and** is shown in the video · [ ] before/after metrics shown

**Prediction process (10)**
- [ ] Insert one image → correct label shown (use images you verified) · [ ] no API errors on the live URL

**Evaluation of models (10)**
- [ ] Notebook runs top-to-bottom with outputs saved · [ ] preprocessing explained
- [ ] Optimisation: pretrained backbone, dropout, L2, early stopping, LR schedule, hyper-parameter search
- [ ] At least 4 metrics (accuracy, loss, precision, recall, F1) + confusion matrix + baseline comparison
- [ ] Interpretations of at least 3 features written (notebook §2)

**Deployment package (10)**
- [ ] Dockerised UI + public URL · [ ] data insights / visualisations tab · [ ] Swagger docs reachable

**Repo requirements**
- [ ] README: video link, URL, description, setup steps, **Locust results** (1/2/3 containers) filled in
- [ ] Structure matches the brief: `notebook/`, `src/{preprocessing,model,prediction}.py`, `data/{train,test}`, `models/`
- [ ] Model file committed (`.keras` + `.h5`) · [ ] `data/SOURCES.md` completed
- [ ] Attempt 1: **ZIP** of the repo · Attempt 2: **GitHub URL** (public, or lecturer added as collaborator)
- [ ] `git log` shows regular commits
