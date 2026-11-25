# Cloud deployment (AWS EC2 — GCP/Azure are identical apart from the console steps)

**Why a VM + Docker Compose?** The assignment needs *different numbers of Docker containers* behind one URL.
Compose + nginx gives exactly that with one command, on any cloud.

## 1. Create the VM
* Ubuntu 22.04, **4 vCPU / 8 GB** (e.g. EC2 `t3.xlarge`, GCP `e2-standard-4`) — fewer cores hide the scaling effect.
* 30 GB disk. Security group / firewall: allow inbound **22** (SSH, your IP), **80** (API), **8501** (dashboard).

## 2. Install Docker
```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git python3-pip
sudo usermod -aG docker $USER && newgrp docker
```

## 3. Get the code and the model
```bash
git clone <YOUR_REPO_URL> && cd <YOUR_REPO_FOLDER>
ls models/            # heritage_mobilenetv2.keras  class_names.json  metrics.json must exist
ls data/train data/test
```
(If the dataset is too large for git, `scp` it into `data/` — the containers mount `./data` and `./models`.)

## 4. Run
```bash
docker compose up -d --build --scale api=2
docker compose ps
curl http://localhost/health
```
Open `http://<PUBLIC_IP>:8501` (dashboard) and `http://<PUBLIC_IP>/docs` (Swagger).

## 5. Load-test with different container counts
Run Locust **from the VM itself** (or a second machine) so your home network does not distort latency:
```bash
pip3 install -r locust/requirements.txt
./locust/run_experiments.sh 50 5 120s
cat locust/results/summary.md
```
Screenshot: Locust HTML reports (`locust/results/*.html`), `docker compose ps` with 1, 2 and 3 api replicas, and the
dashboard Monitoring tab ("Requests per container" proves load balancing).

## 6. Evidence of "evaluation in production" (rubric)
* Dashboard → Monitoring: uptime, predictions served, avg / p95 latency, confidence histogram, per-container requests.
* Dashboard → Visualisations: test-set metrics + confusion matrix of the **deployed** model version.
* Dashboard → Upload & Retrain → version table shows v1 → v2 with macro-F1 before/after.

## 7. Operations
```bash
docker compose logs -f api         # logs
docker compose down                # stop (data/ and models/ persist on disk)
docker compose up -d --build       # redeploy after pushing code
```
