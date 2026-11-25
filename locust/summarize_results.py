"""Builds a markdown table (and chart) from locust/results/*_containers_stats.csv."""
import glob
import re
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).parent / "results"


def main():
    rows = []
    for path in sorted(glob.glob(str(RESULTS / "*_containers_stats.csv"))):
        n = int(re.search(r"(\d+)_containers", path).group(1))
        df = pd.read_csv(path)
        r = df[df["Name"] == "/predict"].iloc[0]
        rows.append({
            "Containers": n,
            "Requests": int(r["Request Count"]),
            "Failures": int(r["Failure Count"]),
            "Avg (ms)": round(r["Average Response Time"], 1),
            "Median (ms)": round(r["Median Response Time"], 1),
            "p95 (ms)": round(r["95%"], 1),
            "p99 (ms)": round(r["99%"], 1),
            "Max (ms)": round(r["Max Response Time"], 1),
            "Req/s": round(r["Requests/s"], 2),
        })
    if not rows:
        raise SystemExit("No result files found in locust/results/")
    table = pd.DataFrame(rows).sort_values("Containers")
    md = table.to_markdown(index=False) if hasattr(table, "to_markdown") else table.to_string(index=False)
    (RESULTS / "summary.md").write_text(md + "\n")
    print(md)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar(table["Containers"].astype(str), table["Median (ms)"], label="median")
        ax[0].plot(table["Containers"].astype(str), table["p95 (ms)"], "ro-", label="p95")
        ax[0].set_title("Latency vs containers")
        ax[0].set_ylabel("ms")
        ax[0].legend()
        ax[1].bar(table["Containers"].astype(str), table["Req/s"], color="green")
        ax[1].set_title("Throughput vs containers")
        ax[1].set_ylabel("requests / second")
        for a in ax:
            a.set_xlabel("API containers")
        fig.tight_layout()
        fig.savefig(RESULTS / "latency_throughput.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
