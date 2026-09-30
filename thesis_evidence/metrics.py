"""Ricalcola le misure dei capitoli 5 e 6 dalle osservazioni allegate.

Uso:
    python metrics.py
    python metrics.py --no-show
    python metrics.py --prepare-inputs PERCORSO_ARCHIVIO_OPERATIVO

L'esecuzione ordinaria stampa le tabelle, salva i PNG e apre i grafici.
Il comando --prepare-inputs serve solo a ricostruire il piccolo insieme di input
a partire dalle evidenze originali; non è necessario per leggere il fascicolo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
DATASET = ROOT / "dataset"
CONFIGURATIONS = ("E0", "E1", "E2", "E3", "E4")
RETRIEVAL = CONFIGURATIONS[1:]
K_VALUES = (1, 3, 5)
PHASES = ("retrieval", "reranking", "prompt_build", "generation", "total")
OUTPUT_CAP = 3072

CASE_COLUMNS = (
    "configuration", "case_id", "query_type", "K", "relevant_documents",
    "recall", "precision", "hit", "rr", "ndcg_binary",
)
RETRIEVAL_COLUMNS = (
    "configuration", "K", "n_cases", "mean_recall", "mrr_at_k",
    "mean_precision", "hit_rate", "mean_ndcg_binary", "hit_cases",
)
TYPE_COLUMNS = (
    "configuration", "K", "query_type", "n_cases", "mean_recall",
    "mrr_at_k", "hit_rate",
)
REQUEST_COLUMNS = (
    "configuration", "case_id", "retrieval_ms", "reranking_ms",
    "prompt_build_ms", "generation_ms", "total_ms", "input_tokens",
    "output_tokens", "total_tokens", "at_output_cap", "source_count",
)
LATENCY_COLUMNS = (
    "configuration", "phase", "n_requests", "mean_ms", "median_ms",
    "sample_std_ms", "p95_ms", "min_ms", "max_ms",
)
TOKEN_COLUMNS = (
    "configuration", "n_requests", "token_available", "mean_input_tokens",
    "mean_output_tokens", "total_input_tokens", "total_output_tokens",
    "at_output_cap",
)
GPU_COLUMNS = (
    "configuration", "samples", "mean_gpu_util_percent",
    "p95_gpu_util_percent", "max_gpu_memory_mib", "mean_gpu_memory_mib",
    "mean_gpu_power_w",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def prepare_inputs(source: Path) -> None:
    """Estrae soltanto i campi necessari dalle evidenze operative originali."""
    source = source.resolve(strict=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    DATASET.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(source / "01_dataset/development_draft.jsonl", DATASET / "development.jsonl")
    shutil.copyfile(source / "01_dataset/test_draft.jsonl", DATASET / "test.jsonl")
    catalogs = (
        read_csv(source / "01_dataset/dataset_catalog_draft.csv")
        + read_csv(source / "01_dataset/test_catalog_draft.csv")
    )
    catalog_columns = (
        "case_id", "split", "failure_type", "query_type", "complexity", "source_capture_id",
    )
    write_csv(DATASET / "catalog.csv", catalogs, catalog_columns)

    source_rankings = read_json(source / "05_pilot/development_retrieval_draft.json")
    rankings = [
        {
            "configuration": row["configuration"],
            "case_id": row["case_id"],
            "sources": row["sources"],
        }
        for row in source_rankings["records"]
    ]
    write_jsonl(INPUT / "rankings.jsonl", rankings)

    run_dir = source / "05_pilot/generation_runs/20260929T144729Z"
    source_manifest = read_json(run_dir / "manifest.json")
    requests: list[dict[str, Any]] = []
    examples: dict[str, dict[str, Any]] = {}
    for configuration in CONFIGURATIONS:
        run_cases = read_jsonl(run_dir / f"{configuration}_cases.jsonl")
        for row in run_cases:
            requests.append({
                "configuration": configuration,
                "case_id": row["case_id"],
                "operational_metrics": row["operational_metrics"],
                "source_count": len(row["retrieved_sources"]),
                "answer_length": len(row["generated_answer"]),
            })
        if configuration in ("E0", "E4"):
            first = run_cases[0]
            examples[configuration] = {
                "case_id": first["case_id"],
                "answer_prefix": first["generated_answer"][:400],
                "answer_longer_than_excerpt": len(first["generated_answer"]) > 280,
                "source_count": len(first["retrieved_sources"]),
            }
    write_jsonl(INPUT / "requests.jsonl", requests)
    (INPUT / "response_example.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(run_dir / "gpu_samples.csv", INPUT / "gpu_samples.csv")

    fixture_dir = source / "02_telemetry/fixtures_draft"
    telemetry_dir = source / "02_telemetry"
    fault_names = (
        "payment-unreachable", "payment-failure", "cart-failure",
        "product-catalog-failure", "product-catalog-lock-contention",
    )
    faults = []
    for name in fault_names:
        capture = f"CAP-{name}-20260929-02"
        fixture = read_json(fixture_dir / f"{capture}.json")
        item: dict[str, Any] = {
            "fault": name,
            "capture_id": capture,
            "span_count": len(fixture["telemetry"]["spans"]),
        }
        injection = telemetry_dir / "injection_logs" / f"{capture}.json"
        if injection.exists():
            observed = read_json(injection)
            item["http"] = {
                key: value for key, value in observed.items()
                if key.startswith(("cart_http_", "checkout_http_", "target_http_", "control_http_"))
                or key == "checkout_error_code"
            }
        faults.append(item)
    write_jsonl(INPUT / "fault_observations.jsonl", faults)

    trace = read_json(fixture_dir / "CAP-payment-unreachable-20260929-02.json")
    selected = [
        span for span in trace["telemetry"]["spans"]
        if span["service"] in {"frontend", "checkout"}
        and ("PlaceOrder" in span["operation"] or "Charge" in span["operation"])
    ]
    (INPUT / "trace_example.json").write_text(
        json.dumps({"spans": selected}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lock = read_json(
        fixture_dir / "CAP-product-catalog-lock-contention-20260929-02_latency.json"
    )
    (INPUT / "lock_contention.json").write_text(
        json.dumps({"windows": lock["windows"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    input_names = (
        "rankings.jsonl", "requests.jsonl", "gpu_samples.csv",
        "fault_observations.jsonl", "trace_example.json", "response_example.json",
        "lock_contention.json",
    )
    manifest = {
        "run_id": source_manifest["run_id"],
        "model": source_manifest["model"],
        "model_id_abbreviated": source_manifest["model_id_abbreviated"],
        "dataset_sha256": source_manifest["dataset_sha256"],
        "index_sha256": source_manifest["index_sha256"],
        "chunks_sha256": source_manifest["chunks_sha256"],
        "modelfile_sha256": source_manifest["modelfile_sha256"],
        "dataset_case_count": source_manifest["dataset_case_count"],
        "top_k": source_manifest["top_k"],
        "candidate_top_n": source_manifest["candidate_top_n"],
        "demo_stack_active_at_start": source_manifest["demo_stack_active_at_start"],
        "completed": [
            {
                "configuration": row["configuration"],
                "case_count": row["case_count"],
                "wall_seconds": row["wall_seconds"],
            }
            for row in source_manifest["completed"]
        ],
        "input_sha256": {name: sha256(INPUT / name) for name in input_names},
        "dataset_sha256_local": {
            name: sha256(DATASET / name)
            for name in ("development.jsonl", "test.jsonl", "catalog.csv")
        },
    }
    (INPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Preparati {len(catalogs)} casi, {len(rankings)} graduatorie e {len(requests)} richieste in {ROOT}")


def validate_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(INPUT / "manifest.json")
    for name, expected in manifest["input_sha256"].items():
        actual = sha256(INPUT / name)
        if actual != expected:
            raise ValueError(f"Input modificato: {name}; atteso {expected}, rilevato {actual}")
    for name, expected in manifest["dataset_sha256_local"].items():
        actual = sha256(DATASET / name)
        if actual != expected:
            raise ValueError(f"Dataset modificato: {name}; atteso {expected}, rilevato {actual}")
    if sha256(DATASET / "development.jsonl") != manifest["dataset_sha256"]:
        raise ValueError("Il dataset di sviluppo differisce da quello registrato nel run")
    development = read_jsonl(DATASET / "development.jsonl")
    test = read_jsonl(DATASET / "test.jsonl")
    catalog = read_csv(DATASET / "catalog.csv")
    if len(development) != 10 or len(test) != 50:
        raise ValueError("Il dataset deve contenere 10 casi development e 50 test")
    by_id = {row["id"]: row for row in development + test}
    if len(by_id) != 60 or {row["case_id"] for row in catalog} != set(by_id):
        raise ValueError("Catalogo e dataset contengono identificativi diversi o duplicati")
    if {row["case_id"] for row in catalog if row["split"] == "development"} != {row["id"] for row in development}:
        raise ValueError("I casi di sviluppo non coincidono con il catalogo")
    if {row["case_id"] for row in catalog if row["split"] == "test"} != {row["id"] for row in test}:
        raise ValueError("I casi di test non coincidono con il catalogo")
    cases = [
        {**row, "relevant_documents": by_id[row["case_id"]]["relevant_documents"]}
        for row in catalog
    ]
    rankings = read_jsonl(INPUT / "rankings.jsonl")
    requests = read_jsonl(INPUT / "requests.jsonl")
    if len(cases) != 60 or Counter(row["split"] for row in cases) != {"development": 10, "test": 50}:
        raise ValueError("Il catalogo deve contenere 10 casi development e 50 test")
    if len({row["case_id"] for row in cases}) != 60:
        raise ValueError("Identificativi di caso duplicati")
    if len(rankings) != 40 or Counter(row["configuration"] for row in rankings) != {c: 10 for c in RETRIEVAL}:
        raise ValueError("Sono necessarie 40 graduatorie, dieci per E1-E4")
    if len(requests) != 50 or Counter(row["configuration"] for row in requests) != {c: 10 for c in CONFIGURATIONS}:
        raise ValueError("Sono necessarie 50 richieste, dieci per E0-E4")
    development_ids = {row["case_id"] for row in cases if row["split"] == "development"}
    for configuration in RETRIEVAL:
        if {row["case_id"] for row in rankings if row["configuration"] == configuration} != development_ids:
            raise ValueError(f"Casi di retrieval incompleti per {configuration}")
    for configuration in CONFIGURATIONS:
        if {row["case_id"] for row in requests if row["configuration"] == configuration} != development_ids:
            raise ValueError(f"Casi di generazione incompleti per {configuration}")
    if any(len(row["sources"]) != 5 for row in rankings):
        raise ValueError("Ogni graduatoria deve contenere cinque risultati")
    if any(row["answer_length"] <= 0 for row in requests):
        raise ValueError("Una risposta del run è vuota")
    return manifest, cases, rankings, requests


def retrieval_metrics(sources: list[str], relevant: set[str], k: int) -> dict[str, float]:
    if not relevant:
        raise ValueError("Un caso development non ha documenti rilevanti")
    ranked = sources[:k]
    seen: set[str] = set()
    hits: list[bool] = []
    for document in ranked:
        hits.append(document in relevant and document not in seen)
        seen.add(document)
    unique_relevant = relevant.intersection(ranked)
    first_rank = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return {
        "recall": len(unique_relevant) / len(relevant),
        "precision": sum(hits) / k,
        "hit": float(bool(unique_relevant)),
        "rr": 1 / first_rank if first_rank else 0.0,
        "ndcg_binary": dcg / ideal if ideal else 0.0,
    }


def calculate_retrieval(
    cases: list[dict[str, Any]], rankings: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["case_id"]: row for row in cases}
    case_rows = []
    for ranking in rankings:
        case = by_id[ranking["case_id"]]
        relevant = set(case["relevant_documents"])
        for k in K_VALUES:
            case_rows.append({
                "configuration": ranking["configuration"],
                "case_id": ranking["case_id"],
                "query_type": case["query_type"],
                "K": k,
                "relevant_documents": len(relevant),
                **retrieval_metrics(ranking["sources"], relevant, k),
            })

    summary_rows = []
    type_rows = []
    for configuration in RETRIEVAL:
        for k in K_VALUES:
            group = [
                row for row in case_rows
                if row["configuration"] == configuration and row["K"] == k
            ]
            summary_rows.append({
                "configuration": configuration,
                "K": k,
                "n_cases": len(group),
                "mean_recall": statistics.mean(row["recall"] for row in group),
                "mrr_at_k": statistics.mean(row["rr"] for row in group),
                "mean_precision": statistics.mean(row["precision"] for row in group),
                "hit_rate": statistics.mean(row["hit"] for row in group),
                "mean_ndcg_binary": statistics.mean(row["ndcg_binary"] for row in group),
                "hit_cases": sum(row["hit"] > 0 for row in group),
            })
            by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group:
                by_type[row["query_type"]].append(row)
            for query_type, items in sorted(by_type.items()):
                type_rows.append({
                    "configuration": configuration,
                    "K": k,
                    "query_type": query_type,
                    "n_cases": len(items),
                    "mean_recall": statistics.mean(item["recall"] for item in items),
                    "mrr_at_k": statistics.mean(item["rr"] for item in items),
                    "hit_rate": statistics.mean(item["hit"] for item in items),
                })
    return case_rows, summary_rows, type_rows


def calculate_generation(
    requests: list[dict[str, Any]], gpu_samples: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    request_rows = []
    latency_rows = []
    token_rows = []
    gpu_rows = []
    for configuration in CONFIGURATIONS:
        group = [row for row in requests if row["configuration"] == configuration]
        for row in group:
            operational = row["operational_metrics"]
            tokens = operational["token_usage"]
            request_rows.append({
                "configuration": configuration,
                "case_id": row["case_id"],
                "retrieval_ms": operational["retrieval_latency_ms"],
                "reranking_ms": operational["reranking_latency_ms"],
                "prompt_build_ms": operational["prompt_build_latency_ms"],
                "generation_ms": operational["generation_latency_ms"],
                "total_ms": operational["total_latency_ms"],
                "input_tokens": tokens["input_tokens"] if tokens else "N/A",
                "output_tokens": tokens["output_tokens"] if tokens else "N/A",
                "total_tokens": tokens["total_tokens"] if tokens else "N/A",
                "at_output_cap": int(tokens["output_tokens"] >= OUTPUT_CAP) if tokens else "N/A",
                "source_count": row["source_count"],
            })
        measured = [row for row in request_rows if row["configuration"] == configuration]
        for phase in PHASES:
            values = [float(row[f"{phase}_ms"]) for row in measured]
            latency_rows.append({
                "configuration": configuration,
                "phase": phase,
                "n_requests": len(values),
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "sample_std_ms": statistics.stdev(values),
                "p95_ms": percentile(values, 0.95),
                "min_ms": min(values),
                "max_ms": max(values),
            })
        available = [row for row in measured if row["total_tokens"] != "N/A"]
        token_rows.append({
            "configuration": configuration,
            "n_requests": len(measured),
            "token_available": len(available),
            "mean_input_tokens": (
                statistics.mean(int(row["input_tokens"]) for row in available)
                if available else "N/A"
            ),
            "mean_output_tokens": (
                statistics.mean(int(row["output_tokens"]) for row in available)
                if available else "N/A"
            ),
            "total_input_tokens": sum(int(row["input_tokens"]) for row in available),
            "total_output_tokens": sum(int(row["output_tokens"]) for row in available),
            "at_output_cap": sum(int(row["at_output_cap"]) for row in available),
        })

        samples = [
            row for row in gpu_samples
            if row["configuration"] == configuration and row["phase"] == "measured"
        ]
        if not samples:
            raise ValueError(f"Campioni GPU assenti per {configuration}")
        utilization = [float(row["gpu_util_percent"]) for row in samples]
        memory = [float(row["gpu_memory_mib"]) for row in samples]
        power = [float(row["gpu_power_w"]) for row in samples]
        gpu_rows.append({
            "configuration": configuration,
            "samples": len(samples),
            "mean_gpu_util_percent": statistics.mean(utilization),
            "p95_gpu_util_percent": percentile(utilization, 0.95),
            "max_gpu_memory_mib": max(memory),
            "mean_gpu_memory_mib": statistics.mean(memory),
            "mean_gpu_power_w": statistics.mean(power),
        })
    return request_rows, latency_rows, token_rows, gpu_rows


def print_table(title: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    print(f"\n{title} ({len(rows)} righe)")
    strings = [[str(row[column]) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in strings))
        for index, column in enumerate(columns)
    ]
    print(" | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for row in strings:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_extra_measurements(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    print("\nDistribuzione del dataset")
    for field in ("split", "failure_type", "query_type", "complexity"):
        counts = Counter(row[field] for row in cases)
        print(f"  {field}: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))

    print("\nEsiti dei guasti controllati")
    for row in read_jsonl(INPUT / "fault_observations.jsonl"):
        status = ", ".join(f"{key}={value}" for key, value in row.get("http", {}).items())
        print(f"  {row['fault']}: {row['span_count']} span; {status}")
    lock = read_json(INPUT / "lock_contention.json")
    for window in lock["windows"]:
        statuses = Counter(str(item["http_status"]) for item in window["observations"])
        latencies = [float(item["latency_ms"]) for item in window["observations"]]
        if len(latencies) != window["n"]:
            raise ValueError(f"Numero di osservazioni incoerente: {window['phase']}")
        print(
            f"  lock {window['phase']}: n={window['n']}, "
            f"concorrenza={window['concurrency']}, "
            f"mediana={statistics.median(latencies):.1f} ms, "
            f"p95={percentile(latencies, 0.95):.1f} ms, "
            f"HTTP={dict(statuses)}"
        )

    print("\nThroughput sequenziale osservato")
    for row in manifest["completed"]:
        cases_per_minute = row["case_count"] / row["wall_seconds"] * 60
        print(f"  {row['configuration']}: {cases_per_minute:.2f} casi/min")


def generate_tables(
    output: Path,
    case_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
) -> None:
    tables = output / "tabelle"
    specifications = (
        ("retrieval_case_level.csv", case_rows, CASE_COLUMNS),
        ("retrieval_summary.csv", retrieval_rows, RETRIEVAL_COLUMNS),
        ("retrieval_by_query_type.csv", type_rows, TYPE_COLUMNS),
        ("request_level.csv", request_rows, REQUEST_COLUMNS),
        ("latency_summary.csv", latency_rows, LATENCY_COLUMNS),
        ("token_summary.csv", token_rows, TOKEN_COLUMNS),
        ("gpu_summary.csv", gpu_rows, GPU_COLUMNS),
    )
    for name, rows, columns in specifications:
        write_csv(tables / name, rows, columns)
        print_table(name, rows, columns)


COLORS = {
    "E0": "#686868", "E1": "#286DA8", "E2": "#DA8B1F",
    "E3": "#3B8D68", "E4": "#A64A69",
}


def generate_figures(
    output: Path,
    cases: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    gpu_rows: list[dict[str, Any]],
    show: bool,
) -> None:
    """Disegna le dieci figure dai casi, dalle misure e dalle tabelle ricalcolate."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Per generare le figure installare matplotlib: pip install matplotlib") from error

    destination = output / "figure"
    destination.mkdir(parents=True, exist_ok=True)
    created = []

    def common(ax: Any) -> None:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)

    def save(fig: Any, filename: str, dpi: int = 300) -> None:
        fig.savefig(destination / filename, dpi=dpi, bbox_inches="tight", facecolor="white")
        created.append(filename)
        if not show:
            plt.close(fig)

    def value(rows: list[dict[str, Any]], configuration: str, field: str, **filters: Any) -> float:
        matches = [
            row for row in rows
            if row["configuration"] == configuration
            and all(row[key] == expected for key, expected in filters.items())
        ]
        if len(matches) != 1:
            raise ValueError(f"Attesa una misura {field} per {configuration}: {filters}")
        return float(matches[0][field])

    # Figura 5.1: numerosità dei casi per classe.
    fields = (
        ("failure_type", "Scenario di guasto", None),
        ("query_type", "Tipologia di domanda", ("exact", "paraphrase", "distributed")),
        ("complexity", "Complessità", ("simple", "intermediate", "complex")),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, (field, title, order) in zip(axes, fields):
        counts = Counter(row[field] for row in cases)
        labels = list(order or sorted(counts))
        positions = list(range(len(labels)))
        ax.barh(positions, [counts[label] for label in labels], color="#286DA8", height=0.65)
        ax.set_yticks(positions, labels=[label.replace("_", " ") for label in labels])
        ax.invert_yaxis()
        ax.set_xlabel("Numero di casi")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        for position, label in zip(positions, labels):
            ax.text(counts[label] + 0.4, position, str(counts[label]), va="center", fontsize=8)
        ax.set_xlim(0, max(counts.values()) * 1.3)
    fig.suptitle("Distribuzione delle 60 domande per guasto e difficoltà", fontsize=12)
    save(fig, "figure_5_1_dataset.png")

    # Figura 5.2: tre span selezionati dalla cattura Payment.
    spans = sorted(
        read_json(INPUT / "trace_example.json")["spans"],
        key=lambda row: (row["offset_ms"], row["service"]),
    )
    if not spans:
        raise ValueError("La traccia di esempio è vuota")
    labels = [f"{row['service']} · {row['operation'].split('/')[-1]}" for row in spans]
    fig, ax = plt.subplots(figsize=(11, 3.1))
    for index, span in enumerate(spans):
        duration = max(span["duration_ms"], 3)
        ax.barh(
            index, duration, left=span["offset_ms"],
            color="#aa4b68" if "Charge" in span["operation"] else "#2a6f9e",
            height=0.55,
        )
        ax.text(
            span["offset_ms"] + duration + 55, index,
            f"{span['duration_ms']:.2f} ms · {span['tags'].get('otel.status_code', 'N/A')}",
            va="center", fontsize=9,
        )
    ax.set_yticks(range(len(spans)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 5150)
    ax.set_xlabel("Tempo relativo della trace (ms)")
    ax.set_title("Traccia del guasto Payment irraggiungibile")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "figure_5_2_trace_example.png", dpi=180)

    # Figura 5.3: confronto fra i primi estratti della stessa domanda.
    examples = read_json(INPUT / "response_example.json")
    if examples["E0"]["case_id"] != examples["E4"]["case_id"]:
        raise ValueError("Le due risposte di esempio non appartengono alla stessa domanda")
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2))
    for ax, configuration, label, color in (
        (axes[0], "E0", "E0 · nessun documento", "#e6eef5"),
        (axes[1], "E4", f"E4 · {examples['E4']['source_count']} fonti nel prompt", "#e8f2ed"),
    ):
        source = examples[configuration]["answer_prefix"]
        clean = re.sub(r"\s+", " ", source).strip().replace("**", "").replace("`", "")
        cut = clean[:280].rsplit(" ", 1)[0]
        excerpt = textwrap.fill(
            cut + (" …" if examples[configuration]["answer_longer_than_excerpt"] else ""),
            width=82,
        )
        ax.set_facecolor(color)
        ax.text(0.04, 0.95, label, fontsize=12, weight="bold", va="top", transform=ax.transAxes)
        ax.text(0.04, 0.72, excerpt, fontsize=10.5, va="top", linespacing=1.4, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for edge in ax.spines.values():
            edge.set_visible(False)
    fig.suptitle("Due risposte alla stessa domanda")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "figure_5_3_response_example.png", dpi=180)

    # Figura 6.1: andamento di recall, MRR e hit rate al variare di K.
    metrics = (("mean_recall", "Recall@K"), ("mrr_at_k", "MRR@K"), ("hit_rate", "Hit@K"))
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.6), sharey=True, layout="constrained")
    for ax, (field, title) in zip(axes, metrics):
        for configuration in RETRIEVAL:
            ax.plot(
                K_VALUES,
                [value(retrieval_rows, configuration, field, K=k) for k in K_VALUES],
                marker="o", linewidth=1.8, label=configuration, color=COLORS[configuration],
            )
        ax.set_xticks(K_VALUES)
        ax.set_xlabel("K (documenti restituiti)")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        common(ax)
    axes[0].set_ylabel("Valore medio per caso")
    axes[-1].legend(loc="lower right", frameon=False, ncol=2)
    fig.suptitle("Recupero documentale sulle dieci domande di sviluppo", fontsize=12)
    save(fig, "figure_6_1_retrieval_k.png")

    # Figura 6.2: recall per forma della domanda.
    query_types = ("exact", "paraphrase", "distributed")
    counts = {
        row["query_type"]: row["n_cases"]
        for row in type_rows if row["configuration"] == "E1" and row["K"] == 5
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.1), layout="constrained")
    positions = list(range(len(query_types)))
    width = 0.18
    for index, configuration in enumerate(RETRIEVAL):
        ax.bar(
            [x + (index - 1.5) * width for x in positions],
            [value(type_rows, configuration, "mean_recall", K=5, query_type=query_type)
             for query_type in query_types],
            width, label=configuration, color=COLORS[configuration],
        )
    ax.set_xticks(positions, [f"{query_type}\nN={counts[query_type]}" for query_type in query_types])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall@5 medio")
    ax.set_xlabel("Tipologia di domanda")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    common(ax)
    fig.suptitle("Recall@5 per tipo di domanda", fontsize=11)
    save(fig, "figure_6_2_query_type.png")

    # Figura 6.4: durata della generazione e delle fasi di recupero.
    positions = list(range(len(CONFIGURATIONS)))
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.8), layout="constrained")
    axes[0].bar(
        positions, [value(latency_rows, c, "mean_ms", phase="generation") / 1000
                    for c in CONFIGURATIONS],
        color=[COLORS[c] for c in CONFIGURATIONS],
    )
    axes[0].set_xticks(positions, CONFIGURATIONS)
    axes[0].set_ylabel("Secondi medi per richiesta")
    axes[0].set_title("Generazione")
    common(axes[0])
    axes[1].bar(
        [x - 0.18 for x in positions],
        [value(latency_rows, c, "mean_ms", phase="retrieval") for c in CONFIGURATIONS],
        0.36, label="Retrieval", color="#286DA8",
    )
    axes[1].bar(
        [x + 0.18 for x in positions],
        [value(latency_rows, c, "mean_ms", phase="reranking") for c in CONFIGURATIONS],
        0.36, label="Reranking", color="#A64A69",
    )
    axes[1].set_xticks(positions, CONFIGURATIONS)
    axes[1].set_ylabel("Millisecondi medi per richiesta")
    axes[1].set_title("Recupero delle fonti")
    axes[1].legend(frameon=False)
    common(axes[1])
    fig.suptitle("Latenza media per configurazione sulle dieci domande di sviluppo", fontsize=11)
    save(fig, "figure_6_4_latency.png")

    # Figura 6.5: media dei token misurati per richiesta.
    input_tokens = [value(token_rows, c, "mean_input_tokens") for c in CONFIGURATIONS]
    output_tokens = [value(token_rows, c, "mean_output_tokens") for c in CONFIGURATIONS]
    fig, ax = plt.subplots(figsize=(7.8, 3.8), layout="constrained")
    ax.bar(positions, input_tokens, color="#286DA8", label="Input")
    ax.bar(positions, output_tokens, bottom=input_tokens, color="#DA8B1F", label="Output")
    ax.set_xticks(positions, CONFIGURATIONS)
    ax.set_ylabel("Token medi per richiesta")
    ax.set_xlabel("Configurazione")
    ax.legend(frameon=False)
    common(ax)
    fig.suptitle("Token di input e output per configurazione", fontsize=11)
    save(fig, "figure_6_5_tokens.png")

    # Figura 6.6: campioni GPU durante il run misurato.
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.8), layout="constrained")
    for ax, field, ylabel, title in (
        (axes[0], "mean_gpu_util_percent", "GPU media (%)", "Utilizzo GPU"),
        (axes[1], "max_gpu_memory_mib", "VRAM massima (MiB)", "Memoria GPU"),
    ):
        ax.bar(positions, [value(gpu_rows, c, field) for c in CONFIGURATIONS],
               color=[COLORS[c] for c in CONFIGURATIONS])
        ax.set_xticks(positions, CONFIGURATIONS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        common(ax)
    fig.suptitle("Campioni nvidia-smi ogni 1 s; intera GPU, Demo attiva", fontsize=11)
    save(fig, "figure_6_6_gpu.png")

    # Figura 6.7: relazione descrittiva tra recall e durata totale.
    fig, ax = plt.subplots(figsize=(7.1, 4.1), layout="constrained")
    for configuration in RETRIEVAL:
        recall = value(retrieval_rows, configuration, "mean_recall", K=5)
        total = value(latency_rows, configuration, "mean_ms", phase="total") / 1000
        ax.scatter(total, recall, s=95, color=COLORS[configuration])
        ax.annotate(configuration, (total, recall), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Latenza totale media (s/richiesta)")
    ax.set_ylabel("Recall@5 medio")
    ax.set_ylim(0, 1.05)
    common(ax)
    fig.suptitle("Recall@5 e latenza media per configurazione", fontsize=11)
    save(fig, "figure_6_7_retrieval_latency.png")

    # Figura 6.8: casi coperti solo in parte o privi di una fonte rilevante.
    partial, no_hit = [], []
    for configuration in RETRIEVAL:
        group = [row for row in case_rows if row["configuration"] == configuration and row["K"] == 5]
        partial.append(sum(0 < row["recall"] < 1 for row in group))
        no_hit.append(sum(row["hit"] == 0 for row in group))
    fig, ax = plt.subplots(figsize=(7.3, 3.8), layout="constrained")
    x = list(range(len(RETRIEVAL)))
    ax.bar([item - 0.18 for item in x], partial, 0.36, label="Copertura parziale", color="#DA8B1F")
    ax.bar([item + 0.18 for item in x], no_hit, 0.36, label="Nessuna fonte rilevante", color="#A64A69")
    ax.set_xticks(x, RETRIEVAL)
    ax.set_ylabel("Numero di casi su 10")
    ax.set_xlabel("Configurazione")
    ax.set_yticks(range(0, max(partial + no_hit) + 1))
    ax.legend(frameon=False)
    common(ax)
    fig.suptitle("Copertura dei documenti rilevanti nei primi cinque risultati", fontsize=11)
    save(fig, "figure_6_8_retrieval_errors.png")

    print(f"\nSalvate {len(created)} figure in {destination}")
    if show:
        plt.show()
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs", type=Path, metavar="ARCHIVIO", help="estrae gli input dalle evidenze operative")
    parser.add_argument("--no-show", action="store_true", help="salva le figure senza aprire finestre")
    parser.add_argument("--tables-only", action="store_true", help="ricalcola solo le tabelle")
    parser.add_argument("--output-root", type=Path, default=ROOT, help="cartella di destinazione (predefinita: accanto allo script)")
    args = parser.parse_args()
    if args.prepare_inputs:
        prepare_inputs(args.prepare_inputs)
        return
    manifest, cases, rankings, requests = validate_inputs()
    gpu_samples = read_csv(INPUT / "gpu_samples.csv")
    case_rows, retrieval_rows, type_rows = calculate_retrieval(cases, rankings)
    request_rows, latency_rows, token_rows, gpu_rows = calculate_generation(requests, gpu_samples)
    print(f"Run {manifest['run_id']} · modello {manifest['model']}")
    print(f"Input: {len(cases)} casi, {len(rankings)} graduatorie, {len(requests)} richieste, {len(gpu_samples)} campioni GPU")
    print_extra_measurements(cases, manifest)
    generate_tables(args.output_root, case_rows, retrieval_rows, type_rows,
                    request_rows, latency_rows, token_rows, gpu_rows)
    if not args.tables_only:
        generate_figures(args.output_root, cases, case_rows, retrieval_rows, type_rows,
                         latency_rows, token_rows, gpu_rows, not args.no_show)


if __name__ == "__main__":
    main()
